from __future__ import annotations

import numpy as np
import pytest
import torch

duckdb = pytest.importorskip("duckdb")

from probixi.ambigator import (  # noqa: E402
    Ambigator,
    _pair_cc_reference,
    _to_asu,
    ambiguity_operator,
    parse_operator,
    point_group_ops,
)

# Orders of the eleven chiral point groups and their Laue classes
ORDERS = {
    "1": 1,
    "2": 2,
    "222": 4,
    "4": 4,
    "422": 8,
    "3": 3,
    "321": 6,
    "312": 6,
    "6": 6,
    "622": 12,
    "23": 12,
    "432": 24,
    "-1": 2,
    "2/m": 4,
    "mmm": 8,
    "4/m": 8,
    "4/mmm": 16,
    "-3": 6,
    "-3m1": 12,
    "-31m": 12,
    "6/m": 12,
    "6/mmm": 24,
    "m-3": 24,
    "m-3m": 48,
}


def _apply(op, hkl):
    return tuple(sum(op[r][c] * hkl[c] for c in range(3)) for r in range(3))


@pytest.mark.parametrize("name,order", ORDERS.items())
def test_point_group_orders(name, order):
    ops = point_group_ops(name)
    assert len(ops) == order
    assert len(set(ops)) == order
    # closed under composition
    for a in ops:
        for b in ops:
            assert _apply(a, b[0]), (a, b)
    assert ((1, 0, 0), (0, 1, 0), (0, 0, 1)) in ops


def test_unique_axis_variants():
    assert point_group_ops("2_uaa") != point_group_ops("2_uab")
    assert point_group_ops("2") == point_group_ops("2_uac")
    assert len(point_group_ops("2/m_uab")) == 4


def test_parse_operator():
    assert parse_operator("k,h,-l") == ((0, 1, 0), (1, 0, 0), (0, 0, -1))
    assert parse_operator("k, -h-k, l") == ((0, 1, 0), (-1, -1, 0), (0, 0, 1))
    with pytest.raises(ValueError):
        parse_operator("h,k")
    with pytest.raises(ValueError):
        parse_operator("h,k,z")


@pytest.mark.parametrize(
    "symmetry,apparent", [("3", "321"), ("3", "312"), ("4", "422"), ("6", "622")]
)
def test_ambiguity_operator_is_an_involution_outside_the_target(symmetry, apparent):
    op = ambiguity_operator(symmetry, apparent)
    target = point_group_ops(symmetry)
    assert op not in target
    # applying it twice lands back inside the target group
    squared = tuple(
        tuple(sum(op[r][t] * op[t][c] for t in range(3)) for c in range(3))
        for r in range(3)
    )
    assert squared in target


def test_ambiguity_operator_rejects_bad_pairs():
    with pytest.raises(ValueError, match="not a subgroup"):
        ambiguity_operator("4", "321")
    with pytest.raises(ValueError, match="indexing choices"):
        ambiguity_operator("3", "622")


def test_asu_is_constant_over_an_orbit():
    ops = point_group_ops("321")
    hkl = torch.randint(-11, 12, (500, 3))
    asu = _to_asu(hkl, ops)
    assert torch.equal(_to_asu(asu, ops), asu)
    for op in ops:
        moved = torch.stack(
            [sum(op[r][c] * hkl[:, c] for c in range(3)) for r in range(3)], dim=1
        )
        assert torch.equal(_to_asu(moved, ops), asu)


def test_pair_cc_reference_matches_numpy_pearson():
    # two crystals sharing three of four reflections, one resolution group
    offsets = torch.tensor([0, 4, 8], dtype=torch.int32)
    keys = torch.tensor([1, 3, 5, 7, 3, 5, 6, 7], dtype=torch.int32)
    vals = torch.tensor([10.0, 20.0, 30.0, 40.0, 25.0, 31.0, 9.0, 44.0])
    groups = torch.zeros(8, dtype=torch.int32)
    partners = torch.tensor([[1], [0]], dtype=torch.int32)
    cc, novl = _pair_cc_reference(offsets, keys, vals, groups, keys, vals, partners, 1)
    assert novl.tolist() == [[3], [3]]
    a = torch.tensor([20.0, 30.0, 40.0])  # keys 3, 5, 7 of crystal 0
    b = torch.tensor([25.0, 31.0, 44.0])  # same keys of crystal 1
    want = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    assert cc[0, 0] == pytest.approx(want, abs=1e-5)
    assert cc[1, 0] == pytest.approx(want, abs=1e-5)


def _twinned_database(path, n_crystals=180, seed=5):
    """A P3 dataset whose crystals are randomly indexed in one of two choices."""
    ops = point_group_ops("3")
    operator = ambiguity_operator("3", "321")
    gen = torch.Generator().manual_seed(seed)
    pool = torch.tensor(
        [
            (h, k, ell)
            for h in range(-8, 9)
            for k in range(-8, 9)
            for ell in range(0, 8)
            if 0 < h * h + k * k + ell * ell < 70
        ]
    )
    # one intensity per P3-unique reflection, and the same list in the other choice
    _, orbit = torch.unique(_to_asu(pool, ops), dim=0, return_inverse=True)
    truth = torch.rand(int(orbit.max()) + 1, generator=gen) * 400.0 + 20.0
    twin = torch.stack(
        [sum(operator[r][c] * pool[:, c] for c in range(3)) for r in range(3)], dim=1
    )
    # 0.6 to 5 nm^-1, so all three of ambigator's resolution shells are populated
    recip = 0.6 * pool.float().square().sum(1).sqrt()
    take = len(pool) // 2

    flipped, ids, hkl, value, resolution = {}, [], [], [], []
    for index in range(n_crystals):
        cid = f"c{index:04d}"
        flip = bool(torch.rand(1, generator=gen) > 0.5)
        flipped[cid] = flip
        scale = 0.6 + 0.8 * float(torch.rand(1, generator=gen))
        pick = torch.randperm(len(pool), generator=gen)[:take]
        noise = 1.0 + 0.12 * torch.randn(take, generator=gen)
        ids.extend([cid] * take)
        hkl.append((twin if flip else pool)[pick])
        value.append(truth[orbit[pick]] * scale * noise)
        resolution.append(recip[pick])

    hkl = torch.cat(hkl)
    reflections = {
        "crystal_id": np.array(ids, dtype=object),
        **{axis: hkl[:, i].numpy() for i, axis in enumerate("hkl")},
        "intensity": torch.cat(value).numpy(),
        "resolution_nm_inv": torch.cat(resolution).numpy(),
    }
    identity = np.tile(np.eye(3, dtype=np.float64).ravel(), (n_crystals, 1))
    crystals = {
        "crystal_id": np.array(sorted(flipped), dtype=object),
        **{
            f"{s}star_{x}": identity[:, 3 * i + j]
            for i, s in enumerate("abc")
            for j, x in enumerate("xyz")
        },
        **{
            name: np.full(n_crystals, np.nan)
            for name in ("cell_a_A", "cell_b_A", "cell_c_A")
            + ("cell_alpha_deg", "cell_beta_deg", "cell_gamma_deg")
        },
    }
    conn = duckdb.connect(str(path))
    for name, columns in (("reflections", reflections), ("crystals", crystals)):
        conn.register(f"_{name}", columns)
        conn.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
    conn.close()
    return flipped


def test_resolve_recovers_the_two_indexing_choices(tmp_path):
    path = tmp_path / "run.duckdb"
    flipped = _twinned_database(path)
    result = Ambigator(path, symmetry="3", apparent="321").resolve(ncorr=80, seed=1)

    assert len(result) == len(flipped)
    assert result.n_unused == 0
    # converged: nothing flipped on the last pass, and f overtook g
    flips, mean_f, mean_g = result.history[-1]
    assert flips == 0
    assert mean_f > mean_g

    got = dict(zip(result.crystal_ids, result.assignments.tolist()))
    agree = sum(got[cid] == int(flip) for cid, flip in flipped.items())
    # the labelling is arbitrary, so either side may be called "reindexed"
    assert max(agree, len(got) - agree) == len(got)


def test_resolve_sorts_both_reflection_runs(tmp_path):
    # the binary search in either backend needs each crystal's keys ascending
    path = tmp_path / "run.duckdb"
    _twinned_database(path, n_crystals=20)
    amb = Ambigator(path, symmetry="3", apparent="321")
    _, offsets, plain, reindexed, n_groups = amb._load()
    assert n_groups == 3
    for keys, _, _ in (plain, reindexed):
        for lo, hi in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
            run = keys[lo:hi]
            assert torch.equal(run, run.sort().values)


def test_reindex_writes_assignments_and_transforms_hkl(tmp_path):
    path = tmp_path / "run.duckdb"
    _twinned_database(path, n_crystals=40)
    amb = Ambigator(path, symmetry="3", apparent="321")
    result = amb.resolve(ncorr=30, seed=2)

    conn = duckdb.connect(str(path))
    before = dict(
        conn.execute(
            "SELECT crystal_id, sum(h * 1000 + k) FROM reflections GROUP BY 1"
        ).fetchall()
    )
    conn.close()

    amb.reindex(result)

    conn = duckdb.connect(str(path))
    stored = dict(
        conn.execute("SELECT crystal_id, assignment FROM ambiguity").fetchall()
    )
    after = dict(
        conn.execute(
            "SELECT crystal_id, sum(h * 1000 + k) FROM reflections GROUP BY 1"
        ).fetchall()
    )
    # a* started as the identity in nm^-1, so a flipped crystal's real-space
    # basis is inv(M^T): edges of sqrt(2), 1 and 1 nm with gamma = 135 degrees
    cells = conn.execute(
        "SELECT DISTINCT round(cell_a_A, 3), round(cell_b_A, 3), round(cell_c_A, 3), "
        "round(cell_alpha_deg, 2), round(cell_beta_deg, 2), round(cell_gamma_deg, 2) "
        "FROM crystals WHERE crystal_id IN "
        "(SELECT crystal_id FROM ambiguity WHERE assignment = 1)"
    ).fetchall()
    kept = conn.execute(
        "SELECT DISTINCT cell_a_A IS NULL OR isnan(cell_a_A) FROM crystals "
        "WHERE crystal_id IN (SELECT crystal_id FROM ambiguity WHERE assignment = 0)"
    ).fetchall()
    conn.close()

    assert stored == dict(zip(result.crystal_ids, result.assignments.tolist()))
    assert sum(stored.values()) == result.n_reindexed
    for cid, assignment in stored.items():
        if assignment:
            assert after[cid] != before[cid]
        else:
            assert after[cid] == before[cid]
    assert cells == [(14.142, 10.0, 10.0, 90.0, 90.0, 135.0)]
    assert kept == [(True,)]  # unflipped crystals were left alone


def test_requires_exactly_one_of_apparent_or_operator(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        Ambigator(tmp_path / "x.duckdb", symmetry="3")
    with pytest.raises(ValueError, match="exactly one"):
        Ambigator(
            tmp_path / "x.duckdb", symmetry="3", apparent="321", operator="k,h,-l"
        )


def test_operator_given_directly_matches_the_derived_one(tmp_path):
    path = tmp_path / "run.duckdb"
    _twinned_database(path, n_crystals=20)
    derived = Ambigator(path, symmetry="3", apparent="321")
    direct = Ambigator(path, symmetry="3", operator="-h-k,k,-l")
    assert direct.operator == derived.operator


def test_all_twin_proof_reflections_fail_loudly(tmp_path):
    # the identity leaves every reflection where it is, so nothing is informative
    path = tmp_path / "run.duckdb"
    _twinned_database(path, n_crystals=8)
    amb = Ambigator(path, symmetry="3", operator="h,k,l")
    with pytest.raises(ValueError, match="twin-proof"):
        amb.resolve()


def test_reindex_to_an_output_copy_leaves_the_original_alone(tmp_path):
    path = tmp_path / "run.duckdb"
    copy = tmp_path / "detwinned.duckdb"
    _twinned_database(path, n_crystals=40)
    amb = Ambigator(path, symmetry="3", apparent="321")
    result = amb.resolve(ncorr=30, seed=2)

    hkl = "SELECT crystal_id, sum(h * 1000 + k) FROM reflections GROUP BY 1 ORDER BY 1"
    conn = duckdb.connect(str(path))
    before = conn.execute(hkl).fetchall()
    conn.close()

    assert amb.reindex(result, output=copy) == copy
    assert result.n_reindexed  # the copy is only interesting if something moved

    conn = duckdb.connect(str(path))
    assert conn.execute(hkl).fetchall() == before
    assert not conn.execute(
        "SELECT 1 FROM duckdb_tables() WHERE table_name = 'ambiguity'"
    ).fetchall()
    conn.close()

    conn = duckdb.connect(str(copy))
    assert conn.execute(hkl).fetchall() != before
    assert conn.execute("SELECT count(*) FROM ambiguity").fetchone()[0] == len(result)
    conn.close()


def test_reindex_output_pointing_at_the_source_stays_in_place(tmp_path):
    path = tmp_path / "run.duckdb"
    _twinned_database(path, n_crystals=20)
    amb = Ambigator(path, symmetry="3", apparent="321")
    result = amb.resolve(ncorr=15, seed=2)
    assert amb.reindex(result, output=tmp_path / "." / "run.duckdb") == path


def test_reindex_output_discards_a_stale_wal_sidecar(tmp_path):
    path = tmp_path / "run.duckdb"
    copy = tmp_path / "detwinned.duckdb"
    _twinned_database(path, n_crystals=20)
    copy.write_bytes(b"")
    stale = tmp_path / "detwinned.duckdb.wal"
    stale.write_bytes(b"not a real write-ahead log")
    amb = Ambigator(path, symmetry="3", apparent="321")
    amb.reindex(amb.resolve(ncorr=15, seed=2), output=copy)
    assert not stale.exists()
    conn = duckdb.connect(str(copy))
    assert conn.execute("SELECT count(*) FROM ambiguity").fetchone()[0] == 20
    conn.close()


def _cli(*args):
    from click.testing import CliRunner

    from probixi.ambigator import main

    return CliRunner().invoke(main, [str(a) for a in args])


def test_cli_rewrites_in_place_by_default(tmp_path):
    path = tmp_path / "run.duckdb"
    _twinned_database(path, n_crystals=40)
    out = _cli("-i", path, "-y", "3", "-w", "321", "--ncorr", 30)
    assert out.exit_code == 0, out.output
    assert "Ambiguity operator: -h-k,k,-l" in out.output
    assert "pass 1:" in out.output
    assert f"crystals in {path}" in out.output
    conn = duckdb.connect(str(path))
    assert conn.execute("SELECT count(*) FROM ambiguity").fetchone()[0] == 40
    conn.close()


def test_cli_output_writes_a_copy(tmp_path):
    path = tmp_path / "run.duckdb"
    copy = tmp_path / "detwinned.duckdb"
    _twinned_database(path, n_crystals=40)
    out = _cli("-i", path, "-y", "3", "-w", "321", "--ncorr", 30, "-o", copy)
    assert out.exit_code == 0, out.output
    assert copy.exists()
    for db, expected in ((path, False), (copy, True)):
        conn = duckdb.connect(str(db))
        found = bool(
            conn.execute(
                "SELECT 1 FROM duckdb_tables() WHERE table_name = 'ambiguity'"
            ).fetchall()
        )
        conn.close()
        assert found is expected


def test_cli_quiet_reports_only_the_outcome(tmp_path):
    path = tmp_path / "run.duckdb"
    _twinned_database(path, n_crystals=20)
    out = _cli("-i", path, "-y", "3", "-w", "321", "--ncorr", 15, "--quiet")
    assert out.exit_code == 0, out.output
    assert out.output.splitlines() == [out.output.strip()]


def test_cli_requires_input_and_both_point_groups(tmp_path):
    path = tmp_path / "run.duckdb"
    _twinned_database(path, n_crystals=20)
    for args, missing in (
        (("-y", "3", "-w", "321"), "'-i' / '--input'"),
        (("-i", path, "-w", "321"), "'-y' / '--symmetry'"),
        (("-i", path, "-y", "3"), "'-w' / '--apparent'"),
    ):
        out = _cli(*args)
        assert out.exit_code == 2
        assert f"Missing option {missing}" in out.output


def test_cli_reports_bad_input_without_a_traceback(tmp_path):
    path = tmp_path / "run.duckdb"
    _twinned_database(path, n_crystals=20)
    bad_group = _cli("-i", path, "-y", "3", "-w", "999")
    assert bad_group.exit_code == 2
    assert "unknown point group '999'" in bad_group.output
    assert bad_group.exc_info[0] is SystemExit

    # a cutoff that admits nothing leaves the database unusable, not misused
    empty = _cli("-i", path, "-y", "3", "-w", "321", "--lowres", 0.1)
    assert empty.exit_code == 1
    assert "no usable reflections" in empty.output
    assert empty.exc_info[0] is SystemExit
