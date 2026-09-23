"""Run-level re-centring of the target cell (Probixi.cell_calibrate).

The file cell only has to bootstrap the first lattices; afterwards the seed
basis, inlier tolerance and cell-refinement prior follow the data. Motivated by
b2AR, where a Millepede camera length left the cell file ~1% inconsistent with
the geometry and the known-cell seeder lost every peak inside 7 A.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import sim
import torch

from probixi import Probixi
from probixi.indexer import FrameIndexResult, FrameIndexStream, IndexStats
from probixi.indexer.indexer import Indexer, RefineConfig, SeedConfig
from probixi.io.cell import CellParams, median_cell

SEED = SeedConfig(n_directions=1500, n_spin=60, top_directions=12, max_candidates=32)
REFINE_FIXED = RefineConfig(max_iters=150, reassign_every=10, cell=False)


def _scaled(cell: CellParams, s: float) -> CellParams:
    return CellParams(
        cell.a * s,
        cell.b * s,
        cell.c * s,
        cell.alpha,
        cell.beta,
        cell.gamma,
        lattice_type=cell.lattice_type,
        unique_axis=cell.unique_axis,
        centering=cell.centering,
    )


def _rel(x: float, y: float) -> float:
    return abs(x / y - 1.0)


def test_median_cell_is_componentwise_and_keeps_template_metadata():
    template = CellParams(
        10, 20, 30, math.pi / 2, 1.8, math.pi / 2, "monoclinic", "b", "C"
    )
    cells = [
        CellParams(9.0, 21.0, 30.5, math.pi / 2, 1.81, math.pi / 2),
        CellParams(10.0, 19.0, 29.0, math.pi / 2, 1.79, math.pi / 2),
        CellParams(11.0, 20.0, 31.0, math.pi / 2, 1.85, math.pi / 2),
    ]
    m = median_cell(cells, template=template)
    assert (m.a, m.b, m.c) == (10.0, 20.0, 30.5)
    assert m.beta == pytest.approx(1.81)
    assert (m.lattice_type, m.unique_axis, m.centering) == ("monoclinic", "b", "C")
    with pytest.raises(ValueError):
        median_cell([])


def test_set_target_cell_rebuilds_basis_tolerance_and_keeps_metadata(
    geometry_dict, cell
):
    idxr = Indexer(geometry_dict, cell, seed=SEED)
    B0, tol0 = idxr.B_target.clone(), idxr.q_tolerance
    bare = CellParams(
        cell.a * 1.01, cell.b * 1.01, cell.c * 1.01, cell.alpha, cell.beta, cell.gamma
    )
    idxr.set_target_cell(bare)
    # reciprocal basis and the derived |q| tolerance both shrink by 1 %
    assert torch.allclose(idxr.B_target, B0 / 1.01, rtol=1e-5, atol=0)
    assert idxr.q_tolerance == pytest.approx(tol0 / 1.01, rel=1e-5)
    # metadata carried over from the previous target when the new cell has none
    assert idxr.target_cell.lattice_type == cell.lattice_type
    assert idxr.target_cell.centering == cell.centering
    # a fixed SeedConfig.q_tolerance is left alone
    fixed = Indexer(geometry_dict, cell, seed=SeedConfig(q_tolerance=0.0123))
    fixed.set_target_cell(bare)
    assert fixed.q_tolerance == 0.0123


def test_recentring_moves_recovered_cell_from_prior_to_truth(geometry_dict, cell):
    # With per-crystal cell refinement off, the recovered cell IS the target:
    # a 1 % wrong target yields a 1 % wrong cell; re-centring on the truth fixes it.
    U = sim.proper_rotation(0, max_angle_deg=8.0)
    positions, _ = sim.lattice_peaks(geometry_dict, cell, U)
    wrong = _scaled(cell, 1.01)
    idxr = Indexer(geometry_dict, wrong, seed=SEED, refine=REFINE_FIXED)
    before = idxr.index_frames({0: positions})
    if before:
        assert _rel(before[0].cell.a, wrong.a) < 1e-3
    idxr.set_target_cell(cell)
    after = idxr.index_frames({0: positions})
    assert after, "re-centred indexer must solve the clean simulated frame"
    r = after[0]
    for got, true in ((r.cell.a, cell.a), (r.cell.b, cell.b), (r.cell.c, cell.c)):
        assert _rel(got, true) < 1e-3
    assert r.n_indexed >= 0.9 * positions.shape[0]
    if before:
        assert r.n_indexed >= before[0].n_indexed


def _fake_pipeline(monkeypatch, cells_per_frame, **opts):
    """A Probixi whose indexer yields canned crystals; records set_target_cell calls."""
    p = object.__new__(Probixi)
    p._frame_scales = {}
    p._calibration_options = {}
    p._calibration_boundary = 0
    p.threshold_calibration = SimpleNamespace(threshold=5.0)
    p.cell_calibrate = opts.get("cell_calibrate", True)
    p.cell_calibrate_after = opts.get("after", 3)
    p.cell_calibrate_rounds = opts.get("rounds", 2)
    p.cell_calibrate_warn = opts.get("warn", 0.005)
    p.cell_calibrate_alpha = 1e-3
    p._cell_origin = None
    p.screened_frames = []
    p.cell_calibrations = []
    target = CellParams(
        10.0, 20.0, 30.0, math.pi / 2, 1.8, math.pi / 2, "monoclinic", "b", "C"
    )
    calls: list[CellParams] = []
    p_of = opts.get("p_of", lambda c: 1e-6)

    class FakeIndexer:
        def __init__(self):
            self.target_cell = target

        def set_target_cell(self, cell):
            self.target_cell = cell
            calls.append(cell)

        def _cell_matches_target(self, cell, target=None):
            t = target or self.target_cell
            return all(
                abs(x / y - 1) <= 0.05
                for x, y in ((cell.a, t.a), (cell.b, t.b), (cell.c, t.c))
            )

        def index_frame_stream(
            self, peaks, batch_size=8, bright_threshold=None, enrich_alpha=None
        ):
            stats = IndexStats()

            def generate():
                for i, _ in peaks:
                    crystals = [
                        SimpleNamespace(
                            cell=c, enrich_p=p_of(c), scale=None, scale_sigma=None
                        )
                        for c in cells_per_frame[i]
                    ]
                    stats.frames += 1
                    yield FrameIndexResult(
                        i, crystals, torch.empty((0, 2)), torch.empty(0)
                    )

            return FrameIndexStream(generate(), stats)

    monkeypatch.setattr(
        p,
        "peak_stream",
        lambda frames, start_index=0, update_noise=True: enumerate(frames, start_index),
    )
    p.indexer = FakeIndexer()
    return p, calls


def test_pipeline_recentres_on_median_after_n_lattices_for_n_rounds(monkeypatch):
    def c(a):
        return CellParams(a, 20.0, 30.0, math.pi / 2, 1.8, math.pi / 2)

    # frame 0: 2 lattices, frame 1: none, frame 2: 1 -> first round after frame 2;
    # frames 3-5: one each -> second round after frame 5; frames 6-7 pooled no more
    cells = {
        0: [c(9.6), c(9.8)],
        1: [],
        2: [c(9.7)],
        3: [c(9.8)],
        4: [c(9.9)],
        5: [c(9.85)],
        6: [c(1.0)],
        7: [c(1.0)],
    }
    p, calls = _fake_pipeline(monkeypatch, cells, after=3, rounds=2)
    with pytest.warns(UserWarning, match="re-centred"):
        out = list(
            p.index_frame_stream((torch.zeros((1, 1)) for _ in range(8)), batch_size=3)
        )
    assert [r.frame_index for r in out] == list(range(8))
    assert [round(k.a, 6) for k in calls] == [
        9.7,
        9.85,
    ]  # medians of {9.6,9.8,9.7} then {9.8,9.9,9.85}
    assert all(k.lattice_type == "monoclinic" and k.centering == "C" for k in calls)
    assert len(p.cell_calibrations) == 2
    first = p.cell_calibrations[0]
    assert (
        first.n_lattices == 3
        and first.previous.a == 10.0
        and first.cell.a == 9.7
        and first.applied
    )
    assert first.edge_shift == pytest.approx(0.03)
    assert first.angle_shift == 0.0
    # every crystal still streamed through, including the ones after the last round
    assert sum(len(r.crystals) for r in out) == 8


def test_pipeline_leaves_target_alone_when_disabled_or_starved(monkeypatch):
    def c(a):
        return CellParams(a, 20.0, 30.0, math.pi / 2, 1.8, math.pi / 2)

    cells = {i: [c(9.0)] for i in range(4)}
    p, calls = _fake_pipeline(monkeypatch, cells, cell_calibrate=False)
    list(p.index_frame_stream((torch.zeros((1, 1)) for _ in range(4)), batch_size=2))
    assert calls == [] and p.cell_calibrations == []
    p, calls = _fake_pipeline(
        monkeypatch, cells, after=10
    )  # never reaches the pool size
    list(p.index_frame_stream((torch.zeros((1, 1)) for _ in range(4)), batch_size=2))
    assert calls == [] and p.cell_calibrations == []


def test_pipeline_refuses_a_move_outside_the_file_cells_window(monkeypatch):
    def c(a):
        return CellParams(a, 20.0, 30.0, math.pi / 2, 1.8, math.pi / 2)

    # 8 % off: refused, round consumed, target untouched; later frames still stream
    cells = {i: [c(9.2)] for i in range(6)}
    p, calls = _fake_pipeline(monkeypatch, cells, after=3, rounds=1)
    with pytest.warns(UserWarning, match="NOT re-centred"):
        out = list(
            p.index_frame_stream((torch.zeros((1, 1)) for _ in range(6)), batch_size=2)
        )
    assert calls == [] and len(out) == 6
    assert len(p.cell_calibrations) == 1 and not p.cell_calibrations[0].applied
    assert p.indexer.target_cell.a == 10.0


def test_pipeline_pools_only_gate_passing_lattices(monkeypatch):
    def c(a):
        return CellParams(a, 20.0, 30.0, math.pi / 2, 1.8, math.pi / 2)

    # the 9.0 lattices fail the enrichment gate (or carry no p) and never enter the pool
    cells = {0: [c(9.0), c(9.8)], 1: [c(9.0), c(9.9)], 2: [c(9.0), c(9.7)]}
    p_of = lambda k: (0.5 if k.a == 9.0 else 1e-9)
    p, calls = _fake_pipeline(monkeypatch, cells, after=3, p_of=p_of)
    list(p.index_frame_stream((torch.zeros((1, 1)) for _ in range(3)), batch_size=1))
    assert [round(k.a, 6) for k in calls] == [9.8]
    p, calls = _fake_pipeline(monkeypatch, cells, after=3, p_of=lambda k: None)
    list(p.index_frame_stream((torch.zeros((1, 1)) for _ in range(3)), batch_size=1))
    assert calls == []


def test_calibration_prepass_recentres_before_the_first_frame(monkeypatch):
    def c(a):
        return CellParams(a, 20.0, 30.0, math.pi / 2, 1.8, math.pi / 2)

    cells = {i: [c(9.7 + 0.01 * i)] for i in range(5)}
    p, calls = _fake_pipeline(monkeypatch, cells, after=30)  # needs >= 3 lattices
    p.screened_frames = [7]
    p._calibrate_cell([torch.zeros((1, 1)) for _ in range(5)])
    assert [round(k.a, 6) for k in calls] == [9.72]
    assert p.cell_calibrations[0].n_lattices == 5 and p.screened_frames == [7]
    p, calls = _fake_pipeline(monkeypatch, cells, after=100)  # needs >= 10: starved
    p._calibrate_cell([torch.zeros((1, 1)) for _ in range(5)])
    assert calls == [] and p.cell_calibrations == []
