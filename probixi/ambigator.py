"""Resolve indexing ambiguities in a probixi DuckDB run.

The algorithm is ``ambigator``, created by Thomas White as part of CrystFEL,
which implements the method of W. Brehm and K. Diederichs, "Breaking the
indexing ambiguity in serial crystallography", Acta Cryst. D70 (2014) 101-109.
This module reads the database :class:`~probixi.io.db.DuckDBOffloader` writes.

Differences from the CrystFEL program:

* Assignments are updated simultaneously rather than in place during a pass,
  because the pass is a single vectorised reduction.
* ``ncorr`` partners per crystal are drawn once and those with fewer than four
  reflections in common are dropped, rather than drawing until ``ncorr``
  usable partners are found.
* A pair is usable when four reflections are shared across all resolution
  groups, not when four are shared in the last group alone.
* The correlation matrix, f/g graph and assignment files are not written; the
  per-pass means are returned in :attr:`AmbigatorResult.history` instead.
"""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import click
import duckdb
import numpy as np
import torch
from torch import Tensor

from .kernels import select

PathLike = Union[str, Path]

_RADIX = 1 << 21
_HALF = _RADIX // 2
_PAD = 0x7FFFFFFF
_MIN_COMMON = 4

# CC shells as 1/d in nm^-1
_GROUPS = ((None, 1.0), (2.0, 4.0), (4.0, None))

_AXES = {"h": 0, "k": 1, "l": 2}
_TERM = re.compile(r"([+-]?)(\d*)([hkl])")
_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_INVERSION = "-h,-k,-l"

# Generators of the point groups
_GENERATORS = {
    "1": (),
    "2_uaa": ("h,-k,-l",),
    "2_uab": ("-h,k,-l",),
    "2_uac": ("-h,-k,l",),
    "222": ("-h,-k,l", "h,-k,-l"),
    "4": ("-k,h,l",),
    "422": ("-k,h,l", "h,-k,-l"),
    "3": ("k,-h-k,l",),
    "321": ("k,-h-k,l", "k,h,-l"),
    "312": ("k,-h-k,l", "-k,-h,-l"),
    "6": ("-k,h+k,l",),
    "622": ("-k,h+k,l", "k,h,-l"),
    "23": ("-h,-k,l", "h,-k,-l", "l,h,k"),
    "432": ("-k,h,l", "l,h,k"),
}
_LAUE = {
    "-1": "1",
    "2/m": "2",
    "mmm": "222",
    "4/m": "4",
    "4/mmm": "422",
    "-3": "3",
    "-3m1": "321",
    "-31m": "312",
    "6/m": "6",
    "6/mmm": "622",
    "m-3": "23",
    "m-3m": "432",
}


def parse_operator(text: str) -> tuple[tuple[int, ...], ...]:
    """Parse ``"k,h,-l"`` into M with (h', k', l') = M @ (h, k, l)."""
    parts = text.strip().lower().replace(" ", "").replace("*", "").split(",")
    if len(parts) != 3:
        raise ValueError(f"operator needs three components: {text!r}")
    rows = []
    for part in parts:
        if not part or _TERM.sub("", part):
            raise ValueError(f"cannot parse operator component {part!r}")
        row = [0, 0, 0]
        for sign, digits, axis in _TERM.findall(part):
            row[_AXES[axis]] += (-1 if sign == "-" else 1) * int(digits or 1)
        rows.append(tuple(row))
    return tuple(rows)


def _multiply(a, b):
    return tuple(
        tuple(sum(a[r][t] * b[t][c] for t in range(3)) for c in range(3))
        for r in range(3)
    )


def point_group_ops(name: str) -> list[tuple[tuple[int, ...], ...]]:
    """Operations of a Hermann-Mauguin point group"""
    base, _, axis = name.strip().replace(" ", "").partition("_ua")
    centric = base in _LAUE
    base = _LAUE.get(base, base)
    if base == "2":
        base = f"2_ua{axis.lower() or 'c'}"
    elif axis and axis.lower() != "c":
        raise ValueError(f"point group {name!r} has no unique axis choice")
    if base not in _GENERATORS:
        raise ValueError(
            f"unknown point group {name!r}; "
            f"known: {', '.join(sorted(_GENERATORS) + sorted(_LAUE))}"
        )
    generators = [parse_operator(g) for g in _GENERATORS[base]]
    if centric:
        generators.append(parse_operator(_INVERSION))
    ops = {_IDENTITY}
    frontier = [_IDENTITY]
    while frontier:
        op = frontier.pop()
        for gen in generators:
            product = _multiply(gen, op)
            if product not in ops:
                ops.add(product)
                frontier.append(product)
        if len(ops) > 48:
            raise ValueError(f"point group {name!r} does not close")
    return sorted(ops)


def ambiguity_operator(symmetry: str, apparent: str):
    """The operation relating the two indexing choices of ``symmetry``."""
    target = set(point_group_ops(symmetry))
    lattice = point_group_ops(apparent)
    if not target.issubset(lattice):
        raise ValueError(f"{symmetry!r} is not a subgroup of {apparent!r}")
    index = len(lattice) // len(target)
    if index != 2:
        raise ValueError(
            f"{apparent!r}/{symmetry!r} has {index} indexing choices; "
            "ambigator resolves exactly two -- pick a different apparent group"
        )
    return next(op for op in lattice if op not in target)


@dataclass(frozen=True)
class AmbigatorResult:
    """Outcome of :meth:`Ambigator.resolve`.

    Parameters
    ----------
    crystal_ids : list of str
        Database ``crystals.crystal_id`` in the order of ``assignments``.
    assignments : Tensor
        (n_crystals,) uint8 of 0 (keep) or 1 (apply the ambiguity operator).
    history : list of tuple
        Per pass, ``(n_flipped, mean_f, mean_g)``. ``f`` rising above ``g`` as
        ``n_flipped`` reaches zero is convergence; the two staying together
        means there was no real ambiguity.
    n_unused : int
        Crystals with no usable correlation, left on their starting side.
    """

    crystal_ids: list[str]
    assignments: Tensor
    history: list[tuple[int, float, float]]
    n_unused: int

    def __len__(self) -> int:
        return len(self.crystal_ids)

    @property
    def n_reindexed(self) -> int:
        """Crystals assigned to the reindexed side."""
        return int(self.assignments.sum())


class Ambigator:
    """Resolve the indexing ambiguity of the crystals in a probixi database.

    Parameters
    ----------
    path : str or Path
        A ``.duckdb`` file written by :class:`~probixi.io.db.DuckDBOffloader`.
        Only ``reflections`` is read; :meth:`reindex` also writes ``crystals``.
    symmetry : str
        Actual point group of the structure, e.g. ``"3"``.
    apparent : str, optional
        Apparent (lattice) point group, e.g. ``"321"``. The ambiguity operator
        is derived from it and ``symmetry``.
    operator : str, optional
        The ambiguity operator given directly, e.g. ``"k,h,-l"``. Mutually
        exclusive with ``apparent``; one of the two is required.
    lowres, highres : float, optional
        Resolution limits in Angstrom. Setting either replaces the three
        automatic resolution shells with a single one over the given range.
    device : torch.device, optional
        Defaults to CUDA when available, else CPU.
    """

    def __init__(
        self,
        path: PathLike,
        *,
        symmetry: str,
        apparent: Optional[str] = None,
        operator: Optional[str] = None,
        lowres: Optional[float] = None,
        highres: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        if (apparent is None) == (operator is None):
            raise ValueError("give exactly one of 'apparent' or 'operator'")
        self.path = Path(path)
        self.symmetry = symmetry
        self.operator = (
            parse_operator(operator)
            if operator is not None
            else ambiguity_operator(symmetry, apparent)
        )
        # database holds 1/d in nm^-1 NOT A
        self._rmin = 0.0 if lowres is None else 10.0 / float(lowres)
        self._rmax = math.inf if highres is None else 10.0 / float(highres)
        self._auto_res = lowres is None and highres is None
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @torch.no_grad()
    def resolve(
        self, *, iterations: int = 6, ncorr: int = 1000, seed: int = 0
    ) -> AmbigatorResult:
        """Assign each crystal to one of the two indexing choices.

        Parameters
        ----------
        iterations : int, default 6
            Refinement passes; stops early once no crystal changes side.
        ncorr : int, default 1000
            Partners correlated per crystal, capped at the number available.
            Cost is ``O(n_crystals * ncorr)``.
        seed : int, default 0
            Seeds the starting assignments and the partner draw.

        Returns
        -------
        AmbigatorResult
        """
        ids, offsets, plain, reindexed, n_groups = self._load()
        n = len(ids)
        if n < 2:
            raise ValueError(f"{self.path} holds {n} crystals; need at least two")
        ncorr = max(1, min(int(ncorr), n - 1))

        gen = torch.Generator().manual_seed(int(seed))

        partners = torch.randint(0, n - 1, (n, ncorr), generator=gen)
        partners += partners >= torch.arange(n).unsqueeze(1)
        partners = partners.to(self.device, torch.int32)
        assignments = (torch.rand(n, generator=gen) > 0.5).to(self.device)

        cc, novl = self._correlate(offsets, plain, plain, partners, n_groups)
        cc_r, novl_r = self._correlate(offsets, reindexed, plain, partners, n_groups)
        valid = novl >= _MIN_COMMON
        valid_r = novl_r >= _MIN_COMMON

        history: list[tuple[int, float, float]] = []
        n_unused = 0
        for _ in range(iterations):
            assignments, flipped, mean_f, mean_g, n_unused = _pass(
                cc, cc_r, valid, valid_r, partners, assignments
            )
            history.append((flipped, mean_f, mean_g))
            if flipped == 0:
                break
        return AmbigatorResult(
            crystal_ids=ids,
            assignments=assignments.to("cpu", torch.uint8),
            history=history,
            n_unused=n_unused,
        )

    def reindex(
        self, result: AmbigatorResult, output: Optional[PathLike] = None
    ) -> Path:
        """Apply ``result`` to the database, in place.

        Every assignment lands in an ``ambiguity`` table; the reindexed side
        also has its ``reflections`` hkl and its ``crystals`` reciprocal axes
        and cell parameters transformed. Not idempotent.

        Parameters
        ----------
        result : AmbigatorResult
            From :meth:`resolve` on this same database.
        output : str or Path, optional
            Reindex a copy written here instead, leaving the original alone.
            Overwritten if it exists.

        Returns
        -------
        Path
            The database that was written.
        """
        m = self.operator
        target = self._destination(output)
        assignments = {
            "crystal_id": np.array(result.crystal_ids, dtype=object),
            "assignment": result.assignments.numpy(),
        }
        flipped = (
            "crystal_id IN (SELECT crystal_id FROM ambiguity WHERE assignment = 1)"
        )
        conn = duckdb.connect(str(target))
        try:
            conn.execute("BEGIN TRANSACTION")
            conn.execute(
                "CREATE OR REPLACE TABLE ambiguity "
                "(crystal_id VARCHAR PRIMARY KEY, assignment UTINYINT)"
            )
            # registered rather than executemany'd: the latter is orders slower
            conn.register("_assignments", assignments)
            conn.execute("INSERT INTO ambiguity SELECT * FROM _assignments")
            conn.execute(
                "UPDATE reflections SET "
                + ", ".join(
                    f"{axis} = {m[r][0]} * h + {m[r][1]} * k + {m[r][2]} * l"
                    for r, axis in enumerate("hkl")
                )
                + f" WHERE {flipped}"
            )
            # column c of the new reciprocal basis is sum_r M[r][c] * old column r
            conn.execute(
                "UPDATE crystals SET "
                + ", ".join(
                    f"{new}star_{xyz} = "
                    + " + ".join(f"{m[r][c]} * {'abc'[r]}star_{xyz}" for r in range(3))
                    for c, new in enumerate("abc")
                    for xyz in "xyz"
                )
                + f" WHERE {flipped}"
            )
            self._refresh_cells(conn, flipped)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return target

    def _destination(self, output) -> Path:
        # Copy the database
        if output is None:
            return self.path
        target = Path(output)
        if target.resolve() == self.path.resolve():
            return self.path
        shutil.copyfile(self.path, target)
        source_wal = self.path.with_name(self.path.name + ".wal")
        target_wal = target.with_name(target.name + ".wal")
        if source_wal.exists():
            shutil.copyfile(source_wal, target_wal)
        else:
            # a stale sidecar of whatever used to be there would get replayed
            target_wal.unlink(missing_ok=True)
        return target

    # LOADING =========================

    def _load(self):
        # Merge each crystal's reflections into sorted CSR runs of ASU slots.
        clauses = ["intensity IS NOT NULL", "isfinite(intensity)"]
        if self._rmin > 0.0:
            clauses.append(f"resolution_nm_inv >= {self._rmin}")
        if math.isfinite(self._rmax):
            clauses.append(f"resolution_nm_inv <= {self._rmax}")
        where = " AND ".join(clauses)
        conn = duckdb.connect(str(self.path), read_only=True)
        try:
            ids = [
                row[0]
                for row in conn.execute(
                    f"SELECT DISTINCT crystal_id FROM reflections WHERE {where} "
                    "ORDER BY crystal_id"
                ).fetchall()
            ]
            columns = conn.execute(
                "SELECT dense_rank() OVER (ORDER BY crystal_id) - 1 AS ci, "
                "h, k, l, intensity, resolution_nm_inv "
                f"FROM reflections WHERE {where}"
            ).fetchnumpy()
        finally:
            conn.close()
        if not ids:
            raise ValueError(f"{self.path} holds no usable reflections")

        def column(name, dtype):
            data = columns[name]
            return torch.as_tensor(getattr(data, "data", data)).to(self.device, dtype)

        crystal = column("ci", torch.int64)
        hkl = torch.stack([column(axis, torch.int64) for axis in "hkl"], dim=1)
        intensity = column("intensity", torch.float32)
        group, n_groups = self._groups(column("resolution_nm_inv", torch.float32))

        ops = point_group_ops(self.symmetry)
        asu = _to_asu(hkl, ops)
        twin = _to_asu(_apply(asu, self.operator), ops)

        keep = (group >= 0) & (asu != twin).any(dim=1)
        if not int(keep.sum()):
            raise ValueError(
                f"{self.path}: every reflection is twin-proof under "
                f"{_format_operator(self.operator)} or falls outside the "
                "resolution shells"
            )
        crystal, intensity, group = crystal[keep], intensity[keep], group[keep]

        slots, inverse = torch.unique(
            torch.stack([_encode(asu[keep]), _encode(twin[keep])]), return_inverse=True
        )
        slot, twin_slot = inverse
        n_slots = max(1, slots.numel())

        merged, where_merged = torch.unique(
            crystal * n_slots + slot, return_inverse=True
        )
        total = torch.zeros(merged.numel(), dtype=torch.float32, device=self.device)
        count = torch.zeros_like(total)
        total.scatter_add_(0, where_merged, intensity)
        count.scatter_add_(0, where_merged, torch.ones_like(intensity))
        value = total / count
        owner = merged // n_slots
        label = torch.empty(merged.numel(), dtype=torch.int32, device=self.device)
        label.scatter_(0, where_merged, group)
        twinned = torch.empty(merged.numel(), dtype=torch.int64, device=self.device)
        twinned.scatter_(0, where_merged, twin_slot)

        offsets = torch.zeros(len(ids) + 1, dtype=torch.int32, device=self.device)
        offsets[1:] = torch.bincount(owner, minlength=len(ids)).cumsum(0)
        plain = ((merged % n_slots).to(torch.int32), value, label)
        order = torch.argsort(owner * n_slots + twinned)
        reindexed = (twinned[order].to(torch.int32), value[order], label[order])
        return ids, offsets, plain, reindexed, n_groups

    def _groups(self, resolution: Tensor):
        # Resolution shell per reflection
        if not self._auto_res:
            return torch.zeros_like(resolution, dtype=torch.int32), 1
        group = torch.full_like(resolution, -1, dtype=torch.int32)
        for index, (lo, hi) in enumerate(_GROUPS):
            inside = torch.ones_like(group, dtype=torch.bool)
            if lo is not None:
                inside &= resolution > lo
            if hi is not None:
                inside &= resolution < hi
            group[inside] = index
        return group, len(_GROUPS)

    def _refresh_cells(self, conn, flipped: str) -> None:
        # Recompute cell parameters from the reindexed reciprocal axes.
        stars = [f"{s}star_{x}" for s in "abc" for x in "xyz"]
        fetched = conn.execute(
            f"SELECT crystal_id, {', '.join(stars)} FROM crystals WHERE {flipped} "
            f"AND {' AND '.join(f'{s} IS NOT NULL' for s in stars)}"
        ).fetchnumpy()
        if not len(fetched["crystal_id"]):
            return

        star = torch.tensor(
            np.stack([np.ma.getdata(fetched[s]) for s in stars], axis=1),
            dtype=torch.float32,
        ).view(-1, 3, 3)
        real = torch.linalg.inv(star)
        edges = torch.linalg.vector_norm(real, dim=-2)
        angles = [
            torch.rad2deg(
                ((real[:, :, i] * real[:, :, j]).sum(-1) / (edges[:, i] * edges[:, j]))
                .clamp(-1.0, 1.0)
                .acos()
            )
            for i, j in ((1, 2), (0, 2), (0, 1))
        ]
        edges = edges * 10.0
        cells = {
            "crystal_id": fetched["crystal_id"],
            **{f"cell_{n}_A": edges[:, i].numpy() for i, n in enumerate("abc")},
            **{
                f"cell_{n}_deg": angles[i].numpy()
                for i, n in enumerate(("alpha", "beta", "gamma"))
            },
        }
        conn.register("_cells", cells)
        conn.execute("CREATE OR REPLACE TEMP TABLE refreshed AS SELECT * FROM _cells")
        conn.execute(
            "UPDATE crystals SET "
            + ", ".join(f"{column} = refreshed.{column}" for column in list(cells)[1:])
            + " FROM refreshed WHERE crystals.crystal_id = refreshed.crystal_id"
        )

    # CORRELATION =========================

    def _correlate(self, offsets, left, right, partners, n_groups):
        keys_a, vals_a, groups_a = left
        keys_b, vals_b, _ = right
        kernel = select("ambigator", offsets.is_cuda)
        backend = kernel.pair_cc if kernel is not None else _pair_cc_reference
        return backend(
            offsets, keys_a, vals_a, groups_a, keys_b, vals_b, partners, n_groups
        )


def _format_operator(op) -> str:
    # 3x3 matrix back to the "k,h,-l"
    terms = []
    for row in op:
        text = "".join(
            f"{'-' if c < 0 else '+'}{abs(c) if abs(c) != 1 else ''}{axis}"
            for c, axis in zip(row, "hkl")
            if c
        )
        terms.append(text.lstrip("+") or "0")
    return ",".join(terms)


def _apply(hkl: Tensor, op) -> Tensor:
    return torch.stack(
        [
            op[r][0] * hkl[:, 0] + op[r][1] * hkl[:, 1] + op[r][2] * hkl[:, 2]
            for r in range(3)
        ],
        dim=1,
    )


def _encode(hkl: Tensor) -> Tensor:
    # Pack Miller indices into one int64
    if hkl.numel() and int(hkl.abs().max()) >= _HALF:
        raise ValueError("Miller indices exceed the packing range")
    shifted = hkl + _HALF
    return (shifted[:, 0] * _RADIX + shifted[:, 1]) * _RADIX + shifted[:, 2]


def _decode(code: Tensor) -> Tensor:
    place = torch.tensor([_RADIX**2, _RADIX, 1], device=code.device)
    return code.unsqueeze(1).div(place, rounding_mode="floor") % _RADIX - _HALF


def _to_asu(hkl: Tensor, ops) -> Tensor:
    unique, inverse = torch.unique(_encode(hkl), return_inverse=True)
    distinct = _decode(unique)
    best = best_key = None
    for op in ops:
        equivalent = _apply(distinct, op)
        key = _encode(equivalent)
        if best is None:
            best, best_key = equivalent, key
        else:
            better = key > best_key
            best = torch.where(better.unsqueeze(1), equivalent, best)
            best_key = torch.where(better, key, best_key)
    return best[inverse]


def _cc_from_sums(sxy, sx, sy, sx2, sy2, count, n_groups):
    # Pearson CC per resolution group
    safe = count.clamp(min=1.0)
    t1 = sx2 - sx * sx / safe
    t2 = sy2 - sy * sy / safe
    good = (count >= 2.0) & (t1 > 0.0) & (t2 > 0.0)
    cc = torch.where(good, (sxy - sx * sy / safe) / (t1 * t2).sqrt(), 0.0)
    return cc.sum(-1) / n_groups, count.sum(-1).to(torch.int32)


def _gather_rows(offsets, keys, vals, groups, index, width):
    # Ragged CSR runs for index as dense (len(index), width) padded blocks.
    start = offsets[index].long()
    length = (offsets[index + 1] - offsets[index]).long()
    column = torch.arange(width, device=keys.device)
    live = column < length.unsqueeze(1)
    at = (start.unsqueeze(1) + column).clamp_(max=keys.numel() - 1)
    return (
        torch.where(live, keys[at], _PAD),
        torch.where(live, vals[at], 0.0),
        None if groups is None else torch.where(live, groups[at], -1),
    )


def _pair_cc_reference(
    offsets,
    keys_a,
    vals_a,
    groups_a,
    keys_b,
    vals_b,
    partners,
    n_groups,
    budget=1 << 22,
):
    n, ncorr = partners.shape
    width = max(1, int((offsets[1:] - offsets[:-1]).max()))
    rows = max(1, budget // (ncorr * width))
    cc = torch.empty((n, ncorr), dtype=torch.float32, device=keys_a.device)
    novl = torch.empty((n, ncorr), dtype=torch.int32, device=keys_a.device)
    source = torch.arange(n, device=keys_a.device)

    for lo in range(0, n, rows):
        hi = min(lo + rows, n)
        left = source[lo:hi].repeat_interleave(ncorr)
        ka, va, ga = _gather_rows(offsets, keys_a, vals_a, groups_a, left, width)
        kb, vb, _ = _gather_rows(
            offsets, keys_b, vals_b, None, partners[lo:hi].reshape(-1).long(), width
        )
        at = torch.searchsorted(kb.contiguous(), ka.contiguous()).clamp_(max=width - 1)
        hit = (kb.gather(1, at) == ka) & (ka != _PAD)
        bv = vb.gather(1, at)
        sums = []
        for group in range(n_groups):
            weight = (hit & (ga == group)).to(torch.float32)
            a = weight * va
            b = weight * bv
            sums.append(
                torch.stack(
                    [
                        (a * bv).sum(1),
                        a.sum(1),
                        b.sum(1),
                        (a * va).sum(1),
                        (b * bv).sum(1),
                        weight.sum(1),
                    ]
                )
            )
        pair_cc, pair_n = _cc_from_sums(*torch.stack(sums, dim=-1), n_groups)
        cc[lo:hi] = pair_cc.view(hi - lo, ncorr)
        novl[lo:hi] = pair_n.view(hi - lo, ncorr)
    return cc, novl


def _pass(cc, cc_r, valid, valid_r, partners, assignments):
    # One refinement pass over every crystal at once
    same = assignments.unsqueeze(1) == assignments[partners.long()]
    f_mask = valid & same
    g_mask = valid & ~same
    f_mask_r = valid_r & ~same
    g_mask_r = valid_r & same
    p = f_mask.sum(1) + f_mask_r.sum(1)
    q = g_mask.sum(1) + g_mask_r.sum(1)
    f = (cc * f_mask).sum(1) + (cc_r * f_mask_r).sum(1)
    g = (cc * g_mask).sum(1) + (cc_r * g_mask_r).sum(1)
    usable = (p > 0) & (q > 0)
    f = f / p.clamp(min=1)
    g = g / q.clamp(min=1)
    flip = usable & (f < g)
    return (
        assignments ^ flip,
        int(flip.sum()),
        float(f[usable].mean()) if bool(usable.any()) else math.nan,
        float(g[usable].mean()) if bool(usable.any()) else math.nan,
        int((~usable).sum()),
    )


# CLI =====================================


@click.command()
@click.option(
    "-i",
    "--input",
    "database",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="DuckDB database written by probixi (.duckdb/.db).",
)
@click.option("-y", "--symmetry", required=True, help="Actual point group, e.g. '3'.")
@click.option(
    "-w",
    "--apparent",
    required=True,
    help="Apparent (lattice) point group, e.g. '321'. The ambiguity operator is "
    "derived from it and --symmetry.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Reindex a copy written here, leaving the input alone. Without it "
    "the input database is rewritten in place.",
)
@click.option(
    "-n", "--iterations", default=6, show_default=True, help="Refinement passes."
)
@click.option(
    "--ncorr", default=1000, show_default=True, help="Partners correlated per crystal."
)
@click.option("--lowres", default=None, type=float, help="Low-resolution cutoff in A.")
@click.option(
    "--highres", default=None, type=float, help="High-resolution cutoff in A."
)
@click.option(
    "--seed", default=1988, show_default=True, help="Seeds the starting split."
)
@click.option("--device", default=None, help="Torch device, or 'auto'.")
@click.option("--quiet", is_flag=True, help="Only report errors.")
def main(
    database,
    symmetry,
    apparent,
    output,
    iterations,
    ncorr,
    lowres,
    highres,
    seed,
    device,
    quiet,
):
    """Resolve the indexing ambiguity in a probixi DuckDB run.

    Clusters the crystals into the two indexing choices of --symmetry under
    --apparent, then reindexes the crystals on the wrong side. The algorithm
    is CrystFEL's ambigator, by Thomas White.

        probixi-resolve -i run.duckdb -y 3 -w 321 -o detwinned.duckdb
    """
    if device and device.strip().lower() == "auto":
        device = None
    try:
        amb = Ambigator(
            database,
            symmetry=symmetry,
            apparent=apparent,
            lowres=lowres,
            highres=highres,
            device=device,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if not quiet:
        click.echo(f"Ambiguity operator: {_format_operator(amb.operator)}")
    try:
        result = amb.resolve(iterations=iterations, ncorr=ncorr, seed=seed)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not quiet:
        for i, (flips, f, g) in enumerate(result.history, 1):
            click.echo(f"  pass {i}: {flips} flipped, mean f = {f:.4f}, g = {g:.4f}")
        if result.n_unused:
            click.echo(f"{result.n_unused} crystal(s) had no usable correlation")
    written = amb.reindex(result, output=output)
    click.echo(f"Reindexed {result.n_reindexed}/{len(result)} crystals in {written}")


if __name__ == "__main__":
    main()
