from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import duckdb
import numpy as np
import torch
from torch import Tensor

from .kernels import select

PathLike = Union[str, Path]

_RADIX = 1 << 21
_HALF = _RADIX // 2
_PAD = 0x7FFFFFFF  # int32 max: pads the end of every sorted run
_MIN_COMMON = 4

# CC shells as 1/d in nm^-1; 1 to 2 nm^-1 (10 to 5 A) is not correlated
_GROUPS = ((None, 1.0), (2.0, 4.0), (4.0, None))

_AXES = {"h": 0, "k": 1, "l": 2}
_TERM = re.compile(r"([+-]?)(\d*)([hkl])")
_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_INVERSION = "-h,-k,-l"

# generators of the rotation point groups; a Laue group adds the inversion
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
    """Parse an operator such as ``"k,h,-l"`` into M with (h', k', l') = M (h, k, l).

    Components may carry integer coefficients (``"2h-k"``). Spaces and ``*`` are
    ignored.

    Raises
    ------
    ValueError
        If ``text`` does not have three parsable components.
    """
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
    """Return the operations of a point group as integer matrices on (h, k, l).

    Parameters
    ----------
    name : str
        Hermann-Mauguin symbol of a rotation group, such as ``"321"``, or of a
        Laue group, such as ``"-3m1"``, which adds the inversion. ``"2"`` and
        ``"2/m"`` take a unique axis as ``"2_uab"``; the default axis is c.

    Returns
    -------
    list of tuple
        Sorted matrices M with (h', k', l') = M (h, k, l).
    """
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
    """Return the operator that relates the two indexing choices.

    Parameters
    ----------
    symmetry : str
        Point group of the structure, such as ``"3"``.
    apparent : str
        Point group of the lattice, such as ``"321"``.

    Returns
    -------
    tuple
        The first operation of ``apparent``, in sorted order, not in ``symmetry``.

    Raises
    ------
    ValueError
        If ``symmetry`` is not a subgroup of index 2 in ``apparent``.
    """
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
        """Number of crystals assigned to the reindexed side."""
        return int(self.assignments.sum())


class Ambigator:
    """Resolve the indexing ambiguity of the crystals in a probixi database.

    Parameters
    ----------
    path : str or Path
        A ``.duckdb`` file written by :class:`~probixi.io.db.DuckDBOffloader`.
        :meth:`resolve` reads ``reflections``; :meth:`reindex` also writes
        ``reflections``, ``crystals``, and ``ambiguity``.
    symmetry : str
        Point group of the structure, such as ``"3"``.
    apparent : str, optional
        Point group of the lattice, such as ``"321"``, from which the ambiguity
        operator is derived.
    operator : str, optional
        Ambiguity operator, such as ``"k,h,-l"``. Give exactly one of
        ``apparent`` and ``operator``.
    lowres, highres : float, optional
        Resolution limits in Angstrom. Setting either replaces the three
        default shells, d > 10, 5 to 2.5, and d < 2.5 Angstrom, with one shell
        over the given range.
    device : torch.device, optional
        Defaults to CUDA when available, else CPU.

    Notes
    -----
    The method is that of Brehm and Diederichs (2014), Acta Cryst. D70,
    101-109, as implemented in CrystFEL's ``ambigator`` by Thomas White.
    Differences from ``ambigator``:

    * A pass updates every assignment at once, as one vectorised reduction.
    * The ``ncorr`` partners of each crystal are drawn once, and partners with
      fewer than four common reflections are dropped rather than redrawn.
    * A pair is usable when it shares four reflections over all shells, not
      four in the last shell.
    * No correlation matrix, f/g graph, or assignment file is written;
      :attr:`AmbigatorResult.history` holds the per-pass means.
    * Reflections the database flags as masked or overlapping are excluded,
      as they are from the stream.
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
        # lowres and highres are d in A; the database holds 1/d in nm^-1
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

        Raises
        ------
        ValueError
            If the database holds fewer than two crystals with usable reflections.
        """
        ids, offsets, plain, reindexed, n_groups = self._load()
        n = len(ids)
        if n < 2:
            raise ValueError(f"{self.path} holds {n} crystals; need at least two")
        ncorr = max(1, min(int(ncorr), n - 1))

        gen = torch.Generator().manual_seed(int(seed))

        partners = torch.randint(0, n - 1, (n, ncorr), generator=gen)
        partners += partners >= torch.arange(n).unsqueeze(1)  # skip self, stay uniform
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
        """Write ``result`` to the database, or to a copy at ``output``.

        Every assignment is written to an ``ambiguity`` table. For the
        reindexed crystals, ``reflections`` h, k, l and the ``crystals``
        reciprocal axes and cell parameters are transformed. The update is one
        transaction, rolled back on error. A second call transforms the same
        crystals again.

        Parameters
        ----------
        result : AmbigatorResult
            From :meth:`resolve` on this same database.
        output : str or Path, optional
            Path of a copy to reindex, leaving ``path`` unchanged. An existing
            file is overwritten.

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
            # register() inserts in bulk; executemany is orders of magnitude slower
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
        # file to write: path itself, or a copy at output with its .wal sidecar
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
            # DuckDB would replay a stale .wal left at the target into the copy
            target_wal.unlink(missing_ok=True)
        return target

    def _load(self):
        # CSR runs per crystal of (ASU slot, mean I, shell), by plain and twin slot
        clauses = ["intensity IS NOT NULL", "isfinite(intensity)", "sigma > 0"]
        clauses += ["COALESCE(n_masked, 0) = 0", "COALESCE(n_overlap, 0) = 0"]
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

        # a reflection whose twin is its own ASU slot cannot tell the choices apart
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

        # symmetry equivalents within one crystal merge to their mean intensity
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
        # shell index per reflection; -1 outside every shell
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
        # recompute cell parameters from the reindexed reciprocal axes
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
        edges = edges * 10.0  # nm to A
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

    def _correlate(self, offsets, left, right, partners, n_groups):
        keys_a, vals_a, groups_a = left
        keys_b, vals_b, _ = right
        kernel = select("ambigator", offsets.is_cuda)
        backend = kernel.pair_cc if kernel is not None else _pair_cc_reference
        return backend(
            offsets, keys_a, vals_a, groups_a, keys_b, vals_b, partners, n_groups
        )


def _format_operator(op) -> str:
    # inverse of parse_operator: matrix to "k,h,-l" text
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
    # pack hkl into one int64, 21 bits per index
    if hkl.numel() and int(hkl.abs().max()) >= _HALF:
        raise ValueError("Miller indices exceed the packing range")
    shifted = hkl + _HALF
    return (shifted[:, 0] * _RADIX + shifted[:, 1]) * _RADIX + shifted[:, 2]


def _decode(code: Tensor) -> Tensor:
    place = torch.tensor([_RADIX**2, _RADIX, 1], device=code.device)
    return code.unsqueeze(1).div(place, rounding_mode="floor") % _RADIX - _HALF


def _to_asu(hkl: Tensor, ops) -> Tensor:
    # representative of each hkl: the equivalent with the largest packed code
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
    # Pearson CC per shell, averaged over all n_groups shells; an empty shell adds 0
    safe = count.clamp(min=1.0)
    t1 = sx2 - sx * sx / safe
    t2 = sy2 - sy * sy / safe
    good = (count >= 2.0) & (t1 > 0.0) & (t2 > 0.0)
    cc = torch.where(good, (sxy - sx * sy / safe) / (t1 * t2).sqrt(), 0.0)
    return cc.sum(-1) / n_groups, count.sum(-1).to(torch.int32)


def _gather_rows(offsets, keys, vals, groups, index, width):
    # CSR runs of index as dense (len(index), width) blocks padded with _PAD
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
    rows = max(1, budget // (ncorr * width))  # crystals per chunk, ~budget elements
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
        # runs are sorted and _PAD sorts last, so searchsorted finds shared slots
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
    # f: mean CC consistent with the current assignment, g: with the flipped one
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


from .cli.resolve import main  # noqa: E402

if __name__ == "__main__":
    main()
