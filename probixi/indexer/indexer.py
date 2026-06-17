from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional

import torch
from torch import Tensor

from ..io.cell import CellParams
from ..peakfinding.peaks import PeakResult
from .forward import detector_to_q
from .integrate import integrate_predicted
from .lattice import cell_to_B, decompose_A
from .predict import detector_q_max, predict_reflections
from .refine import RefineResult, refine_multiframe_known_B
from .seed import sphere_seed_candidates


@dataclass
class IndexResult:
    """Indexing solution for a single frame.

    Attributes
    ----------
    frame_index : int
        Index of the source frame.
    n_peaks : int
        Number of peaks fed to the indexer.
    n_indexed : int
        Number of peaks explained by the solution.
    rmsd : float
        Root-mean-square q-residual over indexed peaks.
    A : Tensor
        (3, 3) reciprocal-to-lab matrix, ``A = U @ B``.
    U : Tensor
        (3, 3) crystal orientation (rotation).
    B : Tensor
        (3, 3) cell-only reciprocal basis.
    cell : CellParams
        Unit cell recovered from ``B``.
    indexed_mask : Tensor
        (N,) bool mask of which peaks were indexed.
    hkl : Tensor
        (N, 3) integer Miller indices assigned to each peak.
    positions : Tensor
        (N, 2) observed peak ``(row, col)`` positions, aligned with ``hkl``.
    intensities : Tensor
        (N,) integrated peak intensity, aligned with ``positions``.
    sigmas : Tensor
        (N,) 1-sigma uncertainty on each intensity, aligned with ``positions``.
    loss_history : Tensor
        Per-iteration refinement loss (excluded from repr).
    predicted_hkl : Tensor, optional
        (M, 3) Miller indices of all reflections predicted to diffract on this
        frame, or None if prediction/integration was not run.
    predicted_positions : Tensor, optional
        (M, 2) predicted detector ``(row, col)`` positions.
    predicted_intensities : Tensor, optional
        (M,) box-integrated intensity per predicted reflection (snapped to the
        observed centroid where a peak coincides).
    predicted_sigmas : Tensor, optional
        (M,) 1-sigma uncertainty on each predicted intensity.
    predicted_peak : Tensor, optional
        (M,) max background-subtracted pixel in each box (spot height; the
        CrystFEL stream's ``peak`` column).
    predicted_background : Tensor, optional
        (M,) mean per-pixel noise background under each box (the CrystFEL
        stream's ``background`` column; informational only).
    scale : float, optional
        Per-frame relative intensity scale (background fluence vs the
        calibration reference), if inferred; an initialization for downstream
        per-pattern scaling/merging.
    scale_sigma : float, optional
        1-sigma uncertainty on ``scale``.
    """

    frame_index: int
    n_peaks: int
    n_indexed: int
    rmsd: float
    A: Tensor
    U: Tensor
    B: Tensor
    cell: CellParams
    indexed_mask: Tensor
    hkl: Tensor
    positions: Tensor
    intensities: Tensor
    sigmas: Tensor
    loss_history: Tensor = field(repr=False)
    predicted_hkl: Optional[Tensor] = None
    predicted_positions: Optional[Tensor] = None
    predicted_intensities: Optional[Tensor] = None
    predicted_sigmas: Optional[Tensor] = None
    predicted_peak: Optional[Tensor] = None
    predicted_background: Optional[Tensor] = None
    scale: Optional[float] = None
    scale_sigma: Optional[float] = None

    def cell_dict(self) -> dict:
        return self.cell.as_dict_degrees()


@dataclass
class SeedConfig:
    """Tolerances and limits for generating candidate orientations.

    Attributes
    ----------
    q_tolerance : float, optional
        Max ``|q|`` mismatch (A^-1) for a reflection to count as matched. ``None``
        (default) derives it from the cell as ``q_tolerance_fraction`` times the
        smallest reciprocal-basis spacing, so it self-scales.
    q_tolerance_fraction : float
        Fraction of the smallest reciprocal spacing used when ``q_tolerance`` is None.
    max_candidates : int
        Cap on candidate orientations carried into refinement.
    max_seed_peaks : int
        Peaks fed to the seeder are capped to this many (the brightest, by
        intensity, when available). The full peak set still drives refinement and
        the final solution.
    n_directions : int
        Fibonacci-sphere sample count for the sphere seeder.
    n_spin : int
        Number of in-plane (roll) angles tried per kept direction.
    top_directions : int
        Best sphere directions carried into the spin stage.
    adaptive_sparse : bool
        Widen the orientation search on peak-starved frames: when a frame's
        seeding peak count is below ``sparse_peak_threshold``, the sphere search
        counts are multiplied by ``sparse_scale``.
    sparse_peak_threshold : int
        Seeding-peak count below which a frame is treated as sparse.
    sparse_scale : float
        Multiplier applied to the sphere search counts on sparse frames.
    """

    q_tolerance: Optional[float] = None
    q_tolerance_fraction: float = 0.25
    max_candidates: int = 64
    max_seed_peaks: int = 80
    n_directions: int = 6000
    n_spin: int = 120
    top_directions: int = 32
    adaptive_sparse: bool = True
    sparse_peak_threshold: int = 30
    sparse_scale: float = 2.0


@dataclass
class RefineConfig:
    """Settings for the gradient-based orientation refinement.

    Attributes
    ----------
    lr : float
        Adam learning rate on the axis-angle perturbation.
    max_iters : int
        Number of optimization steps (upper bound; early-stop may end sooner).
    reassign_every : int
        Re-assign peak->hkl correspondences every this many steps.
    min_indexed : int
        Minimum indexed peaks for a candidate to be accepted.
    patience : int
        Stop early if the batch loss has not improved (by ``rel_tol``) for this
        many consecutive steps. Set <= 0 to disable early-stopping.
    rel_tol : float
        Relative improvement threshold for the patience counter.
    """

    lr: float = 1e-3
    max_iters: int = 200
    reassign_every: int = 10
    min_indexed: int = 6
    patience: int = 40
    rel_tol: float = 1e-3


@dataclass(frozen=True)
class IntegrateConfig:
    """Settings for predicting and integrating reflections after indexing.

    Attributes
    ----------
    enabled : bool
        Predict the full lattice and box-integrate it. When off, the stream lists
        only observed-and-indexed peaks. Requires the streaming path.
    partiality_threshold : float
        Max ``|S| - 1`` (Ewald excitation error) for a reflection to count as
        diffracting. Larger -> more (more partial) reflections predicted.
    box_radius : int
        Half-width (px) of the integration box.
    snap_radius : float
        A predicted spot within this many px of an observed peak is recentred on
        that peak's centroid before integration.
    """

    enabled: bool = True
    partiality_threshold: float = 0.0005
    box_radius: int = 3
    snap_radius: float = 5.0


@dataclass
class CellMatchConfig:
    """Tolerances for accepting a recovered cell as matching the target.

    Attributes
    ----------
    edge_tolerance : float
        Max fractional difference allowed on sorted cell edges.
    angle_tolerance_deg : float
        Max absolute difference (deg) on sorted cell angles.
    """

    edge_tolerance: float = 0.05
    angle_tolerance_deg: float = 3.0

    @property
    def angle_tolerance_rad(self) -> float:
        return math.radians(self.angle_tolerance_deg)


class IndexStream:
    """Lazy, composable stream of per-frame :class:`IndexResult`s.

    A torch-iterable produced by :meth:`Indexer.index_stream`. Operators
    (``map``/``filter``/``tap``) compose lazily; terminals (``collect``,
    ``to_stream``, ``count``, ...) drive the underlying generator once.
    """

    def __init__(self, source: Iterable[IndexResult]):
        self._source: Iterator[IndexResult] = iter(source)

    def map(self, fn: Callable[[IndexResult], IndexResult]) -> "IndexStream":
        return IndexStream(fn(r) for r in self._source)

    def filter(self, predicate: Callable[[IndexResult], bool]) -> "IndexStream":
        return IndexStream(r for r in self._source if predicate(r))

    def tap(self, fn: Callable[[IndexResult], None]) -> "IndexStream":
        def _gen() -> Iterator[IndexResult]:
            for r in self._source:
                fn(r)
                yield r

        return IndexStream(_gen())

    def to_stream(self, writer: Callable[[IndexResult], None]) -> int:
        n = 0
        for r in self._source:
            writer(r)
            n += 1
        return n

    def collect(self) -> list[IndexResult]:
        return list(self._source)

    def collect_dict(self) -> dict[int, IndexResult]:
        return {r.frame_index: r for r in self._source}

    def count(self) -> int:
        return sum(1 for _ in self._source)

    def for_each(self, fn: Callable[[IndexResult], None]) -> None:
        for r in self._source:
            fn(r)

    def __iter__(self) -> Iterator[IndexResult]:
        return self._source


class Indexer:
    """Match detector peaks to a target unit cell to recover crystal orientations.

    Parameters
    ----------
    geometry : dict
        Detector geometry (``beam_center``, ``clen``, ``pixel_size``, ``wavelength``).
    target_cell : CellParams
        Known unit cell the recovered solutions must match.
    seed : SeedConfig, optional
        Candidate-orientation search settings.
    refine : RefineConfig, optional
        Gradient-refinement settings.
    cell_match : CellMatchConfig, optional
        Tolerances for accepting a recovered cell.
    integrate : IntegrateConfig, optional
        Reflection prediction/integration settings (streaming path).
    dtype : torch.dtype, optional
        Lattice-math dtype (default ``torch.float64``).
    device : torch.device, optional
        Device for all tensors.
    """

    def __init__(
        self,
        geometry: dict,
        target_cell: CellParams,
        seed: Optional[SeedConfig] = None,
        refine: Optional[RefineConfig] = None,
        cell_match: Optional[CellMatchConfig] = None,
        integrate: Optional[IntegrateConfig] = None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        required = {"beam_center", "clen", "pixel_size", "wavelength"}
        missing = required - set(geometry)
        if missing:
            raise ValueError(f"geometry missing keys: {sorted(missing)}")

        self.geometry = geometry
        self.target_cell = target_cell
        self.seed = seed or SeedConfig()
        self.refine = refine or RefineConfig()
        self.cell_match = cell_match or CellMatchConfig()
        self.integrate = integrate or IntegrateConfig()
        self.dtype = dtype
        self.device = device
        self._q_max: Optional[float] = None
        self.B_target = cell_to_B(target_cell, device=device, dtype=dtype)

        if self.seed.q_tolerance is not None:
            self.q_tolerance = float(self.seed.q_tolerance)
        else:
            min_spacing = float(torch.linalg.vector_norm(self.B_target, dim=0).min())
            self.q_tolerance = self.seed.q_tolerance_fraction * min_spacing

    @classmethod
    def fast(
        cls,
        geometry: dict,
        target_cell: CellParams,
        **overrides,
    ) -> "Indexer":
        # Fast preset: narrower search
        overrides.setdefault(
            "seed",
            SeedConfig(
                max_candidates=16,
                n_directions=3000,
                n_spin=72,
                top_directions=16,
            ),
        )
        overrides.setdefault("refine", RefineConfig(max_iters=80, reassign_every=20))
        return cls(geometry, target_cell, **overrides)

    @classmethod
    def thorough(
        cls,
        geometry: dict,
        target_cell: CellParams,
        **overrides,
    ) -> "Indexer":
        # High-recall preset: wider search, more reassignments, takes a million years (not really, but long)
        overrides.setdefault(
            "seed",
            SeedConfig(
                max_candidates=128,
                n_directions=12000,
                n_spin=240,
                top_directions=64,
            ),
        )
        overrides.setdefault("refine", RefineConfig(max_iters=400, reassign_every=5))
        return cls(geometry, target_cell, **overrides)

    def _cell_matches_target(self, cell: CellParams) -> bool:
        # Compare sorted edges/angles so the match is invariant to axis labelling.
        tc = self.target_cell
        tol = self.cell_match.edge_tolerance
        angle_tol = self.cell_match.angle_tolerance_rad
        edges_obs = sorted([cell.a, cell.b, cell.c])
        edges_tgt = sorted([tc.a, tc.b, tc.c])
        if any(abs(o - t) / t > tol for o, t in zip(edges_obs, edges_tgt)):
            return False
        angles_obs = sorted([cell.alpha, cell.beta, cell.gamma])
        angles_tgt = sorted([tc.alpha, tc.beta, tc.gamma])
        return all(abs(o - t) <= angle_tol for o, t in zip(angles_obs, angles_tgt))

    @torch.no_grad()
    def lift(
        self,
        positions: Tensor,
        frame_rotation: Optional[Tensor] = None,
    ) -> Tensor:
        # Lift detector pixel positions (N, 2) to reciprocal-space q-vectors (N, 3).
        return detector_to_q(
            positions.to(device=self.device, dtype=self.dtype),
            self.geometry,
            frame_rotation=frame_rotation,
            dtype=self.dtype,
        )

    def index_frames(
        self,
        positions_by_frame: dict[int, Tensor],
        frame_rotations: Optional[dict[int, Tensor]] = None,
        intensities_by_frame: Optional[dict[int, Tensor]] = None,
        sigmas_by_frame: Optional[dict[int, Tensor]] = None,
        weights_by_frame: Optional[dict[int, Tensor]] = None,
    ) -> dict[int, IndexResult]:
        """Index many frames with batched refinement.

        Lifts and seeds per-frame (cheap), then runs one Adam over all frames at
        once.

        Parameters
        ----------
        positions_by_frame : dict of int -> Tensor
            Per-frame (N, 2) peak ``(row, col)`` positions.
        frame_rotations : dict of int -> Tensor, optional
            Per-frame (3, 3) lab->crystal rotations.
        intensities_by_frame, sigmas_by_frame : dict of int -> Tensor, optional
            Per-peak integrated intensities and 1-sigma uncertainties, aligned
            row-for-row with ``positions_by_frame``. Default to zeros when omitted.
        weights_by_frame : dict of int -> Tensor, optional
            Per-peak detection confidences for soft seeding/refinement.

        Returns
        -------
        dict of int -> IndexResult
            One entry per successfully indexed frame.
        """
        if not positions_by_frame:
            return {}

        seeded: list[tuple[int, int, Tensor, Tensor, Optional[Tensor]]] = []
        for idx, positions in positions_by_frame.items():
            if (
                positions.ndim != 2
                or positions.shape[-1] != 2
                or positions.shape[0] < 4
            ):
                continue
            rotation = (frame_rotations or {}).get(idx)
            q = self.lift(positions, frame_rotation=rotation)
            intensities = (intensities_by_frame or {}).get(idx)
            weights = (weights_by_frame or {}).get(idx)
            with torch.no_grad():
                # Seed on a bounded subset of the brightest peaks
                keep = self._seed_keep_indices(q, intensities)
                q_seed = q if keep is None else q[keep]
                w_seed = (
                    None
                    if weights is None
                    else (weights if keep is None else weights[keep])
                )
                A_init = self._seed_orientations(q_seed, weights=w_seed)
            if A_init.shape[0] == 0:
                continue
            seeded.append((int(idx), int(positions.shape[0]), q, A_init, weights))

        if not seeded:
            return {}

        has_weights = any(s[4] is not None for s in seeded)
        results = refine_multiframe_known_B(
            [s[3] for s in seeded],
            [s[2] for s in seeded],
            q_tolerance=self.q_tolerance,
            lr=self.refine.lr,
            max_iters=self.refine.max_iters,
            reassign_every=self.refine.reassign_every,
            min_indexed=self.refine.min_indexed,
            patience=self.refine.patience,
            rel_tol=self.refine.rel_tol,
            weights_per_frame=[s[4] for s in seeded] if has_weights else None,
        )

        out: dict[int, IndexResult] = {}
        for (idx, n_peaks, _, _, _), rr in zip(seeded, results):
            built = self._build_indexing_result(
                rr,
                frame_index=idx,
                n_peaks=n_peaks,
                positions=positions_by_frame[idx],
                intensities=(intensities_by_frame or {}).get(idx),
                sigmas=(sigmas_by_frame or {}).get(idx),
            )
            if built is not None:
                out[idx] = built
        return out

    def _seed_keep_indices(
        self, q: Tensor, intensities: Optional[Tensor]
    ) -> Optional[Tensor]:
        # Indices of the q-vectors handed to the seeder, capped at max_seed_peaks
        cap = self.seed.max_seed_peaks
        n = q.shape[0]
        if cap <= 0 or n <= cap:
            return None
        if intensities is not None and intensities.shape[0] == n:
            return torch.topk(intensities, k=cap).indices
        return torch.topk(
            torch.linalg.vector_norm(q, dim=-1), k=cap, largest=False
        ).indices

    def _seed_orientations(
        self,
        q_seed: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        n_dir, n_spin, top_dir, top_k = (
            self.seed.n_directions,
            self.seed.n_spin,
            self.seed.top_directions,
            self.seed.max_candidates,
        )
        # widen the search on low peak frames
        if (
            self.seed.adaptive_sparse
            and int(q_seed.shape[0]) < self.seed.sparse_peak_threshold
        ):
            s = self.seed.sparse_scale
            n_dir = int(n_dir * s)
            n_spin = int(n_spin * s)
            top_dir = int(top_dir * s)
            top_k = int(top_k * s)
        return sphere_seed_candidates(
            q_seed,
            self.B_target,
            q_tolerance=self.q_tolerance,
            n_directions=n_dir,
            n_spin=n_spin,
            top_directions=top_dir,
            top_k=top_k,
            weights=weights,
        )

    def _build_indexing_result(
        self,
        result: RefineResult,
        frame_index: int,
        n_peaks: int,
        positions: Tensor,
        intensities: Optional[Tensor] = None,
        sigmas: Optional[Tensor] = None,
    ) -> Optional[IndexResult]:
        # Pick the best refined candidate whose cell matches the target, or None.
        if result.A.shape[0] == 0:
            return None
        if intensities is None:
            intensities = torch.zeros(
                positions.shape[0], dtype=positions.dtype, device=positions.device
            )
        if sigmas is None:
            sigmas = torch.zeros(
                positions.shape[0], dtype=positions.dtype, device=positions.device
            )
        n_indexed = result.n_indexed
        soft = result.soft_score
        rmsd = result.rmsd
        # rank by confidence-weighted inlier evidence breaking ties toward lower rmsd (scaled to stay below 1)
        score = soft - rmsd / (rmsd.max().clamp_min(1e-12) * 1e3)
        ranking = torch.argsort(score, descending=True)
        for cand in ranking.tolist():
            if int(n_indexed[cand]) < self.refine.min_indexed:
                continue
            A_cand = result.A[cand]
            try:
                U, B, cell = decompose_A(A_cand)
            except Exception:
                continue
            if not self._cell_matches_target(cell):
                continue
            return IndexResult(
                frame_index=frame_index,
                n_peaks=n_peaks,
                n_indexed=int(n_indexed[cand]),
                rmsd=float(rmsd[cand]),
                A=A_cand,
                U=U,
                B=B,
                cell=cell,
                indexed_mask=result.indexed[cand],
                hkl=result.hkl[cand],
                positions=positions,
                intensities=intensities,
                sigmas=sigmas,
                loss_history=result.history,
            )
        return None

    def _positions_from_frame(
        self, r: PeakResult
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Pull kept peak-blob centroids and photometry off a PeakResult
        stats = r.kept_stats
        positions = torch.stack(
            [stats.row_centroid, stats.col_centroid], dim=-1
        ).to(device=self.device, dtype=self.dtype)
        intensities = stats.intensity_sum.to(device=self.device, dtype=self.dtype)
        sigmas = stats.intensity_sigma.to(device=self.device, dtype=self.dtype)
        weights = stats.posterior_mean.to(device=self.device, dtype=self.dtype)
        return positions, intensities, sigmas, weights

    def index_stream(
        self,
        peak_stream: Iterable[PeakResult],
        batch_size: int = 8,
        frame_rotations: Optional[dict[int, Tensor]] = None,
    ) -> IndexStream:
        """Lazily index a peakfinder stream.

        Parameters
        ----------
        peak_stream : iterable of PeakResult
            Per-frame peak results from the peakfinder.
        batch_size : int, optional
            Frames batched into each refinement call (and the window over which
            pixel maps are held, then released).
        frame_rotations : dict of int -> Tensor, optional
            Per-frame (3, 3) lab->crystal rotations.

        Returns
        -------
        IndexStream
            Lazy stream of :class:`IndexResult`. When ``IntegrateConfig.enabled``
            and the peak results carry pixel maps, each indexed frame also gets
            its full predicted reflection list integrated from that frame's
            excess/variance maps (a merge-ready chunk).
        """

        def _flush(buf: list[dict]) -> Iterator[IndexResult]:
            results = self.index_frames(
                {b["idx"]: b["pos"] for b in buf},
                frame_rotations=frame_rotations,
                intensities_by_frame={b["idx"]: b["I"] for b in buf},
                sigmas_by_frame={b["idx"]: b["sig"] for b in buf},
                weights_by_frame={b["idx"]: b["w"] for b in buf},
            )
            for b in buf:
                res = results.get(b["idx"])
                if res is None:
                    continue
                if self.integrate.enabled and b["excess"] is not None:
                    self._integrate_result(
                        res, b["excess"], b["var"], b["mask"], b["mean"]
                    )
                yield res

        def _gen() -> Iterator[IndexResult]:
            buf: list[dict] = []
            for r in peak_stream:
                if len(r) < 4:
                    continue
                idx = r.frame_index if r.frame_index is not None else 0
                positions, intensities, sigmas, weights = self._positions_from_frame(r)
                buf.append(
                    {
                        "idx": idx,
                        "pos": positions,
                        "I": intensities,
                        "sig": sigmas,
                        "w": weights,
                        "excess": r.scores.get("excess") if r.scores else None,
                        "var": r.var,
                        "mask": r.valid_mask,
                        "mean": r.mean,
                    }
                )
                if len(buf) >= batch_size:
                    yield from _flush(buf)
                    buf = []
            if buf:
                yield from _flush(buf)

        return IndexStream(_gen())

    def _integrate_result(
        self,
        result: IndexResult,
        excess: Tensor,
        var: Tensor,
        valid_mask: Optional[Tensor],
        mean: Optional[Tensor] = None,
    ) -> None:
        # Predict the full lattice for result.A and box-integrate it.
        frame_shape = (int(excess.shape[-2]), int(excess.shape[-1]))
        if self._q_max is None:
            self._q_max = detector_q_max(self.geometry, frame_shape)
        pred = predict_reflections(
            result.A.to(excess.dtype),
            self.geometry,
            q_max=self._q_max,
            partiality_threshold=self.integrate.partiality_threshold,
            frame_shape=frame_shape,
        )
        if len(pred) == 0:
            return
        # valid mask applied inside the box reduction
        positions, intensity, sigma, _, peak, background = integrate_predicted(
            pred.positions.to(excess.dtype),
            excess,
            var,
            result.positions.to(excess.dtype),
            result.intensities.to(excess.dtype),
            result.sigmas.to(excess.dtype),
            snap_radius=self.integrate.snap_radius,
            box_radius=self.integrate.box_radius,
            mean=mean.to(excess.dtype) if mean is not None else None,
            pixel_valid=valid_mask,
        )
        result.predicted_hkl = pred.hkl
        result.predicted_positions = positions
        result.predicted_intensities = intensity
        result.predicted_sigmas = sigma
        result.predicted_peak = peak
        result.predicted_background = background
