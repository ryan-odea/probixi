from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Generic, Iterable, Iterator, Optional, TypeVar

import torch
from torch import Tensor

from ..io.cell import CellParams
from ..peakfinding.peaks import PeakResult
from .forward import detector_to_q
from .integrate import (
    A_INV_TO_NM_INV,
    disk_overlap_counts,
    falloff_resolution_limit,
    frame_background_stats,
    integrate_rings,
    peak_resolution_limit,
    radius_by_shell,
    resolution_bands,
    ring_sums,
    spot_enrichment,
    unexplained_signal_p,
)
from .lattice import B_to_cell, cell_to_B
from .predict import detector_q_max, predict_reflections
from .refine import RefineResult, refine_cell, refine_multiframe_known_B
from .rocking import estimate_mosaicity
from .seed import sphere_seed_candidates

MIN_PEAKS_TO_INDEX = 5
# Element type of an IndexStream: one lattice, or one frame carrying several.
_ResultT = TypeVar("_ResultT", "IndexResult", "FrameIndexResult")
# Peeling: same-lattice tolerance (relative Frobenius, any setting), min peaks per side
DUPLICATE_LATTICE_TOL = 0.02
MIN_PEAKS_PER_LATTICE = 6
# (disk, annulus inner, outer) px for a few-pixel spot, when calibrate() did not run
FALLBACK_RADII = (3.0, 4.0, 7.36)


#: a found peak within this distance (px) explains a prediction; only the
#: enrichment test uses it, predictions are never moved
EXPLAINED_RADIUS_PX = 5.0


@dataclass
class IndexStats:
    """Running funnel tallies over an index stream.

    Attributes
    ----------
    frames : int
        Frames (peak results) seen.
    hits : int
        Frames with at least ``MIN_PEAKS_TO_INDEX`` peaks (indexing attempted).
    indexed : int
        Frames that yielded at least one accepted solution.
    crystals : int
        Accepted lattices over all frames; exceeds ``indexed`` once
        ``SeedConfig.max_lattices`` is above one.
    """

    frames: int = 0
    hits: int = 0
    indexed: int = 0
    crystals: int = 0

    @property
    def hit_rate(self) -> float:
        """Hits per frame seen (0 when no frames)."""
        return self.hits / self.frames if self.frames else 0.0

    @property
    def index_rate_of_hits(self) -> float:
        """Indexed solutions per hit (0 when no hits)."""
        return self.indexed / self.hits if self.hits else 0.0


@dataclass
class IndexResult:
    """Indexing solution for one crystal in a frame.

    Attributes
    ----------
    frame_index : int
        Index of the source frame.
    n_peaks : int
        Number of peaks fed to the indexer.
    n_indexed : int
        Number of peaks explained by the solution.
    rmsd : float
        Root-mean-square q residual over indexed peaks (A^-1).
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
        (N,) integrated peak intensity (ADU), aligned with ``positions``.
    sigmas : Tensor
        (N,) 1-sigma uncertainty on each intensity.
    loss_history : Tensor
        Per-iteration refinement loss (excluded from repr).
    predicted_hkl : Tensor, optional
        (M, 3) Miller indices of all reflections predicted to diffract on this
        frame, or None if prediction/integration was not run.
    predicted_positions : Tensor, optional
        (M, 2) predicted detector ``(row, col)`` positions, where each
        reflection is integrated.
    predicted_raw_positions : Tensor, optional
        Not set by probixi's integration, which no longer moves predictions.
        Kept for the database's ``pred_fs``/``pred_ss`` columns.
    predicted_intensities : Tensor, optional
        (M,) integrated intensity (ADU) per predicted reflection.
    predicted_sigmas : Tensor, optional
        (M,) 1-sigma uncertainty on each predicted intensity.
    predicted_peak : Tensor, optional
        (M,) largest background-subtracted pixel in each disk, the stream's
        ``peak`` column.
    predicted_background : Tensor, optional
        (M,) mean per-pixel background from each disk's annulus, the stream's
        ``background`` column.
    predicted_n_pixels : Tensor, optional
        (M,) pixels summed in each disk after masking and overlap ownership.
    predicted_bg_model : Tensor, optional
        (M,) noise-model background (ADU) summed over those pixels.
    predicted_bg_model_var : Tensor, optional
        (M,) variance of that sum (ADU^2) from the local-background annulus;
        zero when no local annulus is used.
    predicted_n_pixels_total : Tensor, optional
        (M,) pixels in each nominal signal disk, masked or not.
    predicted_n_masked : Tensor, optional
        (M,) of those, the pixels lost to the detector mask or the frame edge.
    predicted_n_overlap : Tensor, optional
        (M,) of those, the pixels also inside another predicted disk on the
        same panel, over every lattice on the frame. The stream writer skips a
        reflection with either count above zero; the database keeps it.
    peak_background_sum : Tensor, optional
        (N,) noise-model background (ADU) summed over each detected peak's blob.
    peak_n_pixels : Tensor, optional
        (N,) pixel count of each detected peak's blob.
    peak_n_masked : Tensor, optional
        (N,) pixels of each detected peak's blob that border the mask or the
        frame edge; the stream's peak list skips a peak with any.
    peak_posterior : Tensor, optional
        (N,) mean detection posterior of each detected peak's blob.
    predicted_peak_posterior : Tensor, optional
        Not set by probixi's integration; kept for the database's
        ``peak_posterior`` column.
    peak_aux : Tensor, optional
        (N, 2) ``(brightest pixel, eccentricity)`` of each detected peak.
    predicted_peak_max : Tensor, optional
        Not set by probixi's integration; kept for the database's ``peak_max``
        column.
    adu_per_photon : float, optional
        Gain (ADU per photon) the frame was processed with.
    diffraction_limit : float, optional
        Per-crystal resolution limit (nm^-1), the more generous of the
        second-highest-resolution indexed peak and ``falloff_limit``. Written as
        ``diffraction_resolution_limit``, so partialator's ``--push-res`` acts as
        on an indexamajig stream.
    falloff_limit : float, optional
        ``|q|`` (nm^-1) where the shell-mean ``I/sigma`` of the predicted
        reflections falls to ``IntegrateConfig.resolution_isigma``.
    enrichment : float, optional
        Rate at which predicted spots land on bright pixels, over the frame's
        bright-pixel rate; near 1 for a chance lattice.
    n_bright : int, optional
        Predicted spots on above-threshold signal.
    enrich_p : float, optional
        Chance probability of the bright predicted spots or of the signal at
        unexplained predictions, whichever is larger.
        :meth:`IndexStream.enrich_gate` keeps lattices with ``enrich_p <= alpha``.
    scale : float, optional
        Per-frame relative intensity scale (background fluence over the
        calibration reference), when inferred.
    scale_sigma : float, optional
        1-sigma uncertainty on ``scale``.
    mosaicity : float, optional
        Mosaic spread (rad), fitted or the prior.
    profile_radius : float, optional
        Ewald-shell half-width (A^-1) at the resolution limit; the writers
        convert it to nm^-1.
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
    predicted_raw_positions: Optional[Tensor] = None
    predicted_intensities: Optional[Tensor] = None
    predicted_sigmas: Optional[Tensor] = None
    predicted_peak: Optional[Tensor] = None
    predicted_background: Optional[Tensor] = None
    predicted_n_pixels: Optional[Tensor] = None
    predicted_bg_model: Optional[Tensor] = None
    predicted_bg_model_var: Optional[Tensor] = None
    predicted_n_pixels_total: Optional[Tensor] = None
    predicted_n_masked: Optional[Tensor] = None
    predicted_n_overlap: Optional[Tensor] = None
    peak_background_sum: Optional[Tensor] = None
    peak_n_pixels: Optional[Tensor] = None
    peak_n_masked: Optional[Tensor] = None
    peak_posterior: Optional[Tensor] = None
    predicted_peak_posterior: Optional[Tensor] = None
    peak_aux: Optional[Tensor] = None
    predicted_peak_max: Optional[Tensor] = None
    adu_per_photon: Optional[float] = None
    diffraction_limit: Optional[float] = None
    falloff_limit: Optional[float] = None
    enrichment: Optional[float] = None
    n_bright: Optional[int] = None
    enrich_p: Optional[float] = None
    scale: Optional[float] = None
    scale_sigma: Optional[float] = None
    mosaicity: Optional[float] = None
    profile_radius: Optional[float] = None
    # Held from integration to frame-wide overlap exclusion, then cleared
    _integration_positions: Optional[Tensor] = field(
        default=None, init=False, repr=False
    )
    _integration_valid: Optional[Tensor] = field(default=None, init=False, repr=False)
    _integration_radius: Optional[Tensor] = field(default=None, init=False, repr=False)


@dataclass
class FrameIndexResult:
    """One physical image and its accepted lattices.

    Attributes
    ----------
    frame_index : int
        Absolute input index.
    crystals : list of IndexResult
        Accepted lattices in search order.
    positions, intensities : Tensor
        Detected peaks supplied to the first lattice search.
    peak_background_sum, peak_n_pixels, peak_n_masked : Tensor, optional
        Per-peak background sum (ADU), pixel count, and count of pixels
        bordering the mask, aligned with ``positions``.
    peak_posterior : Tensor, optional
        Per-peak mean detection posterior.
    peak_aux : Tensor, optional
        Per-peak ``(brightest pixel, eccentricity)``.
    frame_stats : Tensor, optional
        ``(bg_low, bg_mid, bg_ring, bg_high, bg_edge, ice_score)`` of the frame.
    adu_per_photon : float, optional
        Gain (ADU per photon) the frame was processed with.
    """

    frame_index: int
    crystals: list[IndexResult]
    positions: Tensor
    intensities: Tensor
    peak_background_sum: Optional[Tensor] = None
    peak_n_pixels: Optional[Tensor] = None
    peak_n_masked: Optional[Tensor] = None
    peak_posterior: Optional[Tensor] = None
    peak_aux: Optional[Tensor] = None
    frame_stats: Optional[Tensor] = None
    adu_per_photon: Optional[float] = None

    @property
    def n_peaks(self) -> int:
        """Number of detected peaks before any lattice was removed."""
        return len(self.positions)


@dataclass
class SeedConfig:
    """Tolerances and limits for generating candidate orientations.

    Attributes
    ----------
    q_tolerance : float, optional
        Largest ``|q|`` mismatch (A^-1) for a peak to count as indexed. ``None``
        uses ``q_tolerance_fraction`` times the shortest reciprocal axis.
    q_tolerance_fraction : float
        Fraction of the smallest reciprocal spacing used when ``q_tolerance`` is None.
    max_candidates : int
        Candidate orientations carried into refinement.
    max_seed_peaks : int
        Brightest peaks used for seeding; refinement uses every peak.
    n_directions : int, optional
        Unsigned Fibonacci-sphere axes; both signs are searched. ``None`` scales
        6000 by ``(a / 55 A)**2``, at least 6000, to follow the fitness peak,
        which narrows as the a axis lengthens.
    n_spin : int
        Roll angles tried per kept direction.
    top_directions : int
        Unsigned axes kept; twice this many signed axes enter the roll stage.
    adaptive_sparse : bool
        Multiply the sphere-search counts by ``sparse_scale`` on frames with
        fewer than ``sparse_peak_threshold`` seeding peaks.
    sparse_peak_threshold : int
        Seeding-peak count below which a frame is sparse.
    max_index_peaks : int, optional
        Brightest peaks used for all of indexing, refinement included.
    rank_sigma : float, optional
        When set (A^-1), candidates are ranked by
        ``sum_i exp(-|A h_i - q_i|^2 / (2 rank_sigma^2))`` in place of the
        c-weighted inlier sum.
    max_lattices : int
        Successive lattices searched per frame.
    peel_radius : float, optional
        Residual (A^-1) below which a peak is removed before the next lattice
        search; ``None`` uses the indexing tolerance.
    sparse_scale : float
        Multiplier on the sphere-search counts on sparse frames.
    """

    q_tolerance: Optional[float] = None
    q_tolerance_fraction: float = 0.25
    max_candidates: int = 64
    max_seed_peaks: int = 80
    n_directions: Optional[int] = None
    n_spin: int = 120
    top_directions: int = 32
    adaptive_sparse: bool = True
    sparse_peak_threshold: int = 30
    sparse_scale: float = 2.0
    max_index_peaks: Optional[int] = None
    rank_sigma: Optional[float] = None
    max_lattices: int = 1
    peel_radius: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_lattices < 1:
            raise ValueError("max_lattices must be positive")
        if (
            self.max_index_peaks is not None
            and self.max_index_peaks < MIN_PEAKS_PER_LATTICE
        ):
            raise ValueError(
                f"max_index_peaks must be at least {MIN_PEAKS_PER_LATTICE}"
            )
        for name in ("rank_sigma", "peel_radius"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class RefineConfig:
    """Settings for the orientation and cell refinement.

    Attributes
    ----------
    lr : float
        Adam learning rate on the axis-angle perturbation.
    max_iters : int
        Adam steps.
    reassign_every : int
        Steps between reassignments of peak hkl and inliers.
    min_indexed : int
        Fewest indexed peaks a candidate needs to be accepted.
    cell : bool
        Refine the winning candidate's rotation and the cell parameters its
        lattice type leaves free by Levenberg-Marquardt.
    cell_iters : int
        Levenberg-Marquardt steps per round; two rounds, with hkl reassigned
        between them.
    """

    lr: float = 1e-3
    max_iters: int = 200
    reassign_every: int = 10
    min_indexed: int = 6
    cell: bool = True
    cell_iters: int = 8


@dataclass(frozen=True)
class IntegrateConfig:
    """Settings for predicting and integrating reflections after indexing.

    Attributes
    ----------
    enabled : bool
        Predict and integrate the full lattice. When off, the stream lists only
        the indexed peaks.
    partiality_threshold : float
        Floor on the tolerance on the excitation error ``|S| - 1``; the rocking
        model sets the tolerance above it.
    mosaicity_from_data : bool
        Fit each crystal's mosaicity to the radial Ewald offsets of its indexed
        reflections. When off, every crystal uses ``mosaicity_deg``.
    mosaicity_deg : float
        Mosaic spread (deg) when the fit is off or has fewer than eight
        reflections.
    mosaicity_min_deg, mosaicity_max_deg : float
        Bounds on the fitted mosaicity (deg).
    bandwidth : float
        Beam bandwidth ``dlambda / lambda``, which adds a ``|q|^2`` term to the
        rocking width: about 1e-4 at a synchrotron, 2e-3 for SASE. ``0`` leaves
        it to the mosaicity fit.
    predict_sigma : float
        Rocking half-widths ``rho(|q|)`` to predict out to; the tolerance on
        ``|S| - 1`` is ``wavelength * predict_sigma * rho(|q|)``.
    domain_size_recip : float
        Constant (domain-size) term of the rocking width (A^-1).
    resolution_isigma : float
        Shell-mean ``I/sigma`` that defines ``IndexResult.falloff_limit``.
    resolution_nbins : int
        Equal-count shells in the falloff scan.
    resolution_min_refl : int
        Fewest measured reflections for a falloff estimate.
    resolution_percentile : float
        Quantile of the indexed peaks' ``|q|`` used as the diffraction limit
        when neither the second-highest indexed peak nor the falloff crossing is
        available. ``<= 0`` reports no limit. Every reflection is integrated to
        the detector edge whatever the limit, so the merge sets the cutoff.
    resolution_snr_floor : float
        Smallest peak ``I/sigma`` counted by ``resolution_percentile``; ``<= 0``
        counts every indexed peak, as does a crystal with none above the floor.
    adu_per_photon : float, optional
        Gain for the shot-noise term of ``sigma(I)``. ``None`` takes the
        measured photon-transfer gain, else the geometry's
        ``adu_per_eV * photon_energy``, else 1.
    aperture : {"snr", "flux", "auto"}
        Signal-disk radius. ``"snr"`` maximises the background-limited
        signal-to-noise of the disk sum. ``"flux"`` is where the spot profile
        falls to 2% of its centre. ``"auto"`` keeps ``"snr"`` unless, on the
        calibration peaks, its captured fraction trends with resolution
        differently between crystals and the ``"flux"`` disk costs less noise
        than that spread.
    aperture_tolerance : float
        ``"auto"`` only: between-crystal spread of the captured fraction above
        which ``"flux"`` is considered.
    """

    enabled: bool = True
    partiality_threshold: float = 0.0005
    mosaicity_from_data: bool = True
    mosaicity_deg: float = 0.1
    mosaicity_min_deg: float = 0.02
    mosaicity_max_deg: float = 0.8
    bandwidth: float = 0.0
    predict_sigma: float = 1.5
    domain_size_recip: float = 5.0e-5
    resolution_isigma: float = 0.5
    resolution_nbins: int = 10
    resolution_min_refl: int = 40
    resolution_percentile: float = 0.90
    resolution_snr_floor: float = 0.0
    adu_per_photon: Optional[float] = None
    aperture: str = "auto"
    aperture_tolerance: float = 0.02


@dataclass
class CellMatchConfig:
    """Tolerances for accepting a recovered cell as matching the target.

    Attributes
    ----------
    edge_tolerance : float
        Largest fractional difference on sorted cell edges.
    angle_tolerance_deg : float
        Largest absolute difference (deg) on sorted cell angles.
    """

    edge_tolerance: float = 0.05
    angle_tolerance_deg: float = 3.0

    @property
    def angle_tolerance_rad(self) -> float:
        """``angle_tolerance_deg`` in radians."""
        return math.radians(self.angle_tolerance_deg)


class IndexStream(Generic[_ResultT]):
    """Lazy, single-use stream of :class:`IndexResult`.

    Returned by :meth:`Indexer.index_stream`. ``map``, ``filter``, and ``tap``
    compose lazily and keep the stream type; ``collect``, ``to_stream``,
    ``to_db``, ``count``, and ``for_each`` drain it. :class:`FrameIndexStream`
    yields one :class:`FrameIndexResult` per frame instead. Every composed
    stream shares one :class:`IndexStats`, updated as frames are drawn.
    """

    def __init__(self, source: Iterable[_ResultT], stats: Optional[IndexStats] = None):
        self._source: Iterator[_ResultT] = iter(source)
        self.stats = stats if stats is not None else IndexStats()

    def map(self, fn: Callable[[_ResultT], _ResultT]) -> "IndexStream[_ResultT]":
        """Apply ``fn`` to each element lazily."""
        return type(self)((fn(r) for r in self._source), stats=self.stats)

    def filter(self, predicate: Callable[[_ResultT], bool]) -> "IndexStream[_ResultT]":
        """Keep the elements for which ``predicate`` is true, lazily."""
        return type(self)((r for r in self._source if predicate(r)), stats=self.stats)

    def tap(self, fn: Callable[[_ResultT], None]) -> "IndexStream[_ResultT]":
        """Call ``fn`` on each element as it passes, lazily."""

        def _gen() -> Iterator[_ResultT]:
            for r in self._source:
                fn(r)
                yield r

        return type(self)(_gen(), stats=self.stats)

    def enrich_gate(self, alpha: float = 1e-3) -> "IndexStream[_ResultT]":
        """Drop solutions whose predicted spots are not backed by image signal.

        Parameters
        ----------
        alpha : float, default 1e-3
            Largest ``enrich_p`` kept, a per-lattice false-positive rate.

        Returns
        -------
        IndexStream
            Filtered stream.
        """
        return self.filter(lambda r: r.enrich_p is not None and r.enrich_p <= alpha)

    def to_stream(self, writer: Callable[[_ResultT], None]) -> int:
        """Drain the stream into ``writer``, such as a :class:`DataOffloader`.

        Returns
        -------
        int
            Elements written.
        """
        n = 0
        for r in self._source:
            writer(r)
            n += 1
        return n

    def to_db(
        self,
        path,
        *,
        geometry: dict,
        cell: Optional[CellParams] = None,
        geometry_file=None,
        files: Optional[dict] = None,
        panel: str = "0",
    ) -> int:
        """Drain the stream into a DuckDB database at ``path``.

        Wraps :class:`~probixi.io.db.DuckDBOffloader`, which writes the
        ``geometry``, ``panels``, and ``cell`` tables, then every result.

        Parameters
        ----------
        path : str or Path
            Destination ``.duckdb`` file, overwritten if present.
        geometry : dict
            Indexer geometry (``beam_center``, ``clen``, ...).
        cell : CellParams, optional
            Target unit cell for the ``cell`` table.
        geometry_file : str or Path, optional
            Geometry file stored verbatim.
        files : dict, optional
            Loader file map, used to add ``frames`` rows for frames that never
            reached the indexer.
        panel : str, default "0"
            Panel name for peaks and reflections outside every panel.

        Returns
        -------
        int
            Elements written.
        """
        from ..io.db import DuckDBOffloader

        with DuckDBOffloader(
            path,
            geometry=geometry,
            cell=cell,
            geometry_file=geometry_file,
            files=files,
            panel=panel,
        ) as off:
            return self.to_stream(off)

    def collect(self) -> list[_ResultT]:
        """Drain the stream into a list."""
        return list(self._source)

    def count(self) -> int:
        """Drain the stream and return the number of elements."""
        return sum(1 for _ in self._source)

    def for_each(self, fn: Callable[[_ResultT], None]) -> None:
        """Drain the stream, calling ``fn`` on each element."""
        for r in self._source:
            fn(r)

    def __iter__(self) -> Iterator[_ResultT]:
        return self._source


class FrameIndexStream(IndexStream["FrameIndexResult"]):
    """Lazy, single-use stream of :class:`FrameIndexResult`, one per frame."""

    def flatten(self) -> IndexStream["IndexResult"]:
        """Yield each accepted lattice, keeping the shared frame statistics.

        Returns
        -------
        IndexStream
            Lattices in input order.
        """
        return IndexStream(
            (r for frame in self for r in frame.crystals), stats=self.stats
        )

    def enrich_gate(self, alpha: float = 1e-3) -> "IndexStream[FrameIndexResult]":
        """Unsupported; pass ``enrich_alpha`` to :meth:`Indexer.index_frame_stream`.

        Parameters
        ----------
        alpha : float, default 1e-3
            Ignored.

        Raises
        ------
        ValueError
            Always. Overlap exclusion has already run, and dropping a lattice
            now cannot restore the reflections its disks excluded from the
            lattices kept.
        """
        raise ValueError(
            "pass enrich_alpha to index_frame_stream before overlap exclusion"
        )


def _same_lattice(A: Tensor, B: Tensor) -> bool:
    # same points in any setting: B ~ A @ T with T integral and unimodular
    A, B = A.double(), B.double()
    T = torch.linalg.solve(A, B).round()
    if round(abs(float(torch.linalg.det(T)))) != 1:
        return False
    return float((A @ T - B).norm() / A.norm()) < DUPLICATE_LATTICE_TOL


class Indexer:
    """Recover crystal orientations of a known cell from detector peaks.

    Parameters
    ----------
    geometry : dict
        Detector geometry with ``beam_center`` (row, col px), ``clen`` (m),
        ``pixel_size`` (m), and ``wavelength`` (A).
    target_cell : CellParams
        Unit cell that accepted lattices must match.
    seed : SeedConfig, optional
        Orientation-search settings.
    refine : RefineConfig, optional
        Refinement settings.
    cell_match : CellMatchConfig, optional
        Cell-acceptance tolerances.
    integrate : IntegrateConfig, optional
        Prediction and integration settings.
    dtype : torch.dtype, optional
        Lattice-math dtype, ``torch.float32`` by default.
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
        dtype: Optional[torch.dtype] = None,
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
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device
        self._geometry_gain = geometry.get("adu_per_photon")
        self._measured_gain: Optional[float] = None
        self._bg_annulus_pixels: Optional[float] = None
        self._measured_radii: Optional[tuple[float, float, float]] = None
        self._resolved_aperture: Optional[str] = None
        self._q_max: Optional[float] = None
        # Per-frame wavelengths (A) by frame index; None uses the geometry's
        self.frame_wavelengths: Optional[list[float]] = None
        # learned |q|-shell disk radii, and their profile sums
        self._radius_by_q: Optional[tuple[Tensor, list[float]]] = None
        self._shell_capture: Optional[list[float]] = None
        self._profile_acc: Optional[list] = None
        self.set_target_cell(target_cell)

    def _geometry_at(self, frame_index: Optional[int]) -> dict:
        wl = self.frame_wavelengths
        if wl is None or frame_index is None or not 0 <= frame_index < len(wl):
            return self.geometry
        return {**self.geometry, "wavelength": wl[frame_index]}

    def set_target_cell(self, cell: CellParams) -> None:
        """Re-centre indexing on ``cell``; absent symmetry labels keep their values."""
        prev = getattr(self, "target_cell", None)
        self.target_cell = CellParams(
            cell.a,
            cell.b,
            cell.c,
            cell.alpha,
            cell.beta,
            cell.gamma,
            lattice_type=cell.lattice_type
            or (prev.lattice_type if prev is not None else None),
            unique_axis=cell.unique_axis
            or (prev.unique_axis if prev is not None else None),
            centering=cell.centering or (prev.centering if prev is not None else None),
        )
        self.B_target = cell_to_B(
            self.target_cell, device=self.device, dtype=self.dtype
        )
        if self.seed.q_tolerance is not None:
            self.q_tolerance = float(self.seed.q_tolerance)
        else:
            min_spacing = float(torch.linalg.vector_norm(self.B_target, dim=0).min())
            self.q_tolerance = self.seed.q_tolerance_fraction * min_spacing
        self.n_directions = self.seed.n_directions or max(
            6000, round(6000 * (self.target_cell.a / 55.0) ** 2)
        )

    def _cell_matches_target(
        self, cell: CellParams, target: Optional[CellParams] = None
    ) -> bool:
        # Sorted edges and angles make the match independent of axis labelling
        tc = target or self.target_cell
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
        frame_index: Optional[int] = None,
    ) -> Tensor:
        """Lift (N, 2) ``(row, col)`` positions to q (N, 3) in A^-1."""
        return detector_to_q(
            positions.to(device=self.device, dtype=self.dtype),
            self._geometry_at(frame_index),
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
        peak_bg_by_frame: Optional[dict[int, Tensor]] = None,
        peak_npix_by_frame: Optional[dict[int, Tensor]] = None,
        peak_nmask_by_frame: Optional[dict[int, Tensor]] = None,
        peak_aux_by_frame: Optional[dict[int, Tensor]] = None,
    ) -> dict[int, IndexResult]:
        """Index one lattice per frame, refining every frame in one batch.

        Each frame is lifted and seeded on its own; one Adam run then refines
        the candidates of all frames. Frames with fewer than four peaks are
        skipped.

        Parameters
        ----------
        positions_by_frame : dict of int -> Tensor
            Per-frame (N, 2) peak ``(row, col)`` positions.
        frame_rotations : dict of int -> Tensor, optional
            Per-frame (3, 3) lab-to-crystal rotations.
        intensities_by_frame, sigmas_by_frame : dict of int -> Tensor, optional
            Per-peak intensities and 1-sigma uncertainties, aligned with
            ``positions_by_frame``; zeros when omitted.
        weights_by_frame : dict of int -> Tensor, optional
            Per-peak detection posteriors, which weight seeding and refinement.
        peak_bg_by_frame, peak_npix_by_frame, peak_nmask_by_frame : dict of int -> Tensor, optional
            Per-peak background sum (ADU), pixel count, and mask-bordering pixel
            count, carried onto the result for the writers.
        peak_aux_by_frame : dict of int -> Tensor, optional
            Per-peak ``(brightest pixel, eccentricity)``, carried likewise.

        Returns
        -------
        dict of int -> IndexResult
            One entry per indexed frame.
        """
        if not positions_by_frame:
            return {}

        data = {
            i: self._cap_peak_data(
                p,
                (intensities_by_frame or {}).get(i),
                (sigmas_by_frame or {}).get(i),
                (weights_by_frame or {}).get(i),
                (peak_bg_by_frame or {}).get(i),
                (peak_npix_by_frame or {}).get(i),
                (peak_nmask_by_frame or {}).get(i),
                (peak_aux_by_frame or {}).get(i),
            )
            for i, p in positions_by_frame.items()
        }
        seeded: list[tuple[int, int, Tensor, Tensor, Optional[Tensor]]] = []
        for idx, (positions, intensities, _, weights, *_) in data.items():
            if (
                positions.ndim != 2
                or positions.shape[-1] != 2
                or positions.shape[0] < 4
            ):
                continue
            rotation = (frame_rotations or {}).get(idx)
            q = self.lift(positions, frame_rotation=rotation, frame_index=idx)
            with torch.no_grad():
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
            weights_per_frame=[s[4] for s in seeded] if has_weights else None,
        )

        out: dict[int, IndexResult] = {}
        for (idx, n_peaks, q, _, _), rr in zip(seeded, results):
            built = self._build_indexing_result(
                rr,
                frame_index=idx,
                n_peaks=n_peaks,
                positions=data[idx][0],
                q=q,
                intensities=data[idx][1],
                sigmas=data[idx][2],
                peak_bg=data[idx][4],
                peak_npix=data[idx][5],
                peak_nmask=data[idx][6],
                peak_posterior=data[idx][3],
                peak_aux=data[idx][7],
            )
            if built is not None:
                out[idx] = built
        return out

    def _seed_keep_indices(
        self, q: Tensor, intensities: Optional[Tensor]
    ) -> Optional[Tensor]:
        # The brightest max_seed_peaks, or the lowest |q| without intensities
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
            self.n_directions,
            self.seed.n_spin,
            self.seed.top_directions,
            self.seed.max_candidates,
        )
        # Widens the search on frames with few peaks
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
        q: Optional[Tensor] = None,
        peak_bg: Optional[Tensor] = None,
        peak_npix: Optional[Tensor] = None,
        peak_nmask: Optional[Tensor] = None,
        peak_posterior: Optional[Tensor] = None,
        peak_aux: Optional[Tensor] = None,
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
        if self.seed.rank_sigma is not None:
            q = self.lift(positions, frame_index=frame_index) if q is None else q
            predicted = torch.einsum("cij,cnj->cni", result.A, result.hkl.to(result.A))
            soft = torch.exp(
                -((predicted - q).square().sum(-1)) / (2 * self.seed.rank_sigma**2)
            ).sum(-1)
        rmsd = result.rmsd
        # Ranks by c-weighted inliers; the rmsd term, at most 1e-3, breaks near-ties
        score = soft - rmsd / (rmsd.max().clamp_min(1e-12) * 1e3)
        ranking = torch.argsort(score, descending=True).tolist()
        n_indexed_host = n_indexed.tolist()
        for cand in ranking:
            if n_indexed_host[cand] < self.refine.min_indexed:
                continue
            A_cand = result.A[cand]
            hkl_c, indexed_c = result.hkl[cand], result.indexed[cand]
            n_c, rmsd_c = n_indexed_host[cand], float(rmsd[cand])
            if self.refine.cell:
                q = self.lift(positions, frame_index=frame_index) if q is None else q
                A_cand, hkl_c, indexed_c, n_c, rmsd_c = self._refine_cell(
                    A_cand, q, hkl_c, indexed_c, n_c, rmsd_c, frame_index
                )
            try:
                cell = B_to_cell(A_cand)
                B = cell_to_B(cell, device=A_cand.device, dtype=A_cand.dtype)
                U = A_cand @ torch.linalg.inv(B)
            except Exception:
                continue
            if not self._cell_matches_target(cell):
                continue
            return IndexResult(
                frame_index=frame_index,
                n_peaks=n_peaks,
                n_indexed=n_c,
                rmsd=rmsd_c,
                A=A_cand,
                U=U,
                B=B,
                cell=cell,
                indexed_mask=indexed_c,
                hkl=hkl_c,
                positions=positions,
                intensities=intensities,
                sigmas=sigmas,
                loss_history=result.history,
                peak_background_sum=peak_bg,
                peak_n_pixels=peak_npix,
                peak_n_masked=peak_nmask,
                peak_posterior=peak_posterior,
                peak_aux=peak_aux,
                adu_per_photon=self._adu_per_photon(),
            )
        return None

    def _refine_cell(
        self,
        A: Tensor,
        q: Tensor,
        hkl: Tensor,
        indexed: Tensor,
        n_indexed: int,
        rmsd: float,
        frame_index: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, Tensor, int, float]:
        # Keeps the Adam solution unless LM improves it and still matches the target
        out = refine_cell(
            A,
            q,
            hkl,
            indexed,
            self.target_cell,
            self.q_tolerance,
            iters=self.refine.cell_iters,
            edge_tolerance=self.cell_match.edge_tolerance,
            angle_tolerance_rad=self.cell_match.angle_tolerance_rad,
            wavelength=float(self._geometry_at(frame_index)["wavelength"]),
        )
        if out is None:
            return A, hkl, indexed, n_indexed, rmsd
        A_r, hkl_r, indexed_r, rmsd_r, improved = out
        n_r = int(indexed_r.sum())
        if not improved or n_r < self.refine.min_indexed:
            return A, hkl, indexed, n_indexed, rmsd
        try:
            if not self._cell_matches_target(B_to_cell(A_r)):
                return A, hkl, indexed, n_indexed, rmsd
        except Exception:
            return A, hkl, indexed, n_indexed, rmsd
        return A_r, hkl_r, indexed_r, n_r, rmsd_r

    def _positions_from_frame(self, r: PeakResult) -> tuple[Optional[Tensor], ...]:
        # Pull kept peak-blob centroids and photometry off a PeakResult
        stats = r.kept_stats
        positions = torch.stack([stats.row_centroid, stats.col_centroid], dim=-1).to(
            device=self.device, dtype=self.dtype
        )
        intensities = stats.intensity_sum.to(device=self.device, dtype=self.dtype)
        sigmas = stats.intensity_sigma.to(device=self.device, dtype=self.dtype)
        weights = stats.posterior_mean.to(device=self.device, dtype=self.dtype)
        bg_sum = stats.background_sum.to(device=self.device, dtype=self.dtype)
        n_pix = stats.size.to(device=self.device, dtype=self.dtype)
        n_mask = stats.n_masked
        if n_mask is not None:
            n_mask = n_mask.to(device=self.device, dtype=self.dtype)
        aux = torch.stack([stats.intensity_max, stats.eccentricity], dim=-1).to(
            device=self.device, dtype=self.dtype
        )
        return self._cap_peak_data(
            positions, intensities, sigmas, weights, bg_sum, n_pix, n_mask, aux
        )

    def _cap_peak_data(
        self,
        positions: Tensor,
        intensities: Optional[Tensor],
        sigmas: Optional[Tensor],
        weights: Optional[Tensor],
        peak_bg: Optional[Tensor] = None,
        peak_npix: Optional[Tensor] = None,
        peak_nmask: Optional[Tensor] = None,
        peak_aux: Optional[Tensor] = None,
    ) -> tuple[Optional[Tensor], ...]:
        # Keeps the max_index_peaks brightest peaks in each array aligned with positions
        values = (
            positions,
            intensities,
            sigmas,
            weights,
            peak_bg,
            peak_npix,
            peak_nmask,
            peak_aux,
        )
        cap = self.seed.max_index_peaks
        if cap is None or len(positions) <= cap:
            return values
        keep = (
            intensities.topk(cap).indices
            if intensities is not None
            else torch.arange(cap, device=positions.device)
        )
        return tuple(v[keep] if v is not None else None for v in values)

    def index_lattices(
        self,
        positions_by_frame: dict[int, Tensor],
        *,
        intensities_by_frame: dict[int, Tensor],
        sigmas_by_frame: dict[int, Tensor],
        weights_by_frame: dict[int, Tensor],
        frame_rotations: Optional[dict[int, Tensor]] = None,
        peak_bg_by_frame: Optional[dict[int, Tensor]] = None,
        peak_npix_by_frame: Optional[dict[int, Tensor]] = None,
        peak_nmask_by_frame: Optional[dict[int, Tensor]] = None,
        peak_aux_by_frame: Optional[dict[int, Tensor]] = None,
        maps_by_frame: Optional[dict[int, tuple]] = None,
        bright_threshold: float = 5.0,
        enrich_alpha: Optional[float] = None,
    ) -> dict[int, list[IndexResult]]:
        """Index up to ``SeedConfig.max_lattices`` lattices per frame.

        After each search the peaks an accepted lattice explains are removed,
        and the rest are searched again while at least six peaks remain on
        each side.

        Parameters
        ----------
        positions_by_frame : dict of int -> Tensor
            Per-frame (N, 2) peak ``(row, col)`` positions.
        intensities_by_frame, sigmas_by_frame, weights_by_frame : dict of int -> Tensor
            Peak intensities, sigmas, and detection posteriors aligned with
            ``positions_by_frame``.
        frame_rotations : dict of int -> Tensor, optional
            Per-frame (3, 3) lab-to-crystal rotations.
        peak_bg_by_frame, peak_npix_by_frame, peak_nmask_by_frame : dict of int -> Tensor, optional
            Per-peak photometry as for :meth:`index_frames`, removed with the
            peaks so each lattice keeps its own.
        peak_aux_by_frame : dict of int -> Tensor, optional
            As for :meth:`index_frames`.
        maps_by_frame : dict of int -> tuple, optional
            Per-frame ``(excess, var, mask, mean)`` maps. With ``enrich_alpha``,
            each lattice is integrated and gated as it is found; a rejected
            lattice is dropped, but its peaks are still removed.
        bright_threshold, enrich_alpha
            As for :meth:`index_frame_stream`.

        Returns
        -------
        dict of int -> list of IndexResult
            Accepted lattices keyed by frame index, in search order.
        """
        remaining = {
            i: self._cap_peak_data(
                p,
                intensities_by_frame[i],
                sigmas_by_frame[i],
                weights_by_frame[i],
                (peak_bg_by_frame or {}).get(i),
                (peak_npix_by_frame or {}).get(i),
                (peak_nmask_by_frame or {}).get(i),
                (peak_aux_by_frame or {}).get(i),
            )
            for i, p in positions_by_frame.items()
        }
        output = {}
        for n in range(1, self.seed.max_lattices + 1):
            if not remaining:
                break
            results = self.index_frames(
                {i: v[0] for i, v in remaining.items()},
                frame_rotations=frame_rotations,
                intensities_by_frame={i: v[1] for i, v in remaining.items()},
                sigmas_by_frame={i: v[2] for i, v in remaining.items()},
                weights_by_frame={i: v[3] for i, v in remaining.items()},
                peak_bg_by_frame={i: v[4] for i, v in remaining.items()},
                peak_npix_by_frame={i: v[5] for i, v in remaining.items()},
                peak_nmask_by_frame={i: v[6] for i, v in remaining.items()},
                peak_aux_by_frame={i: v[7] for i, v in remaining.items()},
            )
            next_remaining = {}
            for i, result in results.items():
                previous = output.setdefault(i, [])
                # Skips a lattice found again, in any setting
                if any(_same_lattice(r.A, result.A) for r in previous):
                    continue
                maps = (maps_by_frame or {}).get(i)
                gated = enrich_alpha is not None and maps is not None
                if gated:
                    self._integrate_result(
                        result, *maps, bright_threshold, enrich_alpha=enrich_alpha
                    )
                if not gated or (
                    result.enrich_p is not None and result.enrich_p <= enrich_alpha
                ):
                    previous.append(result)
                if n == self.seed.max_lattices:
                    continue
                # Removes the peaks this lattice explains and searches the rest
                q = self.lift(
                    result.positions,
                    frame_rotation=(frame_rotations or {}).get(i),
                    frame_index=i,
                )
                hkl = torch.round(q @ torch.linalg.inv(result.A).T)
                assigned = (hkl @ result.A.T - q).norm(dim=-1) < (
                    self.seed.peel_radius or self.q_tolerance
                )
                if (
                    int(assigned.sum()) >= MIN_PEAKS_PER_LATTICE
                    and int((~assigned).sum()) >= MIN_PEAKS_PER_LATTICE
                ):
                    next_remaining[i] = tuple(
                        None if v is None else v[~assigned] for v in remaining[i]
                    )
            remaining = next_remaining
        return output

    def index_stream(
        self,
        peak_stream: Iterable[PeakResult],
        batch_size: int = 8,
        frame_rotations: Optional[dict[int, Tensor]] = None,
        bright_threshold: float = 5.0,
        enrich_alpha: Optional[float] = None,
    ) -> IndexStream["IndexResult"]:
        """Index a peak stream lazily, one element per accepted lattice.

        Parameters
        ----------
        peak_stream, batch_size, frame_rotations, bright_threshold, enrich_alpha
            As for :meth:`index_frame_stream`.

        Returns
        -------
        IndexStream
            Accepted lattices, sharing the statistics of the underlying
            :class:`FrameIndexStream`.
        """
        return self.index_frame_stream(
            peak_stream, batch_size, frame_rotations, bright_threshold, enrich_alpha
        ).flatten()

    def index_frame_stream(
        self,
        peak_stream: Iterable[PeakResult],
        batch_size: int = 8,
        frame_rotations: Optional[dict[int, Tensor]] = None,
        bright_threshold: float = 5.0,
        enrich_alpha: Optional[float] = None,
    ) -> "FrameIndexStream":
        """Index a peak stream lazily, one element per input frame.

        Parameters
        ----------
        peak_stream : iterable of PeakResult
            Per-frame peak results, with their pixel maps.
        batch_size : int, default 8
            Frames per refinement batch; their pixel maps are held until the
            batch is yielded.
        frame_rotations : dict of int -> Tensor, optional
            Per-frame (3, 3) lab-to-crystal rotations.
        bright_threshold : float, default 5.0
            Whitened threshold that marks a pixel bright for the enrichment test.
        enrich_alpha : float, optional
            Largest ``enrich_p`` kept, applied to each lattice before overlap
            exclusion; ``None`` disables the gate.

        Returns
        -------
        FrameIndexStream
            One result per input frame, unindexed frames included. With
            ``IntegrateConfig.enabled`` and pixel maps on the peak results, each
            accepted lattice carries its integrated predicted reflections.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        stats = IndexStats()

        def _flush(buf: list[dict]) -> Iterator[FrameIndexResult]:
            hits = [b for b in buf if len(b["pos"]) >= MIN_PEAKS_TO_INDEX]
            results = self.index_lattices(
                {b["idx"]: b["pos"] for b in hits},
                frame_rotations=frame_rotations,
                intensities_by_frame={b["idx"]: b["I"] for b in hits},
                sigmas_by_frame={b["idx"]: b["sig"] for b in hits},
                weights_by_frame={b["idx"]: b["w"] for b in hits},
                peak_bg_by_frame={b["idx"]: b["peak_bg"] for b in hits},
                peak_npix_by_frame={b["idx"]: b["peak_npix"] for b in hits},
                peak_nmask_by_frame={b["idx"]: b["peak_nmask"] for b in hits},
                peak_aux_by_frame={b["idx"]: b["peak_aux"] for b in hits},
                maps_by_frame={
                    b["idx"]: (b["excess"], b["var"], b["mask"], b["mean"])
                    for b in hits
                    if self.integrate.enabled and b["excess"] is not None
                },
                bright_threshold=bright_threshold,
                enrich_alpha=enrich_alpha,
            )
            for b in buf:
                crystals = results.get(b["idx"], [])
                if self.integrate.enabled and b["excess"] is not None:
                    for result in (r for r in crystals if r.enrich_p is None):
                        self._integrate_result(
                            result,
                            b["excess"],
                            b["var"],
                            b["mask"],
                            b["mean"],
                            bright_threshold,
                            enrich_alpha=enrich_alpha,
                        )
                if enrich_alpha is not None:
                    crystals = [
                        r
                        for r in crystals
                        if r.enrich_p is not None and r.enrich_p <= enrich_alpha
                    ]
                self._flag_overlaps(crystals)
                stats.indexed += bool(crystals)
                stats.crystals += len(crystals)
                yield FrameIndexResult(
                    b["idx"],
                    crystals,
                    b["pos"],
                    b["I"],
                    peak_background_sum=b["peak_bg"],
                    peak_n_pixels=b["peak_npix"],
                    peak_n_masked=b["peak_nmask"],
                    peak_posterior=b["w"],
                    peak_aux=b["peak_aux"],
                    frame_stats=self._frame_stats(b),
                    adu_per_photon=self._adu_per_photon(),
                )

        def _gen() -> Iterator[FrameIndexResult]:
            buf: list[dict] = []
            for r in peak_stream:
                stats.frames += 1
                stats.hits += len(r) >= MIN_PEAKS_TO_INDEX
                (
                    positions,
                    intensities,
                    sigmas,
                    weights,
                    peak_bg,
                    peak_npix,
                    peak_nmask,
                    peak_aux,
                ) = self._positions_from_frame(r)
                buf.append(
                    dict(
                        idx=r.frame_index or 0,
                        pos=positions,
                        I=intensities,
                        sig=sigmas,
                        w=weights,
                        peak_bg=peak_bg,
                        peak_npix=peak_npix,
                        peak_nmask=peak_nmask,
                        peak_aux=peak_aux,
                        excess=r.scores.get("excess") if r.scores else None,
                        var=r.var,
                        mask=r.valid_mask,
                        mean=r.mean,
                    )
                )
                if len(buf) >= batch_size:
                    yield from _flush(buf)
                    buf = []
            if buf:
                yield from _flush(buf)

        return FrameIndexStream(_gen(), stats=stats)

    def _resolution_maps(self, shape: tuple[int, int], device) -> tuple[Tensor, Tensor]:
        # Band and ice-window maps at the geometry wavelength, per frame shape
        maps = self.__dict__.setdefault("_res_maps", {})
        key = (tuple(shape), str(device))
        if key not in maps:
            rows, cols = shape
            grid = torch.stack(
                torch.meshgrid(
                    torch.arange(rows, device=device, dtype=torch.float32),
                    torch.arange(cols, device=device, dtype=torch.float32),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
            d = torch.empty(len(grid), device=device, dtype=torch.float32)
            for start in range(0, len(grid), 1 << 20):
                q = detector_to_q(grid[start : start + (1 << 20)], self.geometry)
                d[start : start + (1 << 20)] = 1.0 / q.norm(dim=-1).clamp_min(1e-6)
            maps[key] = resolution_bands(d.view(rows, cols))
        return maps[key]

    def _frame_stats(self, b: dict) -> Optional[Tensor]:
        mean = b.get("mean")
        if mean is None or mean.ndim not in (2, 3):
            return None

        def img(t):
            return None if t is None else (t[0] if t.ndim == 3 else t)

        mean = img(mean)
        band, ice = self._resolution_maps(tuple(mean.shape), mean.device)
        return frame_background_stats(
            mean, img(b.get("excess")), img(b.get("var")), img(b.get("mask")), band, ice
        )

    def _flag_overlaps(self, crystals: list[IndexResult]) -> None:
        # disk pixels shared with any predicted disk on the panel, across all lattices
        active = [r for r in crystals if r._integration_positions is not None]
        if not active:
            return
        positions = torch.cat([r._integration_positions for r in active])
        # panel 0 is off-panel; placements are (min_ss, max_ss, min_fs, max_fs)
        panel = torch.zeros(len(positions), dtype=torch.long, device=positions.device)
        panels = self.geometry.get("panels") or {}
        if panels:
            ss_fs = torch.tensor(
                [
                    [d["min_ss"], d["max_ss"], d["min_fs"], d["max_fs"]]
                    for d in panels.values()
                ],
                dtype=positions.dtype,
                device=positions.device,
            )
            rows, cols = positions[:, :1], positions[:, 1:]
            inside = (
                (rows >= ss_fs[:, 0])
                & (rows <= ss_fs[:, 1])
                & (cols >= ss_fs[:, 2])
                & (cols <= ss_fs[:, 3])
            )
            ids = torch.arange(1, len(panels) + 1, device=positions.device)
            panel = torch.where(inside, ids, 0).amax(1)
        radius = torch.cat(
            [
                (
                    r._integration_radius
                    if r._integration_radius is not None
                    else positions.new_full(
                        (len(r._integration_positions),), self._radii()[0]
                    )
                )
                for r in active
            ]
        )
        shared = disk_overlap_counts(positions, radius, panel)
        offset = 0
        for r in active:
            count = len(r._integration_positions)
            r.predicted_n_overlap = shared[offset : offset + count][
                r._integration_valid
            ]
            offset += count
            r._integration_positions = r._integration_valid = None
            r._integration_radius = None

    def _reflection_radii(self, q: Tensor) -> tuple:
        # The learned shell radius for every reflection; the base disk until learned
        base = self._radii()
        if self._radius_by_q is None:
            return base
        edges, shell = self._radius_by_q
        k = torch.bucketize(q, edges.to(q)).clamp(1, len(shell)) - 1
        r = torch.tensor(shell, dtype=q.dtype, device=q.device)[k]
        r_in = torch.clamp(r + 1.0, min=base[1])
        # r_out keeps the base annulus area
        return r, r_in, (r_in * r_in + base[2] ** 2 - base[1] ** 2).sqrt()

    def _stack_profiles(self, excess, var, valid, positions, pred) -> None:
        # Stacks excess, clipped at 10 sigma, around the predictions per |q| shell
        if not len(positions):
            return
        if not self._profile_acc:  # equal-volume |q| shells to the detector edge
            edges = torch.linspace(0.0, 1.0, 13, dtype=torch.float64) ** (1 / 3)
            edges = (edges * self._q_max).to(self.device)
            edges[-1] = float("inf")
            zeros = torch.zeros(12, 9, dtype=torch.float64, device=self.device)
            self._profile_acc = [edges, zeros, zeros.clone(), zeros.clone()]
        acc = self._profile_acc
        clipped = torch.minimum(excess, 10.0 * var.clamp_min(1e-12).sqrt())
        total, counts, sq = ring_sums(
            clipped, positions, pixel_valid=valid, max_radius=8, squares=True
        )
        shell = torch.bucketize(pred.resolution.to(acc[0]), acc[0])
        k = shell.clamp(1, len(acc[0]) - 1) - 1
        for buf, v in zip(acc[1:], (total, counts, sq)):
            buf.index_add_(0, k.to(buf.device), v.to(buf))

    def start_profile_stacking(self) -> None:
        """Stack profiles around the predictions from now on."""
        self._profile_acc = []

    def finish_profile_stacking(self) -> Optional[list[float]]:
        """Set and return the per-shell disk radii (px)."""
        acc, self._profile_acc = self._profile_acc, None
        if not acc or float(acc[2][:, 0].sum()) <= 0:
            return None
        radii, captured = radius_by_shell(
            acc[1].cpu(), acc[2].cpu(), acc[3].cpu(), self._radii()[0]
        )
        self._radius_by_q = (acc[0].float(), radii)
        self._shell_capture = captured
        return radii

    def _aperture(self) -> str:
        return self._resolved_aperture or self.integrate.aperture

    def _radii(self) -> tuple[float, float, float]:
        return self._measured_radii or FALLBACK_RADII

    def _adu_per_photon(self) -> float:
        if self.integrate.adu_per_photon is not None:
            return float(self.integrate.adu_per_photon)
        if self._measured_gain is not None and self._measured_gain > 0:
            return float(self._measured_gain)
        if self._geometry_gain is not None and self._geometry_gain > 0:
            return float(self._geometry_gain)
        return 1.0

    def _integrate_result(
        self,
        result: IndexResult,
        excess: Tensor,
        var: Tensor,
        valid_mask: Optional[Tensor],
        mean: Optional[Tensor] = None,
        bright_threshold: float = 5.0,
        enrich_alpha: Optional[float] = None,
    ) -> None:
        # Predicts and integrates the full lattice of result.A into predicted_*
        frame_shape = (int(excess.shape[-2]), int(excess.shape[-1]))
        if self._q_max is None:
            self._q_max = detector_q_max(self.geometry, frame_shape)

        geometry = self._geometry_at(result.frame_index)
        wavelength = float(geometry["wavelength"])
        eta = math.radians(self.integrate.mosaicity_deg)
        r_size = self.integrate.domain_size_recip
        indexed = torch.where(result.indexed_mask)[0]
        if self.integrate.mosaicity_from_data and len(indexed):
            q_pred = result.hkl[indexed].to(excess.dtype) @ result.A.to(
                excess.dtype
            ).transpose(-1, -2)
            eta, r_size = estimate_mosaicity(
                q_pred,
                wavelength=wavelength,
                bandwidth=self.integrate.bandwidth,
                prior_eta=math.radians(self.integrate.mosaicity_deg),
                eta_min=math.radians(self.integrate.mosaicity_min_deg),
                eta_max=math.radians(self.integrate.mosaicity_max_deg),
                r_size_floor=self.integrate.domain_size_recip,
            )
        result.mosaicity = eta
        # The detector's |q| range scales as 1 / lambda at this frame's wavelength
        pred = predict_reflections(
            result.A.to(excess.dtype),
            geometry,
            q_max=self._q_max * float(self.geometry["wavelength"]) / wavelength,
            eta=eta,
            r_size=r_size,
            bandwidth=self.integrate.bandwidth,
            predict_sigma=self.integrate.predict_sigma,
            partiality_threshold=self.integrate.partiality_threshold,
            centering=self.target_cell.centering,
            frame_shape=frame_shape,
        )
        # every reflection is integrated where the lattice predicts it
        positions = pred.positions.to(excess.dtype)
        result.n_bright, result.enrichment, result.enrich_p = spot_enrichment(
            positions, excess, var, bright_threshold, pixel_valid=valid_mask
        )
        if enrich_alpha is not None and result.enrich_p > enrich_alpha:
            return
        if len(pred) == 0:
            result.predicted_hkl = pred.hkl
            result.predicted_positions = pred.positions
            for name in (
                "intensities",
                "sigmas",
                "peak",
                "background",
                "n_pixels",
                "bg_model",
                "bg_model_var",
                "n_pixels_total",
                "n_masked",
                "n_overlap",
            ):
                setattr(result, "predicted_" + name, excess.new_empty(0))
            return
        radii = self._reflection_radii(pred.resolution.to(excess.dtype))
        (
            positions,
            intensity,
            sigma,
            peak,
            background,
            n_pix,
            bg_model,
            bg_var,
            n_total,
            n_masked,
        ) = integrate_rings(
            positions,
            excess,
            var,
            mean=mean.to(excess.dtype) if mean is not None else None,
            pixel_valid=valid_mask,
            adu_per_photon=self._adu_per_photon(),
            n_bg=self._bg_annulus_pixels,
            radii=radii,
        )
        peak_res_nm = None
        if result.positions.numel() > 0:
            peak_q = detector_to_q(result.positions.to(excess.dtype), geometry)
            peak_res_nm = peak_q.norm(dim=-1) * A_INV_TO_NM_INV

        sig_ok = torch.isfinite(sigma) & (sigma > 0)
        measured = torch.where(sig_ok)[0]
        falloff = None
        if len(measured):
            q_nm = pred.resolution[measured].to(excess.dtype) * A_INV_TO_NM_INV
            falloff = falloff_resolution_limit(
                q_nm,
                intensity[measured] / sigma[measured],
                target=self.integrate.resolution_isigma,
                nbins=self.integrate.resolution_nbins,
                min_refl=self.integrate.resolution_min_refl,
            )
        result.falloff_limit = falloff
        drl = None
        if (
            peak_res_nm is not None
            and result.indexed_mask.shape == peak_res_nm.shape
            and len(indexed)
        ):
            # second-highest peak: one chance-indexed hot pixel cannot set the limit
            top = torch.sort(peak_res_nm[indexed.to(peak_res_nm.device)]).values
            drl = float(top[-min(2, len(top))])
        if drl is not None and falloff is not None:
            drl = max(drl, falloff)
        elif drl is None:
            drl = falloff
        if drl is None:
            if peak_res_nm is not None:
                peak_snr = None
                if result.intensities is not None and result.sigmas is not None:
                    peak_snr = result.intensities / result.sigmas
                drl = peak_resolution_limit(
                    peak_res_nm,
                    self.integrate.resolution_percentile,
                    snr=peak_snr,
                    snr_floor=self.integrate.resolution_snr_floor,
                )
            else:
                drl = float("inf")
        result.diffraction_limit = drl
        # a real lattice also carries signal where no found peak explains it
        explained = torch.zeros_like(sig_ok)
        if len(result.positions):
            near = torch.cdist(positions, result.positions.to(positions))
            explained = near.min(1).values < EXPLAINED_RADIUS_PX
        inner = sig_ok & ~explained
        inner &= pred.resolution.to(excess.dtype) * A_INV_TO_NM_INV <= drl
        result.enrich_p = max(
            result.enrich_p, unexplained_signal_p(intensity, sigma, inner)
        )
        if enrich_alpha is not None and result.enrich_p > enrich_alpha:
            return
        if self._profile_acc is not None:
            self._stack_profiles(excess, var, valid_mask, positions, pred)
        # Rocking width (A^-1) at the resolution limit, else at the median peak |q|
        if drl is not None and math.isfinite(drl) and drl > 0.0:
            q_rep = drl / A_INV_TO_NM_INV
        elif peak_res_nm is not None:
            q_rep = float(peak_res_nm.median()) / A_INV_TO_NM_INV
        else:
            q_rep = 0.0
        result.profile_radius = (
            r_size
            + 0.5 * eta * q_rep
            + 0.5 * wavelength * self.integrate.bandwidth * q_rep * q_rep
        )
        # a disk with no usable pixel (sigma 0) is kept, flagged, for the database
        stored = torch.isfinite(sigma)
        keep = torch.where(stored)[0]
        result.predicted_hkl = pred.hkl[keep]
        result.predicted_positions = positions[keep]
        result.predicted_intensities = intensity[keep]
        result.predicted_sigmas = sigma[keep]
        result.predicted_peak = peak[keep]
        result.predicted_background = background[keep]
        result.predicted_n_pixels = n_pix[keep]
        result.predicted_bg_model = bg_model[keep]
        result.predicted_bg_model_var = bg_var[keep]
        result.predicted_n_pixels_total = n_total[keep]
        result.predicted_n_masked = n_masked[keep]
        result._integration_positions = positions
        result._integration_radius = torch.as_tensor(
            radii[0], dtype=positions.dtype, device=positions.device
        ).expand(len(positions))
        result._integration_valid = stored
