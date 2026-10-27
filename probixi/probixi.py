from __future__ import annotations

import math
import random
import threading
import warnings
from collections.abc import Sized
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import chain, islice
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional, Union

import torch
from torch import Tensor

from .indexer import (
    CellMatchConfig,
    FrameIndexResult,
    FrameIndexStream,
    Indexer,
    IndexStats,
    IndexStream,
    IntegrateConfig,
    RefineConfig,
    SeedConfig,
)
from .indexer.forward import _lab_xy_pixels, _panel_bases, detector_to_q
from .indexer.indexer import MIN_PEAKS_TO_INDEX
from .indexer.integrate import (
    choose_aperture,
    nearest_neighbour_distance,
    radii_from_profile,
    ring_sums,
)
from .io import (
    CellParams,
    DataLoader,
    Metadata,
    build_physical_assembler,
    iter_frames,
    read_mask,
    render_frame,
)
from .io.cell import median_cell
from .io.geometry import EV_ANGSTROM
from .kernels import Engine, use_engine
from .peakfinding import PeakFinder, PeakStream
from .peakfinding.noise import (
    CalibrationResult,
    FrameScale,
    NoiseModel,
    ScaleReference,
    ThresholdCalibration,
    background_fluctuation_ratio,
    calibrate_noise,
    calibrate_threshold,
    fit_eigen_background,
    fit_photon_transfer,
)
from .peakfinding.noise.model import value_ok

PathLike = Union[str, Path]

# Beamstop-halo |q|_min search; see "Masks learned from the data" in peakfinding.md
_BEAMSTOP_MAX_FRAMES = 200
_BEAMSTOP_N_BINS = 40
_BEAMSTOP_SPIKE_RATIO = 50.0
_BEAMSTOP_EDGE_FRACTION = 0.1
_BEAMSTOP_MIN_FRACTION = 0.05
_BEAMSTOP_MIN_PEAKS = 200
_BEAMSTOP_MIN_BIN_PEAKS = 5
_BEAMSTOP_MIN_BIN_PIXELS = 10_000
_BEAMSTOP_MAX_EXTEND_BINS = 2

_SEED_LEVEL_OVERSAMPLE = 3  # candidates drawn per seed frame before the level cut
_SEED_LEVEL_STRIDE = 97  # a frame's level is the mean of every 97th valid pixel

_LAB_RADIUS_CHUNK_PIXELS = 1 << 18

_SHADOW_MAD_K = 6.0  # dark: log(pixel / radial mean) below median - 6 robust sigma
_SHADOW_MAX_RATIO = 0.5  # and pixel mean below half the radial mean
_SHADOW_CORE_FRACTION = 0.5  # core: dark pixel in a window at least half dark
_RADII_MIN_FRAMES = 32
_RADII_MAX_FRAMES = 512
_RADII_MIN_PEAKS = 200
_APERTURE_MIN_PEAKS = 4000  # --aperture auto tests strong, isolated peaks
_FLUX_AUTO_RATIO = 1.5  # flux variance turns on above this background fluctuation
_CLEN_MIN_SHIFT = 1e-3  # a camera-length re-fit must exceed 0.1% (relative)
_CLEN_MIN_Z = 5.0  # and five standard errors
_CLEN_MAX_RESIDUAL_PX = 6.0  # radial residual beyond which a peak is an outlier
_CLEN_MAX_ROUNDS = 3


# --- BEGIN GENERATED CITATION ---
_CITATION = r"""
@software{odea_probixi,
  author  = {O'Dea, Ryan and Kepa, Michal and Weinert, Tobias},
  title   = {{probixi}: Self-Calibrating Probabilistic Peak Finding for Serial X-Ray Crystallographic Data},
  version = {0.10.0},
  year    = {2026},
  url     = {https://github.com/ryan-odea/probixi}
}
"""
# --- END GENERATED CITATION ---
_CITATION_URL = (
    "https://raw.githubusercontent.com/ryan-odea/probixi/main/docs/assets/citation.bib"
)

# Bundled entry; citation() prefers the current one on GitHub
__citation__ = _CITATION.strip() + "\n"


@lru_cache(maxsize=1)
def mps_is_usable() -> bool:
    # is_available() is insufficient: the pipeline needs torch.linalg.inv on MPS
    try:
        torch.linalg.inv(torch.eye(3, device="mps"))
    except Exception:
        return False
    return True


def auto_device() -> torch.device:
    """Return the best available torch device.

    Returns
    -------
    torch.device
        ``cuda``; else ``mps`` when ``torch.linalg.inv`` runs on it; else ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps_is_usable():
        return torch.device("mps")
    return torch.device("cpu")


def citation(timeout: float = 3.0) -> str:
    """Print and return the BibTeX entry for probixi.

    Parameters
    ----------
    timeout : float, default 3.0
        Seconds to wait for the current entry on GitHub before using the bundled
        ``__citation__``.

    Returns
    -------
    str
        The BibTeX entry, newline-terminated.
    """
    # urlopen's timeout applies per address tried; the daemon thread caps the total
    fetched: list[str] = []

    def pull() -> None:
        try:
            from urllib.request import urlopen

            with urlopen(_CITATION_URL, timeout=timeout) as resp:
                fetched.append(resp.read().decode("utf-8"))
        except Exception:
            pass

    worker = threading.Thread(target=pull, daemon=True)
    worker.start()
    worker.join(timeout)
    text = fetched[0] if fetched and fetched[0].strip() else _CITATION
    entry = text.strip() + "\n"
    print(entry, end="")
    return entry


@dataclass
class CellCalibration:
    # One entry of Probixi.cell_calibrations
    n_lattices: int  # refined cells pooled
    cell: CellParams  # their component-wise median
    previous: CellParams
    edge_shift: float  # largest relative change over a, b, c
    angle_shift: float  # largest absolute change over alpha, beta, gamma (rad)
    applied: bool = True  # False: the median left the cell file's match window


@dataclass
class CameraLengthCalibration:
    # Probixi.camera_length_calibration, fitted on the seed lattices
    n_lattices: int
    previous: float  # m, before the first re-fit
    distance: float  # m
    eps: float  # distance / previous - 1
    sigma: float  # standard error of the last fit's relative shift
    applied: bool  # distance differs from previous


@dataclass
class Probixi:
    """Self-calibrating probabilistic peak finder and indexer.

    :meth:`calibrate` learns the background model, the peak contrast ``kappa``,
    the peak prior, and the detection threshold on seed frames;
    :meth:`index_stream` then detects, indexes, predicts, and integrates frames.

    Parameters
    ----------
    list_file, geometry_file, cell_file : path-like
        CrystFEL list, geometry, and cell files. ``cell_file`` is required for
        indexing and may be omitted for peakfinding only.
    mask_file : path-like, optional
        HDF5 file holding the bad-pixel mask, such as an Eiger master file;
        overrides the geometry's ``mask_file``. The dataset is the geometry's
        ``mask`` path when the file has it, else the NeXus ``pixel_mask`` under
        ``/entry/instrument/detector`` or its ``detectorSpecific`` group. An
        unreadable mask raises.
    noise_mode : {"online", "per_frame"}, default "online"
        ``online`` keeps updating the background after calibration;
        ``per_frame`` rebuilds it for each frame.
    warmup_frames : int, default 16
        Frames seen before zero-variance pixels are masked as dead.
    noise_decay : float, default 1.0
        Per-frame weight of the running statistics; 1 averages the run, ``d < 1``
        remembers about ``1/d`` frames.
    finder_kappa : float, default 10.0
        Peak contrast before calibration; :meth:`calibrate` replaces it.
    peak_size_max : int, default 30
        Floor of the component-size cap, in pixels; the cap grows with the
        matched-filter response.
    posterior_threshold : float, default 0.5
        Posterior above which a pixel is detected when ``matched_filter`` is off.
    candidate_threshold : float, optional
        Lower posterior threshold that also emits weak spots, in
        ``(0, posterior_threshold]``; used only when ``matched_filter`` is off.
    matched_filter : bool, default True
        Detect on the multi-scale matched filter; False thresholds the posterior.
    mf_scales : tuple of float, default (1.0, 1.6, 2.4)
        Gaussian template widths of the matched filter, in pixels.
    mf_threshold : float, default 5.0
        Matched-filter threshold until :meth:`calibrate` learns one.
    seed_level_frac : float, default 0.0
        Drop seed frames whose mean level is below this fraction of the seed
        sample's median; 0 disables.
    frame_screen_frac : float, default 0.1
        Withhold frames whose mean level is below this fraction of the run's
        from the running background and list them in ``screened_frames``;
        0 disables (``--force-all``).
    flux_variance : bool or None, default None
        Whiten each pixel against the variance a fitted photon-transfer curve
        predicts at the frame's own level. None decides at calibration: on when
        the background changes between shots beyond photon noise.
    flux_var_floor : float, default 0.15
        Lower bound on the flux-scaled variance, as a fraction of the running
        variance.
    random_seed : int, default 1988
        Seed for the draw of calibration and aperture-training frames.
    cell_calibrate : bool, default True
        Re-centre the target cell on the median refined cell of the calibration
        lattices and then during the stream, and re-fit the camera length.
    cell_calibrate_after : int, default 200
        Accepted lattices pooled before each re-centring during the stream.
    cell_calibrate_rounds : int, default 2
        Re-centrings during the stream.
    cell_calibrate_warn : float, default 0.005
        Relative edge shift above which a re-centring warns.
    cell_calibrate_alpha : float, default 1e-3
        Largest enrichment chance probability of a lattice whose cell is pooled.
    seed, refine, cell_match, integrate : optional
        Indexer configurations; None takes each class's defaults.
    device : torch.device, optional
        None selects CUDA, then MPS, then CPU.
    dtype : torch.dtype, default torch.float32
        Frame dtype.

    Attributes
    ----------
    threshold_calibration : ThresholdCalibration or None
        Matched-filter threshold fit, set by :meth:`calibrate`.
    screened_frames : list of int
        Absolute indices of frames withheld from the background as blank.
    cell_calibrations : list
        One record per target-cell re-centring.
    camera_length_calibration : optional
        Camera-length fit on the seed lattices, set by :meth:`calibrate`.
    """

    list_file: PathLike
    geometry_file: PathLike
    cell_file: Optional[PathLike] = None
    mask_file: Optional[PathLike] = None

    noise_mode: Literal["per_frame", "online"] = "online"
    warmup_frames: int = 16
    noise_decay: float = 1.0
    finder_kappa: float = 10.0
    peak_size_max: int = 30
    posterior_threshold: float = 0.5
    candidate_threshold: Optional[float] = None
    matched_filter: bool = True
    mf_scales: tuple[float, ...] = (1.0, 1.6, 2.4)
    mf_threshold: float = 5.0
    seed_level_frac: float = 0.0
    frame_screen_frac: float = 0.1
    flux_variance: Optional[bool] = None
    flux_var_floor: float = 0.15
    random_seed: int = 1988
    cell_calibrate: bool = True
    cell_calibrate_after: int = 200
    cell_calibrate_rounds: int = 2
    cell_calibrate_warn: float = 0.005
    cell_calibrate_alpha: float = 1e-3
    seed: Optional[SeedConfig] = None
    refine: Optional[RefineConfig] = None
    cell_match: Optional[CellMatchConfig] = None
    integrate: Optional[IntegrateConfig] = None
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    loader: DataLoader = field(init=False, repr=False)
    indexer: Optional[Indexer] = field(default=None, init=False, repr=False)
    threshold_calibration: Optional[ThresholdCalibration] = field(
        default=None, init=False, repr=False
    )
    _level_ref: Optional[float] = field(default=None, init=False, repr=False)
    _shadow_fraction: float = field(default=0.0, init=False, repr=False)
    screened_frames: list = field(default_factory=list, init=False, repr=False)
    cell_calibrations: list = field(default_factory=list, init=False, repr=False)
    camera_length_calibration: Optional[CameraLengthCalibration] = field(
        default=None, init=False, repr=False
    )
    predicted_radii: Optional[list] = field(default=None, init=False, repr=False)
    background_fluctuation: Optional[float] = field(
        default=None, init=False, repr=False
    )
    _cell_origin: Optional[CellParams] = field(default=None, init=False, repr=False)
    _noise: Optional[NoiseModel] = field(default=None, init=False, repr=False)
    _finder: Optional[PeakFinder] = field(default=None, init=False, repr=False)
    _scale_ref: Optional[ScaleReference] = field(default=None, init=False, repr=False)
    _frame_scales: dict = field(default_factory=dict, init=False, repr=False)
    _calibration_options: dict = field(default_factory=dict, init=False, repr=False)
    _calibration_boundary: int = field(default=0, init=False, repr=False)
    _h5_mask: Optional[Tensor] = field(default=None, init=False, repr=False)
    _h5_mask_loaded: bool = field(default=False, init=False, repr=False)
    _beamstop_qmin: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.device = (
            torch.device(self.device) if self.device is not None else auto_device()
        )
        self.loader = DataLoader(
            self.list_file, geometry_file=self.geometry_file, cell_file=self.cell_file
        )
        m = self.loader.metadata
        if m.geometry is None:
            raise ValueError(f"could not parse geometry from {self.geometry_file}")
        if self.cell_file is not None and m.cell is None:
            raise ValueError(f"could not parse cell from {self.cell_file}")
        if m.cell is not None:
            self.indexer = Indexer(
                m.geometry.to_dict(),
                m.cell,
                seed=self.seed,
                refine=self.refine,
                cell_match=self.cell_match,
                integrate=self.integrate,
                device=self.device,
            )
            self.indexer.frame_wavelengths = _frame_wavelengths(m)

    @property
    def integration_radii(self) -> Optional[tuple[float, float, float]]:
        """Learned ``(disk, annulus_in, annulus_out)`` radii in pixels, or None.

        :meth:`calibrate` measures them from the radial profile of peaks on
        frames outside the seed set. None before calibration or with too few
        peaks, in which case the indexer uses fixed radii.
        """
        if self.indexer is None:
            return None
        return self.indexer._measured_radii

    @property
    def aperture_choice(self) -> Optional[tuple[str, dict]]:
        """``(aperture, diagnostics)`` chosen under ``aperture="auto"``, or None.

        Set by :meth:`calibrate`. ``diagnostics`` holds the strong-peak count
        ``n``, the captured flux fraction ``f_mean`` and its ``common_trend`` over
        ``|q|``, the per-crystal ``residual_trend`` and ``crystal_spread`` with
        ``spread_z``, the ``noise_ratio`` of flux to snr per-observation variance
        when the spread is significant (flux wins only below 1), both radii, and
        the ``reason``. An early exit, on too few peaks or crystals, carries only
        the keys measured before it.
        """
        diag = getattr(self, "_aperture_diag", None)
        if diag is None or self.indexer is None:
            return None
        return self.indexer._aperture(), diag

    @property
    def integration_recipe(self) -> Optional[dict]:
        """Integration settings written to the database's ``integration`` table.

        ``radii`` (disk, annulus_in, annulus_out) in pixels, ``adu_per_photon``,
        the local-background ``bg_annulus_pixels``, and the ``aperture`` rule, as
        applied after :meth:`calibrate`. None without a cell.
        """
        if self.indexer is None:
            return None
        return dict(
            radii=tuple(float(r) for r in self.indexer._radii()),
            adu_per_photon=self.indexer._adu_per_photon(),
            bg_annulus_pixels=self.indexer._bg_annulus_pixels,
            aperture=self.indexer._aperture(),
        )

    @property
    def calibration_summary(self) -> Optional[dict]:
        """Calibration provenance for the database's ``calibration`` table.

        The noise model (gain, read variance, flux model), the detection
        threshold with its quiet-frame diagnostics, the peak mixture, the
        integration aperture and the cell and camera length in use. None before
        :meth:`calibrate`.
        """
        cal = getattr(self, "calibration", None)
        if cal is None:
            return None
        tc = getattr(self, "threshold_calibration", None)
        ap = self.aperture_choice
        idx = self.indexer
        cell = idx.target_cell if idx is not None else None
        return dict(
            n_seed_frames=self.n_seed_frames,
            gain=self.noise.gain,
            read_var=self.noise.read_var,
            flux_variance=bool(self.flux_variance),
            background_fluctuation=getattr(self, "background_fluctuation", None),
            mf_threshold=None if tc is None else tc.threshold,
            target_noise_peaks=None if tc is None else tc.target_noise_peaks,
            tail_ratio=None if tc is None else tc.tail_ratio,
            n_quiet_frames=None if tc is None else tc.n_quiet_frames,
            kappa=cal.kappa,
            prior_peak=cal.prior_peak,
            var_scale=cal.var_scale,
            aperture=None if ap is None else ap[0],
            clen=None if idx is None else idx.geometry.get("clen"),
            cell=(
                None
                if cell is None
                else (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
            ),
        )

    @property
    def shadow_fraction(self) -> float:
        """Fraction of live pixels masked as static shadow by :meth:`calibrate`."""
        return float(self._shadow_fraction)

    @property
    def blank_frame_floor(self) -> Optional[float]:
        """Mean level below which a frame is a blank shot, or None.

        None when the screen is disabled or before :meth:`calibrate` measures the
        run's reference level.
        """
        if self.frame_screen_frac <= 0.0 or not self._level_ref:
            return None
        return self.frame_screen_frac * float(self._level_ref)

    def frame_source(self, index: int) -> str:
        """CrystFEL ``filename //event`` label of an absolute frame index."""
        offset = 0
        for info in (self.metadata.files or {}).values():
            n = int(getattr(info, "n_frames", 0))
            if offset <= index < offset + n:
                ev = index - offset + int(getattr(info, "event_start", 0))
                if getattr(info, "eventless", False):
                    return f"{info.filename} //"
                if getattr(info, "events", None) is not None:
                    return f"{info.filename} {info.events[ev]}//"
                return f"{info.filename} //{ev}"
            offset += n
        return f"frame {index}"

    @property
    def metadata(self) -> Metadata:
        """Parsed run metadata (frames, geometry, cell)."""
        return self.loader.metadata

    @property
    def geometry(self) -> dict:
        """Detector geometry dict, including any camera-length re-fit."""
        if self.indexer is not None:
            return self.indexer.geometry
        geom = self.loader.metadata.geometry
        assert geom is not None  # __post_init__ raises otherwise
        return geom.to_dict()

    @property
    def target_cell(self) -> CellParams:
        """Unit cell accepted lattices must match, after any re-centring."""
        if self.indexer is None:
            raise RuntimeError(
                "no target cell; construct Probixi with a cell_file to index"
            )
        return self.indexer.target_cell

    @property
    def noise(self) -> NoiseModel:
        """Live noise model, built on the first frame seen."""
        if self._noise is None:
            raise RuntimeError(
                "Probixi not yet seeded; pass a frame through fit_noise, calibrate, "
                "or a stream method first to infer the frame size."
            )
        return self._noise

    @property
    def finder(self) -> PeakFinder:
        """Peak finder over the noise model, built on the first frame seen."""
        if self._finder is None:
            raise RuntimeError(
                "Probixi not yet seeded; pass a frame through fit_noise, calibrate, "
                "or a stream method first to infer the frame size."
            )
        return self._finder

    def frames(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        batch_size: int = 1,
        prefetch: int = 2,
    ) -> Iterator[Tensor]:
        """Stream frames off disk as device tensors, prefetching reads.

        Parameters
        ----------
        start, stop : int, optional
            Absolute frame range ``[start, stop)``; the whole run by default.
        batch_size : int, default 1
            Frames per yielded tensor.
        prefetch : int, default 2
            Batches a background thread reads ahead.

        Returns
        -------
        Iterator[torch.Tensor]
            ``(H, W)`` frames, or ``(B, H, W)`` batches when ``batch_size > 1``,
            on the configured device and dtype.
        """
        return iter_frames(
            self.loader,
            start=start,
            stop=stop,
            device=self.device,
            dtype=self.dtype,
            batch_size=batch_size,
            prefetch=prefetch,
        )

    def _sample_frame_indices(self, k: int, exclude: Iterable[int] = ()) -> list[int]:
        skip = set(exclude)
        pool = [i for i in range(int(self.metadata.n_frames)) if i not in skip]
        return sorted(random.Random(self.random_seed).sample(pool, min(k, len(pool))))

    def _frames_at(self, indices: Iterable[int]) -> Iterator[Tensor]:
        # Consecutive indices are read as one contiguous range.
        idx = list(indices)
        i = 0
        while i < len(idx):
            j = i + 1
            while j < len(idx) and idx[j] == idx[j - 1] + 1:
                j += 1
            yield from self.frames(start=idx[i], stop=idx[j - 1] + 1)
            i = j

    def _resolve_frame_index(self, frame: Union[int, str, tuple]) -> int:
        # int -> absolute index; "file//event" or (file, event) -> cumulative index
        if isinstance(frame, int):
            return frame
        if isinstance(frame, str):
            name, _, ev = frame.partition("//")
            filename, event = name.strip(), ev
            if " " in filename and not event:  # CrystFEL's named "file tag-N//"
                filename, event = filename.rsplit(" ", 1)
        else:
            filename, event = frame
        # CrystFEL event strings carry a "//" prefix (e.g. "//136")
        event = str(event).strip().lstrip("/") or "0"
        number = int(event) if event.isdigit() else -1
        base = Path(filename).name
        offset = 0
        for info in self.metadata.files.values():
            if str(info.filename) == filename or Path(info.filename).name == base:
                names = getattr(info, "events", None) or []
                ev = names.index(event) if event in names else number
                if info.event_start <= ev < info.event_start + info.n_frames:
                    return offset + ev - info.event_start
            offset += int(info.n_frames)
        raise KeyError(f"frame source not found: {filename}")

    def show_frame(
        self,
        frame: Union[int, str, tuple],
        *,
        path: PathLike,
        peaks: bool = True,
        predictions: bool = True,
        update_noise: bool = False,
        **render_kwargs,
    ) -> Path:
        """Render one frame with its detected peaks and predicted reflections.

        Runs peakfinding, and indexing when a cell is set, on that frame alone.
        Tiled detectors are drawn assembled. Call :meth:`calibrate` first.

        Parameters
        ----------
        frame : int, str, or tuple
            Absolute frame index, ``"filename//event"``, ``"filename tag-N//"``,
            or a ``(filename, event)`` pair.
        path : str or Path
            Output image path.
        peaks : bool, default True
            Overlay detected peaks as open circles.
        predictions : bool, default True
            Overlay predicted reflections as crosses when the frame indexes.
        update_noise : bool, default False
            Fold this frame into the live noise model.
        **render_kwargs
            Forwarded to :func:`probixi.io.render_frame` (``title``, ``cmap``,
            ``vmax_pct``, ...).

        Returns
        -------
        Path
            The written image path.
        """
        idx = self._resolve_frame_index(frame)
        image = next(iter(self.frames(start=idx, stop=idx + 1)))
        pk = None
        refl = None
        if predictions and self.indexer is not None:
            indexed = self.index_stream(
                [image], start_index=idx, update_noise=update_noise
            ).collect()
            if indexed:
                r = indexed[0]
                pk = r.positions if peaks else None
                refl = r.predicted_positions
        if peaks and pk is None and refl is None:
            for res in self.peak_stream(
                [image],
                start_index=idx,
                update_noise=update_noise,
                estimate_scale=False,
            ):
                ks = res.kept_stats
                pk = torch.stack([ks.row_centroid, ks.col_centroid], dim=-1)
                break
        mask = self._noise.valid_mask if self._noise is not None else None
        asm = build_physical_assembler(self.loader.metadata.geometry)
        if asm is not None:
            assemble, _, _, place = asm
            if mask is not None:
                image = image.masked_fill(~mask.to(image.device), float("nan"))
            image, mask = assemble(image.detach().cpu().numpy()), None
            pk = None if pk is None else place(pk.detach().cpu().numpy())
            refl = None if refl is None else place(refl.detach().cpu().numpy())
        render_kwargs.setdefault("title", f"frame {idx}")
        return render_frame(
            image, path=path, peaks=pk, reflections=refl, mask=mask, **render_kwargs
        )

    def _ensure_built(self, item: Tensor) -> None:
        if self._noise is not None:
            return
        frame_size = (int(item.shape[-2]), int(item.shape[-1]))
        first = item[0] if item.ndim == 3 else item
        geom = self.loader.metadata.geometry
        radius = _lab_radius_map(geom, frame_size, device=self.device)
        if radius is None and len(getattr(geom, "panels", None) or {}) > 1:
            warnings.warn(
                "geometry: could not build a lab-frame radius map (panels are "
                "missing fs/ss or corner_x/corner_y); the radial background "
                "model will bin on array radius, which is wrong for a tiled "
                "detector",
                stacklevel=2,
            )
        panels = _panel_boxes(geom, frame_size)
        valid = self._static_mask(first, frame_size)
        if panels is not None:
            covered = torch.zeros_like(valid)
            for r0, r1, c0, c1 in panels:
                covered[r0:r1, c0:c1] = True
            valid &= covered
        self._noise = NoiseModel(
            frame_size=frame_size,
            mode=self.noise_mode,
            decay=self.noise_decay,
            warmup_frames=self.warmup_frames,
            beam_center=geom.beam_center if geom is not None else None,
            radial_radius=radius,
            panels=panels,
            valid_mask=valid,
            value_flags=self._value_flags(),
            device=self.device,
            dtype=self.dtype,
        )
        self._finder = PeakFinder(
            self._noise,
            kappa=self.finder_kappa,
            size_max=self.peak_size_max,
            posterior_threshold=self.posterior_threshold,
            candidate_threshold=self.candidate_threshold,
            matched_filter=self.matched_filter,
            mf_scales=self.mf_scales,
            mf_threshold=self.mf_threshold,
            flux_variance=self.flux_variance,
            flux_var_floor=self.flux_var_floor,
        )

    def _static_mask(self, frame: Tensor, frame_size: tuple[int, int]) -> Tensor:
        # A priori bad: >= max_adu, flag_* values, panel edges, bad regions, stored mask
        mask = torch.ones(frame_size, dtype=torch.bool, device=frame.device)
        geom = self.loader.metadata.geometry
        max_adu = geom.parameters.get("max_adu") if geom else None
        if isinstance(max_adu, (int, float)):
            mask &= frame < float(max_adu)
        mask &= value_ok(frame, self._value_flags())
        if geom is not None:
            _mask_panel_edges(mask, geom)
        for br in geom.bad_regions if geom else []:
            if br.is_lab_frame:
                continue
            r0, r1 = max(0, br.min_ss), min(frame_size[0] - 1, br.max_ss)
            c0, c1 = max(0, br.min_fs), min(frame_size[1] - 1, br.max_fs)
            if r0 <= r1 and c0 <= c1:
                mask[r0 : r1 + 1, c0 : c1 + 1] = False
        if geom is not None:
            _mask_lab_bad_regions(mask, geom)
        h5_mask = self._hdf5_valid_mask(frame_size)
        if h5_mask is not None:
            mask &= h5_mask.to(device=mask.device)
        return mask.to(self.device)

    def _value_flags(self) -> list[tuple]:
        # geometry value flags as (rows, cols, key, value), re-tested on every frame
        geom = self.loader.metadata.geometry
        flags = []
        for name, entries in (geom.panel_flags or {}).items() if geom else []:
            panel = geom.panels.get(name)
            if panel is None:
                continue
            rows = slice(max(0, int(panel["min_ss"])), int(panel["max_ss"]) + 1)
            cols = slice(max(0, int(panel["min_fs"])), int(panel["max_fs"]) + 1)
            flags += [(rows, cols, key, value) for key, value in entries]
        return flags

    def _hdf5_valid_mask(self, frame_size: tuple[int, int]) -> Optional[Tensor]:
        if self._h5_mask_loaded:
            return self._h5_mask
        self._h5_mask_loaded = True
        geom = self.loader.metadata.geometry
        files = self.loader.metadata.files
        if geom is None or not files:
            return None
        data_file = next(iter(files.values())).filename
        override = str(self.mask_file) if self.mask_file is not None else None
        good = read_mask(geom, data_file, frame_size, mask_file=override)
        if good is not None and tuple(good.shape) == tuple(frame_size):
            self._h5_mask = torch.as_tensor(good, dtype=torch.bool)
        return self._h5_mask

    def _qmap(self, frame_size: tuple[int, int], device=None) -> Optional[Tensor]:
        # Per-pixel |q| (A^-1) from the detector geometry
        geom = self.loader.metadata.geometry
        if geom is None:
            return None
        rr = torch.arange(frame_size[0], dtype=torch.float32, device=device).view(-1, 1)
        cc = torch.arange(frame_size[1], dtype=torch.float32, device=device).view(1, -1)
        pos = torch.stack(
            [rr.expand(frame_size).reshape(-1), cc.expand(frame_size).reshape(-1)],
            dim=-1,
        )
        return torch.linalg.vector_norm(
            detector_to_q(pos, geom.to_dict(), dtype=torch.float32), dim=-1
        ).reshape(frame_size)

    def _infer_beamstop_qmin(self, seed: Iterable[Tensor]) -> Optional[float]:
        # Beamstop-halo |q|_min (A^-1) from the |q| density of the seed peaks
        if self._noise is None:
            return None
        shp = self._noise.valid_mask.shape
        frame_size = (int(shp[0]), int(shp[1]))
        qmap = self._qmap(frame_size, device=self._noise.valid_mask.device)
        if qmap is None:
            return None
        rows, cols = frame_size
        n_bins = _BEAMSTOP_N_BINS
        qs: list[Tensor] = []
        for res in self.peak_stream(
            islice(seed, _BEAMSTOP_MAX_FRAMES), update_noise=False, estimate_scale=False
        ):
            ks = res.kept_stats
            if ks is None or ks.row_centroid.numel() == 0:
                continue
            r = ks.row_centroid.round().long().clamp(0, rows - 1)
            c = ks.col_centroid.round().long().clamp(0, cols - 1)
            qs.append(qmap[r, c])
        if not qs:
            return None
        allq = torch.cat(qs).float()
        if allq.numel() < _BEAMSTOP_MIN_PEAKS:
            return None
        qmax = float(qmap.max())
        if qmax <= 0:
            return None
        edges = torch.linspace(0.0, qmax, n_bins + 1)
        ph = torch.histc(allq, bins=n_bins, min=0.0, max=qmax)
        pix = torch.histc(
            qmap[self._noise.valid_mask].float(), bins=n_bins, min=0.0, max=qmax
        )
        q_min = _beamstop_qmin_from_histogram(ph, pix, edges, n_bins)
        if q_min is None:
            return None
        if q_min <= 0.0:
            return None
        if float((allq < q_min).sum()) / allq.numel() < _BEAMSTOP_MIN_FRACTION:
            return None
        return q_min

    def _learn_integration_radii(
        self, exclude: Iterable[int]
    ) -> Optional[tuple[float, float, float]]:
        profile = None
        n_peaks = 0
        n_frames = 0
        auto = self.indexer.integrate.aperture == "auto"
        sums, sqs, cnts, qs, nns, frames, qmap = [], [], [], [], [], [], None
        pool = self._sample_frame_indices(_RADII_MAX_FRAMES, exclude)
        if len(pool) < _RADII_MIN_FRAMES:  # a short run: reuse the seed frames
            pool = self._sample_frame_indices(_RADII_MAX_FRAMES)
        for res in self.peak_stream(
            self._frames_at(pool),
            update_noise=False,
            estimate_scale=False,
        ):
            n_frames += 1
            if n_frames > _RADII_MIN_FRAMES and n_peaks >= (
                _APERTURE_MIN_PEAKS if auto else _RADII_MIN_PEAKS
            ):
                break
            excess = res.scores.get("excess") if res.scores else None
            ks = res.kept_stats
            if excess is None or ks is None or ks.row_centroid.numel() == 0:
                continue
            positions = torch.stack([ks.row_centroid, ks.col_centroid], dim=-1).to(
                excess
            )
            total, counts, sq = ring_sums(
                excess, positions, pixel_valid=res.valid_mask, squares=auto
            )
            one = (total / counts.clamp_min(1.0)).median(dim=0).values
            profile = one if profile is None else profile + one
            n_peaks += int(positions.shape[0])
            if auto and sq is not None:
                if qmap is None:
                    qmap = self._qmap(tuple(excess.shape), device=excess.device)
                if qmap is None:
                    auto = False
                    continue
                c = positions.round().long()
                qs.append(
                    qmap[
                        c[:, 0].clamp(0, excess.shape[0] - 1),
                        c[:, 1].clamp(0, excess.shape[1] - 1),
                    ]
                )
                sums.append(total)
                sqs.append(sq)
                cnts.append(counts)
                nns.append(nearest_neighbour_distance(positions))
                frames.append(
                    torch.full((len(positions),), n_frames, device=positions.device)
                )
        if profile is None or n_peaks < _RADII_MIN_PEAKS:
            return None
        aperture = self.indexer.integrate.aperture
        if auto and sums:
            flux = radii_from_profile(profile)
            snr = radii_from_profile(profile, snr=True)
            if flux is None or snr is None:
                return None
            aperture, diag = choose_aperture(
                torch.cat(sums),
                torch.cat(sqs),
                torch.cat(cnts),
                torch.cat(qs),
                torch.cat(frames),
                snr,
                flux,
                adu_per_photon=self.indexer._adu_per_photon(),
                tolerance=self.indexer.integrate.aperture_tolerance,
                nn_dist=torch.cat(nns),
            )
            self.indexer._resolved_aperture = aperture
            self._aperture_diag = diag
            # inputs to the aperture choice, kept for offline inspection
            self._aperture_data = dict(
                sums=torch.cat(sums).cpu(),
                squares=torch.cat(sqs).cpu(),
                counts=torch.cat(cnts).cpu(),
                q=torch.cat(qs).cpu(),
                nn_dist=torch.cat(nns).cpu(),
                frame=torch.cat(frames).cpu(),
                radii_snr=snr,
                radii_flux=flux,
                adu_per_photon=self.indexer._adu_per_photon(),
            )
            return snr if aperture == "snr" else flux
        return radii_from_profile(profile, snr=aperture != "flux")

    def _apply_beamstop_qmin(self, q_min: float) -> None:
        # Exclude |q| < q_min from detection and from every background source
        if self._noise is None or q_min <= 0:
            return
        shp = self._noise.valid_mask.shape
        qmap = self._qmap(
            (int(shp[0]), int(shp[1])), device=self._noise.valid_mask.device
        )
        if qmap is None:
            return
        self._apply_static_exclusion(qmap >= q_min)

    def _apply_static_exclusion(self, keep: Tensor) -> None:
        assert self._noise is not None
        keep = keep.to(self._noise.valid_mask.device)
        self._noise.static_mask &= keep
        self._noise.valid_mask &= keep
        self._noise._mask_version += 1  # PeakFinder recomputes mask denominators

    def _infer_shadow_mask(self) -> Optional[Tensor]:
        # Grown by the annulus radius so no local background straddles a shadow edge
        if self._noise is None:
            return None
        finder = self.finder
        grow = (
            finder.local_outer_radius
            if finder.local_background
            else int(math.ceil(3.0 * max(finder.mf_scales)))
        )
        shadow = shadow_mask(
            self._noise.pixel.mean(),
            self._noise.rotational.mean(),
            self._noise.valid_mask,
            grow,
        )
        live = float(self._noise.valid_mask.sum())
        self._shadow_fraction = (
            float(shadow.sum()) / live if shadow is not None and live > 0 else 0.0
        )
        return shadow

    def _frame_level(self, item: Tensor) -> float:
        # A mean stays continuous on integer counts, where a median sits on 0 or 1.
        self._ensure_built(item)
        assert self._noise is not None
        mask = self._noise.valid_mask
        keep = mask.reshape(-1)[::_SEED_LEVEL_STRIDE]
        v = item.reshape(-1, mask.numel())[:, ::_SEED_LEVEL_STRIDE]
        v = v.to(device=keep.device, dtype=torch.float32)[:, keep]
        return float(v.mean()) if v.numel() else 0.0

    def _update_noise(self, item: Tensor) -> None:
        if item.ndim == 3:
            for frame in item:
                self.noise.update(frame)
        else:
            self.noise.update(item)

    def fit_noise(self, frames: Iterable[Tensor]) -> "Probixi":
        """Fold frames into the noise model, building it on the first.

        Parameters
        ----------
        frames : iterable of torch.Tensor
            ``(H, W)`` frames or ``(B, H, W)`` batches.

        Returns
        -------
        Probixi
            Self.
        """
        for item in frames:
            self._ensure_built(item)
            self._update_noise(item)
        return self

    def noise_diagnostics(
        self,
        path: PathLike,
        *,
        frames: Optional[Iterable[Tensor]] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        batch_size: int = 15,
        **kwargs,
    ) -> Path:
        """Write a GIF of the live noise model over a slice of the run.

        Animates the running mean, its radial profile, and the per-batch drift.
        Every frame is folded into the live model, so call it before
        :meth:`calibrate` or on a fresh pipeline. One mean image is kept per
        batch in memory.

        Parameters
        ----------
        path : path-like
            Output ``.gif`` path.
        frames : iterable of torch.Tensor, optional
            Frames to animate; the ``[start, stop)`` slice of the run by default.
        start, stop : int, optional
            Frame range when ``frames`` is not given.
        batch_size : int, default 15
            Frames folded in between snapshots.
        **kwargs
            Forwarded to the noise model's diagnostics (``fps``, ``cmap``,
            ``dpi``, ``max_radius``, ``figsize``).

        Returns
        -------
        Path
            The written GIF path.
        """
        src = self.frames(start=start, stop=stop) if frames is None else frames
        it = iter(src)
        try:
            first = next(it)
        except StopIteration as exc:
            raise ValueError("no frames available to animate") from exc
        self._ensure_built(first)
        # Tiled detectors are drawn assembled so rings stay continuous
        asm = build_physical_assembler(self.loader.metadata.geometry)
        if asm is not None:
            assemble_fn, beam_yx, *_ = asm
            kwargs.setdefault("assemble", assemble_fn)
            kwargs.setdefault("assembled_beam", beam_yx)
        return self.noise.diagnostics(
            chain([first], it), path, batch_size=batch_size, **kwargs
        )

    @use_engine(Engine.REFERENCE)
    def calibrate(
        self,
        n_seed: int = 32,
        seed_frames: Optional[Iterable[Tensor]] = None,
        eigen_modes: int = 0,
        target_noise_peaks: Union[float, str, None] = "auto",
        threshold_opts: Optional[dict] = None,
        **opts,
    ) -> CalibrationResult:
        """Calibrate the detector on seed frames and freeze the learned parameters.

        Fits the background blend and variance scale that make the background
        ``z`` standard normal, the peak contrast ``kappa`` and peak prior by EM,
        and, with ``target_noise_peaks`` set, the matched-filter threshold. Also
        decides ``flux_variance`` when it is None and learns the shadow and
        beamstop masks and the integration aperture. With a cell it indexes the
        seed frames to size the disks at predicted positions and, under
        ``cell_calibrate``, to re-centre the cell and re-fit the camera length.

        Parameters
        ----------
        n_seed : int, default 32
            Frames drawn at random from the run when ``seed_frames`` is not given.
        seed_frames : iterable of torch.Tensor, optional
            Calibration frames replacing the random draw; a generator is copied
            into a list first.
        eigen_modes : int, default 0
            Low-rank background modes to fit; 0 fits none.
        target_noise_peaks : float, "auto" or None, default "auto"
            Noise budget: expected noise detections per signal-free frame;
            ``"auto"`` is 1.25 per million valid pixels. None keeps
            ``mf_threshold``.
        threshold_opts : dict, optional
            Keyword arguments for :func:`probixi.peakfinding.calibrate_threshold`.
        **opts
            Keyword arguments for :func:`probixi.peakfinding.calibrate_noise`.

        Returns
        -------
        CalibrationResult
            The applied noise calibration (``kappa``, ``prior_peak``,
            ``var_scale``, ...).
        """
        seed_indices: list[int] = []
        self.blank_seed_floor = None
        self.n_blank_seeds_dropped = 0
        seed: Union[Sized, Iterable[Tensor]]
        if seed_frames is not None:
            seed = seed_frames if isinstance(seed_frames, Sized) else list(seed_frames)
        elif self.seed_level_frac > 0.0:
            cand = self._sample_frame_indices(n_seed * _SEED_LEVEL_OVERSAMPLE)
            levels = [self._frame_level(f) for f in self._frames_at(cand)]
            floor = _blank_seed_floor(levels, self.seed_level_frac)
            keep = [i for i, lv in zip(cand, levels) if lv >= floor]
            self.blank_seed_floor = floor
            self.n_blank_seeds_dropped = len(cand) - len(keep)
            if len(keep) < max(4, n_seed // 4):
                warnings.warn(
                    f"calibration: only {len(keep)} of {len(cand)} seed "
                    f"candidates clear the blank-shot floor {floor:.3g}; "
                    f"ignoring the level test for this run",
                    stacklevel=2,
                )
                keep = cand
                self.n_blank_seeds_dropped = 0
            seed_indices = keep[:n_seed]
            seed = _FrameSource(self, seed_indices)
        else:
            seed_indices = self._sample_frame_indices(n_seed)
            seed = _FrameSource(self, seed_indices)
        if not seed:
            raise ValueError("no seed frames available to calibrate on")
        self.n_seed_frames = len(seed)
        if seed_frames is None and len(seed) < n_seed:
            warnings.warn(
                f"calibrating on all {len(seed)} frames of the run, fewer than the "
                f"{n_seed} seed frames asked for",
                stacklevel=2,
            )
        self._calibration_options = dict(
            n_seed=n_seed,
            eigen_modes=eigen_modes,
            target_noise_peaks=target_noise_peaks,
            threshold_opts=threshold_opts,
            **opts,
        )
        if self._level_ref is None and seed:
            lv = sorted(self._frame_level(f) for f in seed)
            self._level_ref = lv[len(lv) // 2]
        self.fit_noise(seed)
        shadow = self._infer_shadow_mask()
        if shadow is not None:
            # refit every background source with the shadow out of the masks
            self._apply_static_exclusion(~shadow)
            self.noise.reset()
            self.fit_noise(seed)
        result = calibrate_noise(self.noise, seed, warm=False, **opts)
        result.apply(self.noise, self.finder)
        self.calibration = result
        if self.flux_variance is None:
            self.background_fluctuation = background_fluctuation_ratio(
                self.noise,
                seed,
                self.finder.local_inner_radius,
                self.finder.local_outer_radius,
            )
            self.flux_variance = self.background_fluctuation > _FLUX_AUTO_RATIO
            self.finder.flux_variance = self.flux_variance
        if self.flux_variance:
            fit_photon_transfer(self.noise)
        if eigen_modes > 0:
            fit_eigen_background(self.noise, seed, n_modes=eigen_modes)
        if target_noise_peaks is not None:
            topts = dict(threshold_opts or {})
            topts.setdefault("flux_variance", self.flux_variance)
            topts.setdefault("flux_var_floor", self.flux_var_floor)
            self.threshold_calibration = calibrate_threshold(
                self.noise,
                seed,
                target_noise_peaks=(
                    None if target_noise_peaks == "auto" else float(target_noise_peaks)
                ),
                finder=self.finder,
                **topts,
            )
        self._scale_ref = ScaleReference.from_noise_model(self.noise)
        if self.indexer is not None and self.noise.gain is not None:
            self.indexer._measured_gain = self.noise.gain
        if self.indexer is not None:
            self.indexer._bg_annulus_pixels = self.finder.background_annulus_pixels()
        if self.indexer is not None:
            self.indexer._measured_radii = self._learn_integration_radii(seed_indices)
        self._beamstop_qmin = self._infer_beamstop_qmin(seed)
        if self._beamstop_qmin:
            self._apply_beamstop_qmin(self._beamstop_qmin)
        self._sync_active_noise_sources()
        if self.indexer is not None and not self.cell_calibrations:
            self._calibrate_cell(seed)
        return result

    def _calibrate_cell(self, frames: Iterable[Tensor]) -> None:
        tc = self.threshold_calibration

        def index() -> list:
            return list(
                self.indexer.index_frame_stream(
                    self.peak_stream(frames, update_noise=False),
                    bright_threshold=(
                        tc.threshold if tc is not None else self.mf_threshold
                    ),
                    enrich_alpha=self.cell_calibrate_alpha,
                )
            )

        self.indexer.start_profile_stacking()
        results = index()
        for _ in range(_CLEN_MAX_ROUNDS if self.cell_calibrate else 0):
            if not self._recalibrate_camera_length(
                [c for r in results for c in r.crystals]
            ):
                break
            # re-index at the new distance
            self.indexer.start_profile_stacking()
            results = index()
        cl = self.camera_length_calibration
        if cl is not None and cl.applied:
            warnings.warn(
                f"camera length re-fitted on {cl.n_lattices} seed lattices: "
                f"{1e3 * cl.previous:.3f} -> {1e3 * cl.distance:.3f} mm "
                f"({100 * cl.eps:+.2f}%); the geometry's clen does not match the "
                "radial curvature of the indexed peaks",
                stacklevel=2,
            )
        self.predicted_radii = self.indexer.finish_profile_stacking()
        cells = [c.cell for r in results for c in r.crystals]
        self._frame_scales.clear()
        if self.cell_calibrate and len(cells) >= max(
            1, self.cell_calibrate_after // 10
        ):
            self._recalibrate_cell(cells)

    def _recalibrate_camera_length(self, crystals: list) -> bool:
        assert self.indexer is not None
        if len(crystals) < 10:
            return False
        fit = camera_length_error(self.indexer, crystals)
        if fit is None:
            return False
        eps, se, n = fit
        old = float(self.indexer.geometry["clen"])
        step = abs(eps) >= max(_CLEN_MIN_SHIFT, _CLEN_MIN_Z * se)
        new = old * (1.0 + eps) if step else old
        prev = self.camera_length_calibration
        first = prev.previous if prev is not None else old
        self.camera_length_calibration = CameraLengthCalibration(
            n, first, new, new / first - 1.0, se, new != first
        )
        if step:
            self.indexer.geometry["clen"] = new
            self.indexer.geometry["clen_shift"] = new - first  # writers move file clen
            self.loader.metadata.geometry.distance = new
            self.indexer._q_max = None
        return step

    @property
    def beamstop_min_res(self) -> Optional[float]:
        """Low-resolution limit (A) of the learned beamstop exclusion, or None."""
        return 1.0 / self._beamstop_qmin if self._beamstop_qmin else None

    def _sync_active_noise_sources(self) -> None:
        finder = getattr(self, "_finder", None)
        if finder is None:
            return
        if finder.combination == "calibrated":
            weights = self.noise.calibrated_weights or {"pixel": 1.0}
        else:
            weights, _ = self.noise._preset_weights(finder.combination)
        self.noise.set_active_update_sources({k for k, w in weights.items() if w > 0})

    def peak_stream(
        self,
        frames: Iterable[Tensor],
        start_index: int = 0,
        update_noise: bool = True,
        estimate_scale: bool = True,
    ) -> PeakStream:
        """Open a lazy stream of per-frame peak results.

        Parameters
        ----------
        frames : iterable of torch.Tensor
            ``(H, W)`` frames or ``(B, H, W)`` batches; the first builds the
            noise model if needed.
        start_index : int, default 0
            Absolute index of the first frame.
        update_noise : bool, default True
            Fold each non-blank frame into the running noise model.
        estimate_scale : bool, default True
            Estimate each frame's fluence scale for :meth:`index_stream`; False
            skips the regression, as peakfinding alone never reads it.

        Returns
        -------
        PeakStream
            Lazy stream of ``PeakResult``; tensors stay on the device.
        """

        floor = self.blank_frame_floor

        def _tee() -> Iterator[Tensor]:
            offset = 0
            for item in frames:
                self._ensure_built(item)
                self.noise.record_drift = False
                blank = (
                    update_noise
                    and floor is not None
                    and self._frame_level(item) < floor
                )
                if blank:
                    self.screened_frames.append(start_index + offset)
                if update_noise and not blank:
                    self._update_noise(item)
                if estimate_scale and self._scale_ref is not None:
                    subs = self.noise.clean(item)[0]
                    subs = subs if subs.ndim == 3 else subs.unsqueeze(0)
                    pred = (
                        self.finder._pred()
                        if self._scale_ref.noise_model is self.noise
                        and self.noise.mode == "online"
                        and self._scale_ref._combination() == self.finder.combination
                        else None
                    )
                    for sub in subs:
                        idx = start_index + offset
                        self._frame_scales[idx] = self._scale_ref.estimate_later(
                            sub, idx, pred
                        )
                        offset += 1
                else:
                    offset += int(item.shape[0]) if item.ndim == 3 else 1
                yield item

        gen = _tee()
        try:
            first = next(gen)
        except StopIteration:
            return PeakStream(iter([]))
        return self.finder.peak_stream(chain([first], gen), start_index=start_index)

    def index_stream(
        self,
        frames: Iterable[Tensor],
        batch_size: int = 8,
        start_index: int = 0,
        update_noise: bool = True,
        recalibrate_every: Optional[int] = None,
        enrich_alpha: Optional[float] = None,
    ) -> IndexStream[FrameIndexResult]:
        """Open a lazy stream with one result per accepted lattice.

        Parameters
        ----------
        frames, batch_size, start_index, update_noise, recalibrate_every, enrich_alpha
            As in :meth:`index_frame_stream`.

        Returns
        -------
        IndexStream
            Single-use stream of accepted lattices in input order; tensors stay
            on the device, and only the writers move them off.
        """
        return self.index_frame_stream(
            frames,
            batch_size,
            start_index,
            update_noise,
            recalibrate_every,
            enrich_alpha,
        ).flatten()

    def index_frame_stream(
        self,
        frames: Iterable[Tensor],
        batch_size: int = 8,
        start_index: int = 0,
        update_noise: bool = True,
        recalibrate_every: Optional[int] = None,
        enrich_alpha: Optional[float] = None,
    ) -> FrameIndexStream:
        """Open a lazy stream with one result per input frame, indexed or not.

        Parameters
        ----------
        frames : iterable of torch.Tensor
            Frames in list order, the first at ``start_index``.
        batch_size : int, default 8
            Frames per batched refinement pass.
        start_index : int, default 0
            Absolute index of the first frame.
        update_noise : bool, default True
            Fold frames into the running statistics; ignored when
            ``recalibrate_every`` is set.
        recalibrate_every : int, optional
            Recalibrate on the first ``n_seed`` frames of every block of this many
            absolute frames and freeze the statistics in between; 0 freezes the
            initial calibration. Reuses the options of a prior :meth:`calibrate`.
        enrich_alpha : float, optional
            Keep only lattices whose enrichment chance probability is at most this,
            tested before overlap exclusion; None disables the gate.

        Returns
        -------
        FrameIndexStream
            Single-use stream of per-frame results, with running counts in
            ``stats``.
        """
        if self.indexer is None:
            raise RuntimeError(
                "index_frame_stream requires a target cell; construct Probixi "
                "with a cell_file (omit it only for peak-only use via "
                "peak_stream)"
            )
        indexer = self.indexer
        if recalibrate_every is not None:
            if recalibrate_every < 0:
                raise ValueError("recalibrate_every must be nonnegative")
            if not self._calibration_options:
                raise RuntimeError("call calibrate before scheduled or frozen indexing")
        self._frame_scales.clear()
        stats = IndexStats()

        # A recalibration boundary can fall inside a batch: schedules split batches
        def _individual() -> Iterator[Tensor]:
            for item in frames:
                yield from item if item.ndim == 3 else (item,)

        def _gen() -> Iterator[FrameIndexResult]:
            source = iter(frames) if recalibrate_every is None else iter(_individual())
            index = start_index
            pending_cells: Optional[list[CellParams]] = (
                [] if self.cell_calibrate and self.cell_calibrate_rounds > 0 else None
            )
            rounds = 0
            while True:
                try:
                    first = next(source)
                except StopIteration:
                    break
                limit = None
                if recalibrate_every:
                    boundary = index // recalibrate_every * recalibrate_every
                    if boundary != self._calibration_boundary:
                        self._recalibrate(boundary)
                    limit = boundary + recalibrate_every - index
                segment = chain(
                    (first,), islice(source, limit - 1) if limit else source
                )
                tc = self.threshold_calibration
                base = indexer.index_frame_stream(
                    self.peak_stream(
                        segment,
                        start_index=index,
                        update_noise=(
                            update_noise if recalibrate_every is None else False
                        ),
                    ),
                    batch_size=batch_size,
                    bright_threshold=(
                        tc.threshold if tc is not None else self.mf_threshold
                    ),
                    enrich_alpha=enrich_alpha,
                )
                for result in base:
                    fs = self._frame_scales.pop(result.frame_index, None)
                    for crystal in result.crystals:
                        if fs is not None:
                            crystal.scale, crystal.scale_sigma = fs.scale, fs.sigma
                    stats.frames += 1
                    stats.hits += result.n_peaks >= MIN_PEAKS_TO_INDEX
                    stats.indexed += bool(result.crystals)
                    stats.crystals += len(result.crystals)
                    if pending_cells is not None and result.crystals:
                        pending_cells.extend(
                            c.cell
                            for c in result.crystals
                            if c.enrich_p is not None
                            and c.enrich_p <= self.cell_calibrate_alpha
                        )
                        if len(pending_cells) >= self.cell_calibrate_after:
                            self._recalibrate_cell(pending_cells)
                            rounds += 1
                            pending_cells = (
                                [] if rounds < self.cell_calibrate_rounds else None
                            )
                    index += 1
                    yield result
                if limit is None:
                    break

        return FrameIndexStream(_gen(), stats=stats)

    def _recalibrate_cell(self, cells: list[CellParams]) -> None:
        assert self.indexer is not None
        old = self.indexer.target_cell
        if self._cell_origin is None:
            self._cell_origin = old
        new = median_cell(cells, template=old)
        edge_shift = max(
            abs(n / o - 1.0)
            for n, o in ((new.a, old.a), (new.b, old.b), (new.c, old.c))
        )
        angle_shift = max(
            abs(n - o)
            for n, o in (
                (new.alpha, old.alpha),
                (new.beta, old.beta),
                (new.gamma, old.gamma),
            )
        )
        applied = self.indexer._cell_matches_target(new, self._cell_origin)
        if applied:
            self.indexer.set_target_cell(new)
            new = self.indexer.target_cell
        self.cell_calibrations.append(
            CellCalibration(len(cells), new, old, edge_shift, angle_shift, applied)
        )
        if not applied:
            warnings.warn(
                f"target cell NOT re-centred: the median of {len(cells)} lattices lies "
                "outside the match window of the cell file; check the cell and geometry.",
                stacklevel=2,
            )
        elif edge_shift > self.cell_calibrate_warn:
            warnings.warn(
                f"target cell re-centred by {100 * edge_shift:.2f}% on its edges "
                f"after {len(cells)} lattices: the cell file and the geometry's "
                "camera length were inconsistent (cell scale and camera length "
                "are degenerate). Indexing continues on the data-derived cell.",
                stacklevel=2,
            )

    def _recalibrate(self, boundary: int) -> None:
        options = self._calibration_options.copy()
        reference = self._scale_ref
        self._noise = self._finder = None
        self._beamstop_qmin = None
        self.threshold_calibration = None
        device = torch.device(self.device or "cpu")
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(boundary)
            self.calibrate(
                seed_frames=_FrameSource(
                    self, range(boundary, boundary + options["n_seed"])
                ),
                **options,
            )
        self._scale_ref = reference
        self._calibration_boundary = boundary

    def scale_stream(
        self,
        frames: Iterable[Tensor],
        start_index: int = 0,
    ) -> Iterator[FrameScale]:
        """Yield each frame's fluence scale relative to the calibration background.

        Each frame is regressed against the background reference frozen by
        :meth:`calibrate`, so online drift does not enter the scale.

        Parameters
        ----------
        frames : iterable of torch.Tensor
            ``(H, W)`` frames; a batch contributes only its first frame.
        start_index : int, default 0
            Absolute index of the first frame.

        Yields
        ------
        FrameScale
            ``frame_index``, ``scale``, ``sigma``, and ``offset`` of each frame.
        """
        if self._scale_ref is None:
            raise RuntimeError("call calibrate() before scale_stream() (no reference)")
        ref = self._scale_ref
        for offset, item in enumerate(frames):
            frame = item[0] if item.ndim == 3 else item
            yield ref.estimate(frame, start_index + offset)


def _beamstop_qmin_from_histogram(
    ph: Tensor, pix: Tensor, edges: Tensor, n_bins: int
) -> Optional[float]:
    # ph, pix: seed peaks and valid pixels per |q| bin; returns the spike's outer edge
    density = ph / pix.clamp_min(1.0)
    big = pix >= _BEAMSTOP_MIN_BIN_PIXELS
    if not bool(big.any()):
        return None
    med = float(density[big].median())
    if med <= 0:
        return None

    masked = (pix <= 0).nonzero().flatten().tolist()
    lo_bin = (max(masked) + 1) if masked else 0
    hi_cap = min(n_bins - 1, lo_bin + _BEAMSTOP_MAX_EXTEND_BINS - 1)
    cand = [
        i
        for i in range(lo_bin, hi_cap + 1)
        if float(pix[i]) >= _BEAMSTOP_MIN_BIN_PIXELS
        and float(ph[i]) >= _BEAMSTOP_MIN_BIN_PEAKS
    ]
    if not cand:
        return None
    amax = max(cand, key=lambda i: float(density[i]))
    ref = float(density[amax])
    if ref < _BEAMSTOP_SPIKE_RATIO * med:
        return None
    thr = _BEAMSTOP_EDGE_FRACTION * ref
    hi = amax
    while hi + 1 <= hi_cap and float(density[hi + 1]) >= thr:
        hi += 1
    return float(edges[hi + 1])


class _FrameSource:
    # Each pass re-reads the frames from disk, so only one is resident at a time.
    def __init__(self, pipeline: Probixi, indices: Iterable[int]):
        self._pipeline = pipeline
        self._indices = list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[Tensor]:
        return self._pipeline._frames_at(self._indices)


def camera_length_error(
    indexer: Indexer, crystals: list
) -> Optional[tuple[float, float, int]]:
    # clen error eps from radial residuals, per-crystal cell scale profiled out (Huber)
    g0 = indexer.geometry
    bases = _panel_bases(g0, None, torch.float64)
    xs, us, ys, ids = [], [], [], []
    for c in crystals:
        sel = c.indexed_mask.cpu()
        if int(sel.sum()) < 6:
            continue
        g = indexer._geometry_at(c.frame_index)
        L = float(g["clen"]) / float(g["pixel_size"])
        S = c.hkl.cpu()[sel].double() @ c.A.cpu().double().T * float(g["wavelength"])
        pred = S[:, :2] * (L / (S[:, 2:] + 1.0))
        obs = _lab_xy_pixels(c.positions.cpu()[sel].double(), g, bases)
        rho = obs.norm(dim=-1).clamp_min(1e-9)
        dr = ((obs - pred) * obs).sum(-1) / rho
        keep = dr.abs() < _CLEN_MAX_RESIDUAL_PX
        xs.append(rho[keep])  # radial shift per unit camera-length error
        us.append((rho * torch.sqrt(rho * rho + L * L) / L)[keep])  # rho / cos 2theta
        ys.append(dr[keep])
        ids.append(torch.full((int(keep.sum()),), len(ids)))
    if len(ids) < 10:
        return None
    x, u, y, cid = (torch.cat(t) for t in (xs, us, ys, ids))
    if len(x) < 10 * len(ids):
        return None

    def per(v: Tensor) -> Tensor:  # per-crystal sums, per peak
        return torch.zeros(len(ids), dtype=v.dtype).index_add_(0, cid, v)[cid]

    w = torch.ones_like(y)
    for _ in range(5):
        n = per(w * u * u).clamp_min(1e-12)
        X = x - u * per(w * u * x) / n
        Y = y - u * per(w * u * y) / n
        eps = float((w * X * Y).sum() / (w * X * X).sum().clamp_min(1e-12))
        res = Y - X * eps
        s = 1.4826 * float(res.abs().median())
        w = (1.5 * s / res.abs().clamp_min(1e-12)).clamp_max(1.0)  # Huber, k = 1.5
    se = s / math.sqrt(float((w * X * X).sum().clamp_min(1e-12)))
    return eps, se, len(ids)


def _frame_wavelengths(meta: Metadata) -> Optional[list[float]]:
    # per-frame wavelengths (A) from event photon energies, None without any
    infos = list(meta.files.values())
    if not any(getattr(i, "photon_energy_eV", None) for i in infos):
        return None
    out: list[float] = []
    for info in infos:
        e = info.photon_energy_eV or []
        start = int(info.event_start)
        out += [
            EV_ANGSTROM / e[k] if k < len(e) and e[k] > 0 else meta.geometry.wavelength
            for k in range(start, start + int(info.n_frames))
        ]
    return out


def _blank_seed_floor(levels: list[float], frac: float) -> float:
    if not levels or frac <= 0.0:
        return float("-inf")
    ref = float(torch.tensor(levels).median())
    return frac * ref if ref > 0.0 else float("-inf")


def _lab_radius_map(geom, frame_size: tuple[int, int], device=None) -> Optional[Tensor]:
    if geom is None:
        return None
    g = geom.to_dict()
    bases = _panel_bases(g, device, torch.float32)
    if bases is None:
        return None
    rows, cols = frame_size
    out = torch.empty(frame_size, dtype=torch.float32, device=device)
    cc = torch.arange(cols, dtype=torch.float32, device=device)
    step = max(1, _LAB_RADIUS_CHUNK_PIXELS // max(cols, 1))
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        n = r1 - r0
        rr = torch.arange(r0, r1, dtype=torch.float32, device=device)
        pos = torch.stack(
            [
                rr.view(-1, 1).expand(n, cols).reshape(-1),
                cc.view(1, -1).expand(n, cols).reshape(-1),
            ],
            dim=-1,
        )
        xy = _lab_xy_pixels(pos, g, bases)
        out[r0:r1] = torch.linalg.vector_norm(xy, dim=-1).view(n, cols)
    return out


def _panel_boxes(geom, frame_size: tuple[int, int]) -> Optional[list]:
    panels = list((getattr(geom, "panels", None) or {}).values())
    if len(panels) < 2:
        return None
    rows, cols = frame_size
    boxes = []
    for panel in panels:
        try:
            r0, r1 = int(panel["min_ss"]), int(panel["max_ss"])
            c0, c1 = int(panel["min_fs"]), int(panel["max_fs"])
        except (KeyError, TypeError, ValueError):
            return None
        r0, r1 = max(0, r0), min(rows - 1, r1)
        c0, c1 = max(0, c0), min(cols - 1, c1)
        if r0 > r1 or c0 > c1:
            return None
        boxes.append((r0, r1 + 1, c0, c1 + 1))
    return boxes


def _mask_panel_edges(mask: Tensor, geom) -> None:
    # mask ``mask_edge_pixels`` pixels around the border of every panel
    n = geom.parameters.get("mask_edge_pixels")
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        return
    n = int(n)
    if n <= 0:
        return
    rows, cols = int(mask.shape[0]), int(mask.shape[1])
    for panel in (geom.panels or {}).values():
        try:
            r0, r1 = int(panel["min_ss"]), int(panel["max_ss"])
            c0, c1 = int(panel["min_fs"]), int(panel["max_fs"])
        except (KeyError, TypeError, ValueError):
            continue
        r0, r1 = max(0, r0), min(rows - 1, r1)
        c0, c1 = max(0, c0), min(cols - 1, c1)
        if r0 > r1 or c0 > c1:
            continue
        mask[r0 : min(r0 + n, r1 + 1), c0 : c1 + 1] = False
        mask[max(r1 - n + 1, r0) : r1 + 1, c0 : c1 + 1] = False
        mask[r0 : r1 + 1, c0 : min(c0 + n, c1 + 1)] = False
        mask[r0 : r1 + 1, max(c1 - n + 1, c0) : c1 + 1] = False


def shadow_mask(
    pixel_mean: Tensor, radial_mean: Tensor, valid: Tensor, grow: int
) -> Optional[Tensor]:
    # (H, W) bool or None: dark pixels in mostly dark windows, grown by grow px
    ok = valid & torch.isfinite(pixel_mean) & (radial_mean > 0)
    if int(ok.sum()) < 1000:
        return None
    ratio = (pixel_mean / radial_mean.clamp_min(1e-12)).clamp_min(1e-6)
    log_ratio = ratio.log()
    body = log_ratio[ok]
    med = body.median()
    mad = (body - med).abs().median() * 1.4826
    if float(mad) <= 0:
        return None
    cut = min(float(med - _SHADOW_MAD_K * mad), math.log(_SHADOW_MAX_RATIO))
    dark = ok & (log_ratio < cut)
    if not bool(dark.any()):
        return None
    k = 2 * int(grow) + 1
    dense = torch.nn.functional.avg_pool2d(
        dark[None, None].float(),
        k,
        stride=1,
        padding=int(grow),
        count_include_pad=False,
    )[0, 0]
    core = dark & (dense >= _SHADOW_CORE_FRACTION)
    if not bool(core.any()):
        return None
    grown = torch.nn.functional.max_pool2d(
        core[None, None].float(), k, stride=1, padding=int(grow)
    )[0, 0]
    return grown > 0


def _mask_lab_bad_regions(mask: Tensor, geom) -> None:
    # Bad regions given in lab-frame x/y (px), such as a beamstop or guard ring
    regions = [br for br in geom.bad_regions if br.is_lab_frame]
    if not regions:
        return
    bases = _panel_bases(geom.to_dict(), mask.device, torch.float32)
    if bases is None:
        return
    rows, cols = int(mask.shape[0]), int(mask.shape[1])
    for name, basis in zip((geom.panels or {}).keys(), bases):
        min_ss, max_ss = int(basis[0]), int(basis[1])
        min_fs, max_fs = int(basis[2]), int(basis[3])
        cx, cy = float(basis[4]), float(basis[5])
        fsx, fsy = float(basis[6]), float(basis[7])
        ssx, ssy = float(basis[8]), float(basis[9])
        r0, r1 = max(0, min_ss), min(rows - 1, max_ss)
        c0, c1 = max(0, min_fs), min(cols - 1, max_fs)
        if r0 > r1 or c0 > c1:
            continue
        ss_i = torch.arange(
            r0 - min_ss, r1 - min_ss + 1, device=mask.device, dtype=torch.float32
        )[:, None]
        fs_j = torch.arange(
            c0 - min_fs, c1 - min_fs + 1, device=mask.device, dtype=torch.float32
        )[None, :]
        x = cx + fs_j * fsx + ss_i * ssx
        y = cy + fs_j * fsy + ss_i * ssy
        for br in regions:
            if br.panel is not None and br.panel != name:
                continue
            hit = (x >= br.min_x) & (x <= br.max_x) & (y >= br.min_y) & (y <= br.max_y)
            if bool(hit.any()):
                mask[r0 : r1 + 1, c0 : c1 + 1] &= ~hit
