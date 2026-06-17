from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain, islice
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional, Union

import torch
from torch import Tensor

from .indexer import (
    CellMatchConfig,
    Indexer,
    IndexStream,
    IntegrateConfig,
    RefineConfig,
    SeedConfig,
)
from .io import CellParams, DataLoader, Metadata, iter_frames
from .peakfinding import PeakFinder, PeakStream
from .peakfinding.noise import (
    CalibrationResult,
    FrameScale,
    NoiseModel,
    ScaleReference,
    ThresholdCalibration,
    calibrate_noise,
    calibrate_threshold,
    fit_eigen_background,
    fit_photon_transfer,
)

PathLike = Union[str, Path]


@dataclass
class Probixi:
    """Self-calibrating probabilistic peak finder and indexer.

    Observe the detector noise on a slice of seed frames, calibrate a
    probabilistic detector from it (peak/noise scale ``kappa``, peak prior, the
    pixel/radial/panel mean blend, a variance scale pinning the background to
    ``N(0, 1)``, and the matched-filter detection threshold), then stream frames
    to emit indexed, merge-ready results.

    Parameters
    ----------
    list_file, geometry_file, cell_file : path-like
        CrystFEL ``.lst`` list, ``.geom`` geometry and ``.cell`` unit-cell inputs.
    noise_mode : {"online", "per_frame"}, default "online"
        Whether the running noise model keeps updating per frame ("online",
        tracking slow drift) or resets each frame.
    warmup_frames : int, default 16
        Frames observed before the dead-pixel mask is committed.
    finder_kappa, posterior_threshold, candidate_threshold
        Pre-calibration detector defaults; ``calibrate`` overrides ``kappa`` and
        the peak prior with learned values.
    matched_filter : bool, default True
        Use the multi-scale matched filter (the recommended operating point).
    mf_scales, mf_threshold
        Matched-filter kernel scales and the fallback threshold (learned by
        ``calibrate`` when ``target_noise_peaks`` is set).
    flux_variance : bool, default False
        Replace the frozen variance floor with a learned photon-transfer curve
        so dim/low-flux shots are whitened against their own Poisson noise.
        Opt-in; intended for XFEL/SFX or jet-intensity-variable data.
    seed, refine, cell_match, integrate
        Optional indexer configuration objects.
    device, dtype
        Torch device and frame dtype.

    Attributes
    ----------
    threshold_calibration : ThresholdCalibration or None
        The fitted matched-filter threshold result, set by ``calibrate``.
    """

    list_file: PathLike
    geometry_file: PathLike
    cell_file: PathLike

    noise_mode: Literal["per_frame", "online"] = "online"
    warmup_frames: int = 16
    finder_kappa: float = 10.0
    posterior_threshold: float = 0.5
    candidate_threshold: Optional[float] = None
    matched_filter: bool = True
    mf_scales: tuple[float, ...] = (1.0, 1.6, 2.4)
    mf_threshold: float = 5.0
    flux_variance: bool = False
    flux_var_floor: float = 0.15
    seed: Optional[SeedConfig] = None
    refine: Optional[RefineConfig] = None
    cell_match: Optional[CellMatchConfig] = None
    integrate: Optional[IntegrateConfig] = None
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    loader: DataLoader = field(init=False, repr=False)
    indexer: Indexer = field(init=False, repr=False)
    threshold_calibration: Optional[ThresholdCalibration] = field(
        default=None, init=False, repr=False
    )
    _noise: Optional[NoiseModel] = field(default=None, init=False, repr=False)
    _finder: Optional[PeakFinder] = field(default=None, init=False, repr=False)
    _scale_ref: Optional[ScaleReference] = field(default=None, init=False, repr=False)
    _frame_scales: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.loader = DataLoader(
            self.list_file, geometry_file=self.geometry_file, cell_file=self.cell_file
        )
        m = self.loader.metadata
        if m.geometry is None:
            raise ValueError(f"could not parse geometry from {self.geometry_file}")
        if m.cell is None:
            raise ValueError(f"could not parse cell from {self.cell_file}")
        self.indexer = Indexer(
            m.geometry.to_dict(),
            m.cell,
            seed=self.seed,
            refine=self.refine,
            cell_match=self.cell_match,
            integrate=self.integrate,
            device=self.device,
        )

    @property
    def metadata(self) -> Metadata:
        """Parsed run metadata (frames, geometry, cell)."""
        return self.loader.metadata

    @property
    def geometry(self) -> dict:
        """Detector geometry dict (beam_center, clen, pixel_size, wavelength, ...)."""
        return self.indexer.geometry

    @property
    def target_cell(self) -> CellParams:
        """Target unit cell that accepted orientations must match."""
        return self.indexer.target_cell

    @property
    def noise(self) -> NoiseModel:
        """The live noise model (built lazily on the first frame seen)."""
        if self._noise is None:
            raise RuntimeError(
                "Probixi not yet seeded; pass a frame through fit_noise, calibrate, "
                "or a stream method first to infer the frame size."
            )
        return self._noise

    @property
    def finder(self) -> PeakFinder:
        """The peak finder over the noise model (built lazily on the first frame)."""
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
            Half-open absolute frame range over the run; full run if omitted.
        batch_size : int, default 1
            Frames stacked per yielded tensor (1 yields single frames).
        prefetch : int, default 2
            Batches read ahead on a background thread to overlap I/O.

        Returns
        -------
        Iterator[torch.Tensor]
            Frames on the configured device and dtype.
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

    def _ensure_built(self, item: Tensor) -> None:
        if self._noise is not None:
            return
        frame_size = (int(item.shape[-2]), int(item.shape[-1]))
        first = item[0] if item.ndim == 3 else item
        self._noise = NoiseModel(
            frame_size=frame_size,
            mode=self.noise_mode,
            warmup_frames=self.warmup_frames,
            valid_mask=self._static_mask(first, frame_size),
            device=self.device,
            dtype=self.dtype,
        )
        self._finder = PeakFinder(
            self._noise,
            kappa=self.finder_kappa,
            posterior_threshold=self.posterior_threshold,
            candidate_threshold=self.candidate_threshold,
            matched_filter=self.matched_filter,
            mf_scales=self.mf_scales,
            mf_threshold=self.mf_threshold,
            flux_variance=self.flux_variance,
            flux_var_floor=self.flux_var_floor,
        )

    def _static_mask(self, frame: Tensor, frame_size: tuple[int, int]) -> Tensor:
        # a-priori bad pixels: at/above max_adu (gaps/dead/saturated) + geometry
        # bad regions, kept out of the background and detection.
        mask = torch.ones(frame_size, dtype=torch.bool, device=frame.device)
        geom = self.loader.metadata.geometry
        max_adu = geom.parameters.get("max_adu") if geom else None
        if isinstance(max_adu, (int, float)):
            mask &= frame < float(max_adu)
        for br in geom.bad_regions if geom else []:
            r0, r1 = max(0, br.min_ss), min(frame_size[0] - 1, br.max_ss)
            c0, c1 = max(0, br.min_fs), min(frame_size[1] - 1, br.max_fs)
            if r0 <= r1 and c0 <= c1:
                mask[r0 : r1 + 1, c0 : c1 + 1] = False
        return mask.to(self.device)

    def _update_noise(self, item: Tensor) -> None:
        if item.ndim == 3:
            for frame in item:
                self.noise.update(frame)
        else:
            self.noise.update(item)

    def fit_noise(self, frames: Iterable[Tensor]) -> "Probixi":
        """Observe ``frames`` to warm the noise model (builds it on the first).

        Returns
        -------
        Probixi
            Self, for chaining.
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
        """Write a noise-model diagnostic GIF, driving the live model.

        Streams ``frames`` (or a ``[start, stop)`` slice of the run) through the
        pipeline's live noise model, snapshotting after every ``batch_size``
        frames, and animates the running mean background, its radial profile,
        and the per-batch drift. The model is built lazily (with the
        geometry-aware bad-pixel mask) if it has not been seeded yet.

        Because it drives the *live* model, every frame is folded into the
        running stats: run it before ``calibrate`` to watch warmup, or on a
        fresh ``Probixi`` to inspect a run's drift. One mean-image snapshot is
        retained per batch, so prefer a bounded slice for long runs.

        Parameters
        ----------
        path : path-like
            Output ``.gif`` path.
        frames : iterable of torch.Tensor, optional
            Explicit frames; defaults to the ``[start, stop)`` run slice.
        start, stop : int, optional
            Half-open frame range when ``frames`` is not given.
        batch_size : int, default 15
            Frames folded in between animation snapshots.
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
        return self.noise.diagnostics(
            chain([first], it), path, batch_size=batch_size, **kwargs
        )

    def calibrate(
        self,
        n_seed: int = 32,
        seed_frames: Optional[Iterable[Tensor]] = None,
        eigen_modes: int = 0,
        target_noise_peaks: Optional[float] = 5.0,
        threshold_opts: Optional[dict] = None,
        **opts,
    ) -> CalibrationResult:
        """Calibrate the detector on seed frames, then freeze the learned params.

        Warms the noise model, then learns the peak/noise scale ``kappa`` and
        peak prior (EM mixture) and the mean blend + variance scale that pin the
        background ``z`` to ``N(0, 1)``. With ``target_noise_peaks`` set (the
        default), also calibrates the matched-filter threshold so a signal-free
        frame yields at most that many noise blobs, and installs it on the
        finder. With ``flux_variance`` enabled on the pipeline, also fits the
        photon-transfer curve.

        Parameters
        ----------
        n_seed : int, default 32
            Leading frames to calibrate on when ``seed_frames`` is not given.
        seed_frames : iterable of torch.Tensor, optional
            Explicit calibration frames; overrides ``n_seed``.
        eigen_modes : int, default 0
            If > 0, also fit this many low-rank background modes (XFEL/SFX).
        target_noise_peaks : float or None, default 5.0
            Matched-filter operating point (expected noise blobs per signal-free
            frame). ``None`` skips threshold calibration. 5 ~ ``mf_threshold`` 5.5.
        threshold_opts : dict, optional
            Extra keyword arguments forwarded to ``calibrate_threshold``.
        **opts
            Extra keyword arguments forwarded to ``calibrate_noise``.

        Returns
        -------
        CalibrationResult
            The applied noise calibration.
        """
        seed = (
            list(seed_frames)
            if seed_frames is not None
            else list(islice(self.frames(), n_seed))
        )
        if not seed:
            raise ValueError("no seed frames available to calibrate on")
        self.fit_noise(seed)
        result = calibrate_noise(self.noise, seed, warm=False, **opts)
        result.apply(self.noise, self.finder)
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
                target_noise_peaks=float(target_noise_peaks),
                finder=self.finder,
                **topts,
            )
        self._scale_ref = ScaleReference.from_noise_model(self.noise)
        return result

    def peak_stream(
        self,
        frames: Iterable[Tensor],
        start_index: int = 0,
        update_noise: bool = True,
    ) -> PeakStream:
        """Open a lazy stream of per-frame peak results.

        Parameters
        ----------
        frames : iterable of torch.Tensor
            Frames to search; the first triggers lazy model construction.
        start_index : int, default 0
            Absolute index assigned to the first frame (so results carry true
            run indices when starting partway through).
        update_noise : bool, default True
            Fold each frame into the running noise model as it passes.

        Returns
        -------
        PeakStream
            Lazy, composable stream of ``PeakResult`` (torch-resident).
        """

        def _tee() -> Iterator[Tensor]:
            offset = 0
            for item in frames:
                self._ensure_built(item)
                if update_noise:
                    self._update_noise(item)
                if self._scale_ref is not None:
                    subs = item if item.ndim == 3 else item.unsqueeze(0)
                    for sub in subs:
                        idx = start_index + offset
                        self._frame_scales[idx] = self._scale_ref.estimate(sub, idx)
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
    ) -> IndexStream:
        """Open a lazy stream of indexing solutions over ``frames``.

        Runs the full pipeline per frame: detect peaks, lift to reciprocal
        space, seed and refine an orientation whose cell matches the target, then
        predict and integrate the lattice.

        Parameters
        ----------
        frames : iterable of torch.Tensor
            Frames to process.
        batch_size : int, default 8
            Frames per batched refinement pass.
        start_index : int, default 0
            Absolute index assigned to the first frame.
        update_noise : bool, default True
            Fold each frame into the running noise model as it passes.

        Returns
        -------
        IndexStream
            Lazy stream of ``IndexResult``, one per indexed frame.
        """
        self._frame_scales.clear()
        base = self.indexer.index_stream(
            self.peak_stream(
                frames, start_index=start_index, update_noise=update_noise
            ),
            batch_size=batch_size,
        )

        def _attach() -> Iterator:
            for r in base:
                fs = self._frame_scales.pop(r.frame_index, None)
                if fs is not None:
                    r.scale, r.scale_sigma = fs.scale, fs.sigma
                yield r

        return type(base)(_attach())

    def scale_stream(
        self,
        frames: Iterable[Tensor],
        start_index: int = 0,
    ) -> Iterator[FrameScale]:
        """Infer the per-frame relative intensity scale, frame by frame.

        Each frame is regressed against the background reference frozen at
        ``calibrate`` (so online drift-tracking does not contaminate the scale),
        yielding ``g_f`` and its uncertainty -- the shot-to-shot fluence scale
        that dominates at an XFEL and that initializes downstream per-pattern
        scaling. Requires ``calibrate`` to have been run.

        Parameters
        ----------
        frames : iterable of torch.Tensor
            Frames to scale (single 2-D frames).
        start_index : int, default 0
            Absolute index assigned to the first frame.

        Yields
        ------
        FrameScale
            ``(frame_index, scale, sigma, offset)`` per frame.
        """
        if self._scale_ref is None:
            raise RuntimeError("call calibrate() before scale_stream() (no reference)")
        ref = self._scale_ref
        for offset, item in enumerate(frames):
            frame = item[0] if item.ndim == 3 else item
            yield ref.estimate(frame, start_index + offset)
