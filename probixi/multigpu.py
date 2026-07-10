from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.multiprocessing as mp

from .io import DataLoader, DataOffloader, DuckDBOffloader, is_duckdb_path
from .probixi import Probixi

PathLike = Union[str, Path]

_CHUNK_MARKER = "----- Begin chunk -----"
_SERIAL_PREFIX = "Image serial number:"
_STREAM_VERSION_PREFIX = "CrystFEL stream format"
_DB_DATA_TABLES = ("frames", "reflections", "peaks")
_DB_META_TABLES = ("geometry", "panels", "cell")

__all__ = [
    "run_data_parallel",
    "run_block",
    "run_block_from_env",
    "merge_streams",
    "merge_dbs",
    "resolve_devices",
    "block_bounds",
    "BlockConfig",
]


def block_bounds(start: int, stop: int, rank: int, world_size: int) -> tuple[int, int]:
    """Contiguous, balanced ``[lo, hi)`` sub-range of ``[start, stop)`` for ``rank``"""
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    total = max(0, stop - start)
    base, rem = divmod(total, world_size)
    lo = start + rank * base + min(rank, rem)
    hi = lo + base + (1 if rank < rem else 0)
    return lo, hi


def resolve_devices(
    devices: Optional[Union[int, Sequence[Union[str, torch.device]]]] = None,
) -> list[torch.device]:
    """Normalise a device spec to a list of ``torch.device``.

    ``None`` -> every visible CUDA device (or ``[cpu]`` if none); an ``int`` ->
    the first N CUDA devices; a sequence -> those devices verbatim
    (e.g. ``["cuda:0", "cuda:1"]`` or ``["cpu", "cpu"]`` for testing).
    """
    if devices is None:
        if torch.cuda.is_available():
            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        return [torch.device("cpu")]
    if isinstance(devices, int):
        if devices < 1:
            raise ValueError("device count must be >= 1")
        return [torch.device(f"cuda:{i}") for i in range(devices)]
    out = [torch.device(d) for d in devices]
    if not out:
        raise ValueError("devices must be non-empty")
    return out


def _validate_cuda_devices(dev_list: Sequence[torch.device]) -> None:
    # fail fast in the parent
    cuda = [d for d in dev_list if d.type == "cuda"]
    if not cuda:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA devices requested but CUDA is not available")
    count = torch.cuda.device_count()
    bad = sorted({str(d) for d in cuda if (d.index or 0) >= count})
    if bad:
        raise RuntimeError(
            f"requested CUDA device(s) not present: {', '.join(bad)} ({count} visible)"
        )


def merge_streams(part_paths: Sequence[PathLike], output_path: PathLike) -> int:
    """Concatenate per-rank ``.stream`` files into one, renumbering serials."""
    serial = 0
    header_written = False
    with Path(output_path).open("w") as out:
        for p in part_paths:
            part = Path(p)
            if not part.exists() or part.stat().st_size == 0:
                continue  # skip a crashed/empty worker's part
            has_header = None
            in_chunks = False
            with part.open("r") as fh:
                for i, line in enumerate(fh):
                    if i == 0:
                        has_header = line.startswith(_STREAM_VERSION_PREFIX)
                    if line.startswith(_CHUNK_MARKER):
                        in_chunks = True
                    if not in_chunks:
                        if not header_written and has_header:  # header from first part
                            out.write(line)
                        continue
                    if line.startswith(_SERIAL_PREFIX):
                        serial += 1
                        out.write(f"{_SERIAL_PREFIX} {serial}\n")
                    else:
                        out.write(line)
            if has_header:
                header_written = True
    return serial


def merge_dbs(part_paths: Sequence[PathLike], output_path: PathLike) -> int:
    """Union per-rank DuckDB parts into one database."""
    import duckdb

    from .io import db as _db

    out = Path(output_path)
    out.unlink(missing_ok=True)
    conn = duckdb.connect(str(out))
    meta_done = False
    try:
        conn.execute(_db._SCHEMA)
        for i, p in enumerate(part_paths):
            part = Path(p)
            if not part.exists() or part.stat().st_size == 0:
                continue
            alias = f"part{i}"
            escaped = str(part).replace("'", "''")
            try:
                conn.execute(f"ATTACH '{escaped}' AS {alias} (READ_ONLY)")
            except duckdb.Error:
                continue  # skip a crashed/corrupt worker's part
            try:
                if not meta_done:
                    for tbl in _DB_META_TABLES:
                        conn.execute(f"INSERT INTO {tbl} SELECT * FROM {alias}.{tbl}")
                    meta_done = True
                for tbl in _DB_DATA_TABLES:
                    conn.execute(f"INSERT INTO {tbl} SELECT * FROM {alias}.{tbl}")
            finally:
                conn.execute(f"DETACH {alias}")
        conn.execute(_db._INDEXES)
        (n_indexed,) = conn.execute(
            "SELECT COUNT(*) FROM frames WHERE indexed"
        ).fetchone()
    finally:
        conn.close()
    return int(n_indexed)


# worker
@dataclass
class BlockConfig:
    list_file: str
    geometry_file: str
    cell_file: Optional[str]
    start: int
    stop: int
    batch_size: int = 8
    seed_frames: int = 32
    target_noise_peaks: Optional[float] = 5.0
    noise_mode: str = "online"
    warmup_frames: int = 16
    panel: str = "0"
    enrich_gate: bool = False
    enrich_alpha: float = 1e-3
    threads: Optional[int] = None
    quiet: bool = False
    db: bool = False  # write per-rank DuckDB parts instead of .stream parts


def _part_path(output: PathLike, rank: int) -> Path:
    return Path(f"{output}.rank{rank}")


def _remove_part(part: PathLike) -> None:
    Path(part).unlink(missing_ok=True)
    Path(f"{part}.stats.json").unlink(missing_ok=True)
    Path(f"{part}.wal").unlink(missing_ok=True)


def run_block(
    rank: int,
    world_size: int,
    device: Union[str, torch.device],
    cfg: BlockConfig,
    part_path: PathLike,
) -> dict:
    """Calibrate on the shared seed frames, then index this rank's block."""
    # cap CPU threads before any torch work so co-resident workers don't oversubscribe
    if cfg.threads is not None and cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    p = Probixi(
        list_file=cfg.list_file,
        geometry_file=cfg.geometry_file,
        cell_file=cfg.cell_file,
        noise_mode=cfg.noise_mode,  # type: ignore[arg-type]
        warmup_frames=cfg.warmup_frames,
        device=dev,
    )
    if p.indexer is None:
        raise RuntimeError("multi-GPU indexing requires a cell_file")
    # deterministic calibration -> every rank recovers identical detector params
    p.calibrate(n_seed=cfg.seed_frames, target_noise_peaks=cfg.target_noise_peaks)

    lo, hi = block_bounds(cfg.start, cfg.stop, rank, world_size)
    if not cfg.quiet:
        print(
            f"[rank {rank}/{world_size}] device={dev} frames [{lo}, {hi})", flush=True
        )

    stream = p.index_stream(
        p.frames(start=lo, stop=hi), batch_size=cfg.batch_size, start_index=lo
    )
    if cfg.enrich_gate:
        stream = stream.enrich_gate(cfg.enrich_alpha)

    n = 0
    offload_kwargs = dict(
        geometry=p.geometry,
        cell=p.target_cell,
        geometry_file=cfg.geometry_file,
        files=p.metadata.files,
        panel=cfg.panel,
    )
    if cfg.db:
        offloader = DuckDBOffloader
        # each rank backfills only its own block's non-indexed frames
        offload_kwargs["frame_range"] = (lo, hi)
    else:
        offloader = DataOffloader
    with offloader(part_path, **offload_kwargs) as off:
        for result in stream:
            off.write(result)
            n += 1

    stats = {
        "rank": rank,
        "device": str(dev),
        "lo": lo,
        "hi": hi,
        "frames": stream.stats.frames,
        "hits": stream.stats.hits,
        "indexed": n,
    }
    Path(f"{part_path}.stats.json").write_text(json.dumps(stats))
    if not cfg.quiet:
        print(
            f"[rank {rank}/{world_size}] done: {stats['frames']} frames, "
            f"{stats['hits']} hits, {n} indexed",
            flush=True,
        )
    return stats


def _spawn_entry(
    rank: int,
    world_size: int,
    devices: list[torch.device],
    cfg: BlockConfig,
    output: str,
) -> None:
    run_block(rank, world_size, devices[rank], cfg, _part_path(output, rank))


def run_data_parallel(
    list_file: PathLike,
    geometry_file: PathLike,
    cell_file: PathLike,
    output: PathLike,
    *,
    devices: Optional[Union[int, Sequence[Union[str, torch.device]]]] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    batch_size: int = 8,
    seed_frames: int = 32,
    target_noise_peaks: Optional[float] = 5.0,
    noise_mode: str = "online",
    warmup_frames: int = 16,
    panel: str = "0",
    enrich_gate: bool = False,
    enrich_alpha: float = 1e-3,
    threads_per_worker: Optional[int] = None,
    keep_parts: bool = False,
    quiet: bool = False,
) -> Path:
    """Run indexing across several devices on one node and merge the outputs."""
    dev_list = resolve_devices(devices)
    _validate_cuda_devices(dev_list)
    world = len(dev_list)
    if threads_per_worker is None and world > 1:
        threads_per_worker = max(1, (os.cpu_count() or world) // world)

    # resolve the absolute frame range once (metadata-only scan)
    loader = DataLoader(list_file, geometry_file=geometry_file, cell_file=cell_file)
    if loader.metadata.cell is None:
        raise RuntimeError("multi-GPU indexing requires a cell_file")
    n_frames = loader.metadata.n_frames
    lo = int(start) if start is not None else 0
    hi = int(stop) if stop is not None else n_frames
    hi = min(hi, n_frames)

    cfg = BlockConfig(
        list_file=str(list_file),
        geometry_file=str(geometry_file),
        cell_file=str(cell_file),
        start=lo,
        stop=hi,
        batch_size=batch_size,
        seed_frames=seed_frames,
        target_noise_peaks=target_noise_peaks,
        noise_mode=noise_mode,
        warmup_frames=warmup_frames,
        panel=panel,
        enrich_gate=enrich_gate,
        enrich_alpha=enrich_alpha,
        threads=threads_per_worker,
        quiet=quiet,
        db=is_duckdb_path(output),
    )

    part_paths = [_part_path(output, r) for r in range(world)]
    for part in part_paths:  # drop stale parts from a previous run
        _remove_part(part)
    if world == 1:
        # inline: no process overhead, and works for a single cpu/mps device
        run_block(0, 1, dev_list[0], cfg, part_paths[0])
    else:
        mp.spawn(
            _spawn_entry,
            args=(world, dev_list, cfg, str(output)),
            nprocs=world,
            join=True,
        )

    n_chunks = (
        merge_dbs(part_paths, output) if cfg.db else merge_streams(part_paths, output)
    )

    # aggregate block stats, then remove the per-rank parts
    totals = {"frames": 0, "hits": 0, "indexed": 0}
    for part in part_paths:
        sidecar = Path(f"{part}.stats.json")
        if sidecar.exists():
            s = json.loads(sidecar.read_text())
            for k in totals:
                totals[k] += int(s.get(k, 0))
    if not quiet:
        print(
            f"Merged {world} block(s) -> {output}: {totals['frames']} frames, "
            f"{totals['hits']} hits, {n_chunks} indexed",
            flush=True,
        )
    if not keep_parts:
        for part in part_paths:
            _remove_part(part)
    return Path(output)


# Cluster (SLURM / torchrun) launcher
def _env_int(*names: str, default: Optional[int] = None) -> Optional[int]:
    for name in names:
        val = os.environ.get(name)
        if val is not None:
            try:
                return int(val)
            except ValueError as exc:
                raise ValueError(f"env var {name}={val!r} is not an integer") from exc
    return default


def run_block_from_env(cfg: BlockConfig, output: PathLike) -> dict:
    """Process this process's block using rank/world/local-rank from the env.

    Reads ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` (torchrun) or
    ``SLURM_PROCID``/``SLURM_NTASKS``/``SLURM_LOCALID`` (srun). Each rank binds to
    ``cuda:LOCAL_RANK`` (or CPU) and writes ``<output>.rank{rank}``. After every
    rank finishes, call :func:`merge_streams` on the parts from a shared
    filesystem (e.g. from rank 0 once a barrier confirms all parts exist).
    """
    rank = _env_int("RANK", "SLURM_PROCID", default=0) or 0
    world = _env_int("WORLD_SIZE", "SLURM_NTASKS", default=1) or 1
    local = _env_int("LOCAL_RANK", "SLURM_LOCALID", default=rank) or 0
    device = (
        torch.device(f"cuda:{local}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    return run_block(rank, world, device, cfg, _part_path(output, rank))
