from __future__ import annotations

_LAZY = {
    # pipeline
    "Probixi": ".probixi",
    "auto_device": ".probixi",
    # citation
    "citation": ".probixi",
    "__citation__": ".probixi",
    # indexer config
    "FrameIndexResult": ".indexer",
    "FrameIndexStream": ".indexer",
    "SeedConfig": ".indexer",
    "RefineConfig": ".indexer",
    "CellMatchConfig": ".indexer",
    "IntegrateConfig": ".indexer",
    # indexing ambiguity
    "Ambigator": ".ambigator",
    "AmbigatorResult": ".ambigator",
    # multi-GPU
    "run_data_parallel": ".multigpu",
    "run_block_from_env": ".multigpu",
    "merge_streams": ".multigpu",
    "BlockConfig": ".multigpu",
    # output writers
    "DataOffloader": ".io",
    "PeakOffloader": ".io",
    "DuckDBOffloader": ".io",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
