from __future__ import annotations

# Imported on first access, so `import probixi` and the CLI's --help skip torch
_LAZY = {
    "Probixi": ".probixi",
    "auto_device": ".probixi",
    "citation": ".probixi",
    "__citation__": ".probixi",
    "FrameIndexResult": ".indexer",
    "FrameIndexStream": ".indexer",
    "SeedConfig": ".indexer",
    "RefineConfig": ".indexer",
    "CellMatchConfig": ".indexer",
    "IntegrateConfig": ".indexer",
    "Ambigator": ".ambigator",
    "AmbigatorResult": ".ambigator",
    "run_data_parallel": ".tools.multigpu",
    "run_block_from_env": ".tools.multigpu",
    "merge_streams": ".tools.multigpu",
    "BlockConfig": ".tools.multigpu",
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
