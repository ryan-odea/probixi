from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from functools import lru_cache
from importlib import import_module

__all__ = ["Engine", "use_engine"]


class Engine(Enum):
    """Backend policy for the optional Triton and nvCOMP kernels.

    ``AUTO`` uses an accelerated kernel wherever the inputs support one,
    ``REFERENCE`` uses the torch implementation everywhere, and ``ACCELERATED``
    raises ``RuntimeError`` where no accelerated kernel applies.
    """

    AUTO = "auto"
    REFERENCE = "reference"
    ACCELERATED = "accelerated"


_engine = ContextVar("probixi_engine", default=Engine.AUTO)


@contextmanager
def use_engine(engine):
    """Select a backend for the enclosed block or decorated function.

    Parameters
    ----------
    engine : Engine or {'auto', 'reference', 'accelerated'}
        Backend policy. The previous policy is restored on exit, including exit
        by an exception.

    Notes
    -----
    The policy is read when a kernel is dispatched, so a lazy frame iterator or
    stream must also be consumed inside the block.
    :meth:`probixi.Probixi.calibrate` always runs under ``REFERENCE``.
    """
    token = _engine.set(Engine(engine))
    try:
        yield
    finally:
        _engine.reset(token)


@lru_cache(None)
def _module(name):
    # None when Triton or another dependency of the kernel module is missing
    try:
        return import_module(f"{__name__}.{name}")
    except ImportError:
        return None


def select(name, supported):
    # kernel module, or None for the torch reference; ACCELERATED raises instead
    engine = _engine.get()
    if engine is Engine.REFERENCE:
        return None
    module = _module(name) if supported else None
    if module is None and engine is Engine.ACCELERATED:
        raise RuntimeError(
            f"{name}: accelerated backend unavailable or unsupported inputs"
        )
    return module


def tensor_key(t):
    # valid while the caller holds t; a freed tensor's id() can be reused
    version = object() if t.is_inference() else t._version  # inference: never matches
    return (id(t), version, t.device, t.dtype, tuple(t.shape))
