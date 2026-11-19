from __future__ import annotations

from typing import Optional

__all__ = ["SCHEMA_VERSION", "TORCHSX_VERSION", "torchsx_mismatch", "version_tuple"]

# Layout of the DuckDB output; 2 adds the rank columns and integration.bg_model_var_*
SCHEMA_VERSION = 2
# First TorchSX release whose reader understands SCHEMA_VERSION
TORCHSX_VERSION = "0.3.0"


# Leading numeric fields of a version string: "0.10.3.dev1" -> (0, 10, 3)
def version_tuple(version: Optional[str]) -> tuple:
    out = []
    for field in str(version or "").split("."):
        digits = ""
        for ch in field:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


# Why the importable TorchSX cannot read this schema; None when it can or is absent
def torchsx_mismatch() -> Optional[str]:
    try:
        import torchsx
    except Exception:  # noqa: BLE001 - TorchSX is optional
        return None
    try:
        from torchsx._compat import SUPPORTED_DB_SCHEMA as supported
    except Exception:  # noqa: BLE001 - TorchSX before the pairing reads schema 1
        supported = 1
    if int(supported) >= SCHEMA_VERSION:
        return None
    found = getattr(torchsx, "__version__", "of unknown version")
    return (
        f"the installed TorchSX {found} reads probixi database schema {int(supported)} and "
        f"older; this probixi writes schema {SCHEMA_VERSION}. TorchSX {TORCHSX_VERSION} "
        "or later reads it."
    )
