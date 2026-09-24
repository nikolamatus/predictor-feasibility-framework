"""Reader registry. Built-ins: csv, parquet. Users register other formats in-process."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    Reader = Callable[[Path], pd.DataFrame]
else:
    Reader = Callable[[Path], Any]

_READERS: dict[str, Reader] = {}


class RegistryError(KeyError):
    """Raised when a plugin name is missing or already taken by a different object."""


def register_reader(name: str) -> Callable[[Reader], Reader]:
    key = name.strip().lower()
    if not key:
        raise RegistryError("Reader name is empty.")

    def decorator(fn: Reader) -> Reader:
        existing = _READERS.get(key)
        if existing is not None and existing is not fn:
            raise RegistryError(f"Reader {key!r} is already registered.")
        _READERS[key] = fn
        return fn

    return decorator


def get_reader(name: str) -> Reader:
    key = name.strip().lower()
    try:
        return _READERS[key]
    except KeyError as exc:
        known = ", ".join(sorted(_READERS)) or "(none)"
        raise RegistryError(f"No reader registered as {key!r}. Known: {known}.") from exc


def registered_readers() -> tuple[str, ...]:
    return tuple(sorted(_READERS))


def unregister_reader(name: str) -> None:
    _READERS.pop(name.strip().lower(), None)


@register_reader("csv")
def read_csv(path: Path) -> pd.DataFrame:
    import pandas as pd

    return pd.read_csv(path)


@register_reader("parquet")
def read_parquet(path: Path) -> pd.DataFrame:
    import pandas as pd

    return pd.read_parquet(path)
