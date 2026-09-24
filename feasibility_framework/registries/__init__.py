"""Registries for readers, composite strategies, and inference methods."""

from __future__ import annotations

from .composites import (
    get_composite,
    register_composite,
    registered_composites,
    unregister_composite,
)
from .readers import get_reader, register_reader, registered_readers, unregister_reader


def get_inference_method(name: str):
    from .inference import get_inference_method as _get

    return _get(name)


def register_inference(name: str):
    from .inference import register_inference as _register

    return _register(name)


def registered_inference_methods() -> tuple[str, ...]:
    from .inference import registered_inference_methods as _registered

    return _registered()


def unregister_inference(name: str) -> None:
    from .inference import unregister_inference as _unregister

    _unregister(name)


def __getattr__(name: str):
    if name in {"ContrastSpec", "InferenceResult"}:
        from . import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ContrastSpec",
    "InferenceResult",
    "get_composite",
    "get_inference_method",
    "get_reader",
    "register_composite",
    "register_inference",
    "register_reader",
    "registered_composites",
    "registered_inference_methods",
    "registered_readers",
    "unregister_composite",
    "unregister_inference",
    "unregister_reader",
]
