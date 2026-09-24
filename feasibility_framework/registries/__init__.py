"""Import built-in plugins so registration runs on package import."""

from .composites import (
    get_composite,
    register_composite,
    registered_composites,
    unregister_composite,
)
from .inference import (
    ContrastSpec,
    InferenceResult,
    get_inference_method,
    register_inference,
    registered_inference_methods,
    unregister_inference,
)
from .readers import get_reader, register_reader, registered_readers, unregister_reader

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
