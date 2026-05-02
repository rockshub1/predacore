"""Shared protobuf stubs for PredaCore services.

Load generated modules lazily so one unrelated gRPC stub version check does not
block importing the specific proto package a caller actually needs.

Dependency ordering: protobuf descriptor pool entries must be loaded in
dependency order. ``egm.proto`` references ``csc.proto``, ``knowledge_nexus.proto``
references ``google/protobuf/empty.proto``, etc. We preload required deps via
``__getattr__`` so any caller order works.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = [
    "csc_pb2",
    "csc_pb2_grpc",
    "daf_pb2",
    "daf_pb2_grpc",
    "egm_pb2",
    "egm_pb2_grpc",
    "wil_pb2",
    "wil_pb2_grpc",
]

# In-package proto deps loaded relative to this package.
_PROTO_DEPS: dict[str, tuple[str, ...]] = {
    "egm_pb2": ("csc_pb2",),
    "egm_pb2_grpc": ("csc_pb2",),
}

# Standard-library proto deps loaded by absolute name (none after KN retirement).
_STDLIB_PROTO_DEPS: dict[str, tuple[str, ...]] = {}


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for dep in _STDLIB_PROTO_DEPS.get(name, ()):
        import_module(dep)
    for dep in _PROTO_DEPS.get(name, ()):
        if dep not in globals():
            globals()[dep] = import_module(f".{dep}", __name__)
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module

