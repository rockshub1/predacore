"""Shared protobuf stubs for Project Prometheus services.

Load generated modules lazily so one unrelated gRPC stub version check does not
block importing the specific proto package a caller actually needs.
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
    "knowledge_nexus_pb2",
    "knowledge_nexus_pb2_grpc",
    "wil_pb2",
    "wil_pb2_grpc",
]


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module

