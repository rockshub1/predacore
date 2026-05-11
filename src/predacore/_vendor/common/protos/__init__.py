"""Shared protobuf stubs for PredaCore services.

Load generated modules lazily so one unrelated gRPC stub version check does not
block importing the specific proto package a caller actually needs.

Dependency ordering: protobuf descriptor pool entries must be loaded in
dependency order. All four of our proto files reference
``google/protobuf/struct.proto``; ``daf.proto`` additionally references
``google/protobuf/timestamp.proto`` and ``wil.proto``; ``egm.proto``
references ``csc.proto``. We preload Google's struct + timestamp into the
descriptor pool eagerly at import time, then resolve our own in-package
deps lazily via ``__getattr__``.

B-001 fix (Wave 12, 2026-05-11): the previous version of this loader
didn't preload ``google.protobuf.struct_pb2`` / ``timestamp_pb2``, so a
test importer that touched ``daf_pb2`` first hit
``TypeError: Couldn't build proto file into descriptor pool: Depends on
file 'google/protobuf/struct.proto'``. Now they're imported at module
load so the descriptor pool is primed before any of ours land.
"""

from __future__ import annotations

import threading
from importlib import import_module
from types import ModuleType

# B-001 fix: eagerly import the well-known Google types so the descriptor
# pool has them registered before our generated stubs try to reference
# them via AddSerializedFile.
import google.protobuf.struct_pb2  # noqa: F401 — descriptor pool side-effect
import google.protobuf.timestamp_pb2  # noqa: F401 — descriptor pool side-effect

# L35 (Wave 12): serialize lazy imports across threads. Without this, two
# threads racing on first access to e.g. `daf_pb2` could each `import_module`
# the stub, and protobuf's descriptor pool raises
# `TypeError: Couldn't build proto file into descriptor pool: Conflict
# register for file "daf.proto": <FileDescriptor> already registered.`
# The lock makes lazy resolution a single-writer / many-reader pattern.
_LAZY_IMPORT_LOCK = threading.Lock()

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
# daf.proto depends on wil.proto (TaskAssignmentMessage references types).
# egm.proto depends on csc.proto.
_PROTO_DEPS: dict[str, tuple[str, ...]] = {
    "daf_pb2": ("wil_pb2",),
    "daf_pb2_grpc": ("wil_pb2",),
    "egm_pb2": ("csc_pb2",),
    "egm_pb2_grpc": ("csc_pb2",),
}

# Standard-library proto deps loaded by absolute name (eagerly imported
# above, but kept here for the per-module dependency declaration so
# future maintainers know which Google types each stub needs).
_STDLIB_PROTO_DEPS: dict[str, tuple[str, ...]] = {
    "csc_pb2": ("google.protobuf.struct_pb2",),
    "daf_pb2": ("google.protobuf.struct_pb2", "google.protobuf.timestamp_pb2"),
    "egm_pb2": ("google.protobuf.struct_pb2",),
    "wil_pb2": ("google.protobuf.struct_pb2",),
}


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Double-checked locking: fast path when another thread already did
    # the work, slow path under the lock for first arrival.
    cached = globals().get(name)
    if cached is not None:
        return cached
    with _LAZY_IMPORT_LOCK:
        cached = globals().get(name)
        if cached is not None:
            return cached
        for dep in _STDLIB_PROTO_DEPS.get(name, ()):
            import_module(dep)
        for dep in _PROTO_DEPS.get(name, ()):
            if dep not in globals():
                globals()[dep] = import_module(f".{dep}", __name__)
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module

