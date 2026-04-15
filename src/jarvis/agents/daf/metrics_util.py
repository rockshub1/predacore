"""Shared Prometheus metric registration utility for DAF modules."""
from __future__ import annotations


def get_or_create_metric(metric_cls, name: str, *args, **kwargs):
    """Create a Prometheus metric, handling duplicate registration gracefully.

    If the metric is already registered (common in tests), unregisters the
    existing collector and re-creates it.
    """
    from prometheus_client import REGISTRY

    try:
        return metric_cls(name, *args, **kwargs)
    except ValueError:
        # Already registered — unregister and re-create
        try:
            collector = REGISTRY._names_to_collectors.get(name)
            if collector:
                REGISTRY.unregister(collector)
        except (KeyError, AttributeError, TypeError):
            for suffix in ["_total", "_created", ""]:
                try:
                    collector = REGISTRY._names_to_collectors.get(name + suffix)
                    if collector:
                        REGISTRY.unregister(collector)
                        break
                except (KeyError, AttributeError, TypeError):
                    continue
        return metric_cls(name, *args, **kwargs)
