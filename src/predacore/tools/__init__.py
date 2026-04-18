"""
PredaCore Tool System — Registry, Dispatcher, Handlers, and Pipeline.

Architecture:
  registry.py           — Tool definitions and metadata
  trust_policy.py       — Confirmation/block policy evaluation
  dispatcher.py         — Handler lookup, timeout, circuit breaker, cache, metrics
  executor.py           — Thin facade + marketplace wiring
  handlers/             — Domain-specific handler modules (12 files)
  resilience.py         — Circuit breaker, result cache, execution history
  subsystem_init.py     — SubsystemFactory for consistent init
  enums.py              — ToolName, ToolStatus, DesktopAction, AndroidAction
  pipeline.py           — Tool chaining and composition
"""
