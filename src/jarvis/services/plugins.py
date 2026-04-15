"""
JARVIS Plugin SDK — extensible tool and hook system.

Provides a clean API for third-party extensions to:
  - Register new tools (commands the LLM can call)
  - Register hooks (lifecycle callbacks — on_message, on_response, etc.)
  - Declare metadata (name, version, author, description)
  - Define dependencies and configuration

Usage:
    from src.jarvis.plugins import Plugin, hook, tool

    class WeatherPlugin(Plugin):
        name = "weather"
        version = "1.0.0"
        description = "Weather lookups via OpenWeatherMap"

        @tool(description="Get current weather for a city")
        async def get_weather(self, city: str) -> str:
            ...

        @hook("on_message")
        async def log_message(self, message):
            ...

    # In JARVIS core:
    registry = PluginRegistry()
    registry.load_plugin(WeatherPlugin)
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Safety constants ───────────────────────────────────────────────

# Maximum time a single plugin tool call may run before being cancelled.
PLUGIN_TOOL_TIMEOUT: float = 30.0

# Maximum number of plugins that can be loaded simultaneously.
MAX_LOADED_PLUGINS: int = 50

# Modules that plugins are NOT allowed to import.  Checked at load time
# via a shallow audit of the plugin class source (not a full sandbox).
_RESTRICTED_MODULES: frozenset[str] = frozenset({
    "subprocess",
    "shutil",
    "ctypes",
    "multiprocessing",
})


# ── Decorators ──────────────────────────────────────────────────────


@dataclass
class ToolMeta:
    """Metadata attached to a tool method by the @tool decorator."""

    name: str
    description: str
    parameters: dict[str, Any]
    requires_confirmation: bool


@dataclass
class HookMeta:
    """Metadata attached to a hook method by the @hook decorator."""

    event: str
    priority: int  # Lower = runs first


def tool(
    description: str = "",
    parameters: dict[str, Any] | None = None,
    requires_confirmation: bool = False,
    name: str | None = None,
):
    """
    Decorator to mark a plugin method as a tool callable by the LLM.

    Args:
        description: Human-readable description for the LLM.
        parameters: JSON Schema for tool parameters. Auto-generated if None.
        requires_confirmation: Whether to ask user before executing.
        name: Override tool name (defaults to method name).
    """

    def decorator(func):
        # Auto-generate parameters from type hints if not provided
        params = parameters
        if params is None:
            sig = inspect.signature(func)
            props = {}
            required = []
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                prop = {"type": "string"}
                if param.annotation != inspect.Parameter.empty:
                    type_map = {
                        str: "string",
                        int: "integer",
                        float: "number",
                        bool: "boolean",
                        list: "array",
                        dict: "object",
                    }
                    prop["type"] = type_map.get(param.annotation, "string")
                props[pname] = prop
                if param.default == inspect.Parameter.empty:
                    required.append(pname)
            params = {
                "type": "object",
                "properties": props,
                "required": required,
            }

        func._tool_meta = ToolMeta(
            name=name or func.__name__,
            description=description or func.__doc__ or "",
            parameters=params,
            requires_confirmation=requires_confirmation,
        )
        return func

    return decorator


def hook(event: str, priority: int = 100):
    """
    Decorator to mark a plugin method as a lifecycle hook.

    Supported events:
        - on_message: Called when a message is received
        - on_response: Called before sending a response
        - on_tool_call: Called before executing a tool
        - on_tool_result: Called after tool execution
        - on_error: Called when an error occurs
        - on_startup: Called when JARVIS starts
        - on_shutdown: Called when JARVIS stops

    Args:
        event: Lifecycle event name.
        priority: Execution priority (lower = earlier). Default: 100.
    """

    def decorator(func):
        func._hook_meta = HookMeta(event=event, priority=priority)
        return func

    return decorator


# ── Valid Hook Events ───────────────────────────────────────────────

VALID_HOOK_EVENTS: set[str] = {
    "on_message",
    "on_response",
    "on_tool_call",
    "on_tool_result",
    "on_error",
    "on_startup",
    "on_shutdown",
}


# ── Plugin Base Class ──────────────────────────────────────────────


class Plugin(ABC):
    """
    Base class for JARVIS plugins.

    Subclass this and use @tool / @hook decorators to register
    capabilities. The PluginRegistry handles discovery and lifecycle.
    """

    name: str = "unnamed_plugin"
    version: str = "0.0.0"
    description: str = ""
    author: str = ""

    # Optional: config keys this plugin expects (override in subclasses)
    config_keys: list[str] = []

    def __init__(self, config: dict[str, Any] | None = None):
        # Prevent mutable class-level list from being shared across instances
        if "config_keys" not in type(self).__dict__:
            self.config_keys = []
        self.config = config or {}
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def get_tools(self) -> list[ToolMeta]:
        """Discover all @tool-decorated methods on this plugin."""
        tools = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if callable(attr) and hasattr(attr, "_tool_meta"):
                tools.append(attr._tool_meta)
        return tools

    def get_hooks(self) -> list[tuple]:
        """Discover all @hook-decorated methods, returns (event, priority, method)."""
        hooks = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if callable(attr) and hasattr(attr, "_hook_meta"):
                meta = attr._hook_meta
                hooks.append((meta.event, meta.priority, attr))
        return hooks

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Return tool definitions in the format expected by BUILTIN_TOOLS.

        This makes plugins seamlessly integrate with the existing tool system.
        """
        definitions = []
        for tool_meta in self.get_tools():
            definitions.append(
                {
                    "name": f"{self.name}.{tool_meta.name}",
                    "description": f"[Plugin: {self.name}] {tool_meta.description}",
                    "parameters": tool_meta.parameters,
                }
            )
        return definitions

    async def on_load(self) -> None:
        """Called when the plugin is loaded. Override for setup logic."""
        pass

    async def on_unload(self) -> None:
        """Called when the plugin is unloaded. Override for cleanup logic."""
        pass

    def __repr__(self) -> str:
        return f"<Plugin {self.name} v{self.version} enabled={self._enabled}>"


# ── Plugin Registry ────────────────────────────────────────────────


@dataclass
class RegisteredPlugin:
    """A plugin instance tracked by the registry."""

    plugin: Plugin
    tools: list[ToolMeta] = field(default_factory=list)
    hooks: list[tuple] = field(default_factory=list)


class PluginRegistry:
    """
    Central registry for all JARVIS plugins.

    Handles plugin lifecycle (load/unload), tool/hook discovery,
    and event dispatching.
    """

    def __init__(self):
        self._plugins: dict[str, RegisteredPlugin] = {}
        self._hooks: dict[str, list[tuple]] = {e: [] for e in VALID_HOOK_EVENTS}
        self._tool_handlers: dict[str, Callable] = {}

    @staticmethod
    def _validate_plugin_class(plugin_class: type[Plugin]) -> None:
        """Validate that a plugin class meets the required interface contract."""
        if not isinstance(plugin_class, type) or not issubclass(plugin_class, Plugin):
            raise TypeError(
                f"Expected a Plugin subclass, got {type(plugin_class).__name__}"
            )
        # Must define a meaningful name (not the default)
        if not hasattr(plugin_class, "name") or plugin_class.name == "unnamed_plugin":
            raise ValueError(
                f"Plugin class {plugin_class.__name__} must define a 'name' attribute"
            )
        # Shallow restricted-module audit on the class source
        try:
            source = inspect.getsource(plugin_class)
            for mod in _RESTRICTED_MODULES:
                if f"import {mod}" in source or f"from {mod}" in source:
                    raise ValueError(
                        f"Plugin {plugin_class.__name__} imports restricted module '{mod}'"
                    )
        except (OSError, TypeError):
            # Built-in or dynamically created classes may not have source
            logger.warning(
                "Cannot audit source for plugin %s — restricted-module check skipped",
                plugin_class.__name__,
            )

    async def load_plugin(
        self,
        plugin_class: type[Plugin],
        config: dict[str, Any] | None = None,
    ) -> Plugin:
        """
        Instantiate and register a plugin.

        Args:
            plugin_class: The Plugin subclass to instantiate.
            config: Optional configuration dict for the plugin.

        Returns:
            The instantiated plugin.

        Raises:
            ValueError: If a plugin with the same name is already loaded.
            TypeError: If plugin_class is not a valid Plugin subclass.
            RuntimeError: If the maximum number of plugins is reached.
        """
        # Enforce max loaded plugins limit
        if len(self._plugins) >= MAX_LOADED_PLUGINS:
            raise RuntimeError(
                f"Maximum plugin limit ({MAX_LOADED_PLUGINS}) reached. "
                "Unload a plugin before loading another."
            )

        # Validate interface before instantiation
        self._validate_plugin_class(plugin_class)

        try:
            plugin = plugin_class(config=config)
        except (TypeError, ValueError, OSError, ImportError) as exc:
            raise RuntimeError(
                f"Plugin {plugin_class.__name__} failed to instantiate: {exc}"
            ) from exc

        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' is already loaded")

        # Discover tools and hooks
        tools = plugin.get_tools()
        hooks = plugin.get_hooks()

        # Register tools
        for tool_meta in tools:
            full_name = f"{plugin.name}.{tool_meta.name}"
            # Find the actual method
            for attr_name in dir(plugin):
                attr = getattr(plugin, attr_name, None)
                if callable(attr) and hasattr(attr, "_tool_meta"):
                    if attr._tool_meta.name == tool_meta.name:
                        self._tool_handlers[full_name] = attr
                        break

        # Register hooks — reject unknown events
        for event, priority, method in hooks:
            if event not in VALID_HOOK_EVENTS:
                logger.warning(
                    "Plugin '%s' registers hook for unknown event '%s' — skipped",
                    plugin.name,
                    event,
                )
                continue
            self._hooks[event].append((priority, method))
            self._hooks[event].sort(key=lambda x: x[0])

        # Store registration
        self._plugins[plugin.name] = RegisteredPlugin(
            plugin=plugin, tools=tools, hooks=hooks
        )

        # Call on_load with error isolation
        try:
            await plugin.on_load()
        except Exception as exc:  # catch-all: top-level boundary
            # Roll back registration if on_load fails
            await self.unload_plugin(plugin.name)
            raise RuntimeError(
                f"Plugin '{plugin.name}' on_load() failed: {exc}"
            ) from exc

        logger.info(
            "Loaded plugin '%s' v%s (%d tools, %d hooks)",
            plugin.name,
            plugin.version,
            len(tools),
            len(hooks),
        )
        return plugin

    async def unload_plugin(self, name: str) -> None:
        """Unload a plugin by name."""
        reg = self._plugins.get(name)
        if not reg:
            raise ValueError(f"Plugin '{name}' is not loaded")

        # Remove tool handlers
        for tool_meta in reg.tools:
            full_name = f"{name}.{tool_meta.name}"
            self._tool_handlers.pop(full_name, None)

        # Remove hooks
        for event, _priority, method in reg.hooks:
            if event in self._hooks:
                self._hooks[event] = [
                    (p, m) for p, m in self._hooks[event] if m is not method
                ]

        # Call on_unload — errors must not prevent cleanup
        try:
            await reg.plugin.on_unload()
        except Exception as exc:  # catch-all: top-level boundary
            logger.error("Plugin '%s' on_unload() error (ignored): %s", name, exc)

        del self._plugins[name]
        logger.info("Unloaded plugin '%s'", name)

    async def dispatch_hook(self, event: str, **kwargs) -> list[Any]:
        """
        Dispatch a lifecycle hook to all registered handlers.

        Returns a list of results from each handler.
        """
        results = []
        if event not in self._hooks:
            return results

        for _priority, handler in self._hooks[event]:
            try:
                result = await handler(**kwargs)
                results.append(result)
            except Exception as e:  # catch-all: top-level boundary
                logger.error("Hook handler error (%s): %s", event, e)
                results.append(None)

        return results

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """
        Call a plugin tool by its full name (plugin.tool_name).

        Returns the tool output as a string.
        Enforces PLUGIN_TOOL_TIMEOUT to prevent runaway plugins.
        """
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            return f"[Plugin tool '{tool_name}' not found]"

        # Check that the owning plugin is enabled
        plugin_name = tool_name.split(".")[0] if "." in tool_name else None
        if plugin_name and plugin_name in self._plugins:
            if not self._plugins[plugin_name].plugin.enabled:
                return f"[Plugin '{plugin_name}' is disabled]"

        try:
            coro = (
                handler(**args)
                if asyncio.iscoroutinefunction(handler)
                else asyncio.to_thread(handler, **args)
            )
            result = await asyncio.wait_for(coro, timeout=PLUGIN_TOOL_TIMEOUT)
            return str(result)
        except asyncio.TimeoutError:
            logger.error(
                "Plugin tool '%s' timed out after %.0fs", tool_name, PLUGIN_TOOL_TIMEOUT
            )
            return f"[Plugin tool '{tool_name}' timed out after {PLUGIN_TOOL_TIMEOUT}s]"
        except Exception as e:  # catch-all: top-level boundary
            logger.error("Plugin tool '%s' error: %s", tool_name, e)
            return f"[Plugin tool error: {e}]"

    def get_all_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions from all loaded plugins (for LLM context)."""
        definitions = []
        for reg in self._plugins.values():
            if reg.plugin.enabled:
                definitions.extend(reg.plugin.get_tool_definitions())
        return definitions

    def get_loaded_plugins(self) -> list[dict[str, Any]]:
        """List all loaded plugins with metadata."""
        return [
            {
                "name": reg.plugin.name,
                "version": reg.plugin.version,
                "description": reg.plugin.description,
                "enabled": reg.plugin.enabled,
                "tools": len(reg.tools),
                "hooks": len(reg.hooks),
            }
            for reg in self._plugins.values()
        ]

    @property
    def plugin_count(self) -> int:
        return len(self._plugins)
