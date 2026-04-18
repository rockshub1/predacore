"""
Implementation of Agent Registries for DAF.
Includes static type definitions and dynamic instance management.
"""
import asyncio  # For locks
import datetime
import logging
from typing import Any, Optional

from predacore._vendor.common.protos import daf_pb2  # For AgentStatus enum
from google.protobuf.timestamp_pb2 import Timestamp

# --- Abstract Base Classes ---


class AbstractAgentTypeRegistry:
    """Interface for retrieving static agent type definitions."""

    def get_agent_type(self, agent_type_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def list_agent_types(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def find_agent_for_capability(self, capability: str) -> str | None:
        raise NotImplementedError


class AbstractAgentInstanceRegistry:
    """Interface for managing dynamic agent instances."""

    async def register_instance(
        self, instance_id: str, type_id: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def unregister_instance(self, instance_id: str) -> bool:
        raise NotImplementedError

    async def get_instance(self, instance_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def list_instances(self, type_id: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def update_instance_status(
        self, instance_id: str, status: Any, task_info: dict[str, Any] | None = None
    ) -> bool:
        raise NotImplementedError

    async def find_idle_instance(self, type_id: str) -> str | None:
        raise NotImplementedError


# --- Concrete Implementations ---


class StaticAgentTypeRegistry(AbstractAgentTypeRegistry):
    """
    A simple in-memory registry for static agent type definitions.
    (Formerly SimpleAgentRegistry)
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        # Hardcoded agent types
        self._agent_types: dict[str, dict[str, Any]] = {
            "web_searcher": {
                "agent_type_id": "web_searcher",
                "description": "Performs web searches.",
                "supported_actions": ["search_web", "find_information"],
                "required_tools": ["google_search_api"],
                "configuration": {},
            },
            "python_executor": {
                "agent_type_id": "python_executor",
                "description": "Executes Python code.",
                "supported_actions": ["execute_code", "run_script"],
                "required_tools": ["python_sandbox"],
                "configuration": {"default_timeout": 60},
            },
            "web_scraper": {
                "agent_type_id": "web_scraper",
                "description": "Fetches web content.",
                "supported_actions": ["scrape_url"],
                "required_tools": ["basic_web_scraper"],
                "configuration": {},
            },
            "weather_fetcher": {
                "agent_type_id": "weather_fetcher",
                "description": "Fetches weather data.",
                "supported_actions": ["get_weather"],
                "required_tools": ["openweathermap_api"],
                "configuration": {},
            },
            "wiki_summarizer": {
                "agent_type_id": "wiki_summarizer",
                "description": "Summarizes Wikipedia topics.",
                "supported_actions": ["summarize_topic"],
                "required_tools": ["wikipedia_api"],
                "configuration": {},
            },
            "advanced_scraper": {
                "agent_type_id": "advanced_scraper",
                "description": "Advanced web scraping.",
                "supported_actions": ["scrape_advanced"],
                "required_tools": ["advanced_web_scraper"],
                "configuration": {},
            },
            "python_lint_agent": {
                "agent_type_id": "python_lint_agent",
                "description": "Lints Python code.",
                "supported_actions": ["lint_code"],
                "required_tools": ["python_linter"],
                "configuration": {},
            },
            "python_formatter_agent": {
                "agent_type_id": "python_formatter_agent",
                "description": "Formats Python code.",
                "supported_actions": ["format_code"],
                "required_tools": ["python_formatter"],
                "configuration": {},
            },
            "translator_agent": {
                "agent_type_id": "translator_agent",
                "description": "Translates text.",
                "supported_actions": ["translate_text"],
                "required_tools": ["translation_api"],
                "configuration": {},
            },
            "doc_summarizer_agent": {
                "agent_type_id": "doc_summarizer_agent",
                "description": "Summarizes documents.",
                "supported_actions": ["summarize_document"],
                "required_tools": ["document_summarizer"],
                "configuration": {},
            },
            "email_agent": {
                "agent_type_id": "email_agent",
                "description": "Sends emails.",
                "supported_actions": ["send_email"],
                "required_tools": ["email_sender"],
                "configuration": {},
            },
            "slack_bot_agent": {
                "agent_type_id": "slack_bot_agent",
                "description": "Sends Slack messages.",
                "supported_actions": ["send_slack_message"],
                "required_tools": ["slack_bot"],
                "configuration": {},
            },
            "discord_bot_agent": {
                "agent_type_id": "discord_bot_agent",
                "description": "Sends Discord messages.",
                "supported_actions": ["send_discord_message"],
                "required_tools": ["discord_bot"],
                "configuration": {},
            },
            "pdf_parser_agent": {
                "agent_type_id": "pdf_parser_agent",
                "description": "Parses PDF files.",
                "supported_actions": ["parse_pdf"],
                "required_tools": ["pdf_parser"],
                "configuration": {},
            },
            "image_analysis_agent": {
                "agent_type_id": "image_analysis_agent",
                "description": "Analyzes images.",
                "supported_actions": ["analyze_image", "ocr", "object_detection"],
                "required_tools": ["image_analysis"],
                "configuration": {},
            },
            "calendar_agent": {
                "agent_type_id": "calendar_agent",
                "description": "Manages calendar events.",
                "supported_actions": ["create_event", "list_events", "delete_event"],
                "required_tools": ["calendar_integration"],
                "configuration": {},
            },
        }
        self.logger.info(
            f"StaticAgentTypeRegistry initialized with {len(self._agent_types)} agent types."
        )

    def get_agent_type(self, agent_type_id: str) -> dict[str, Any] | None:
        agent_type = self._agent_types.get(agent_type_id)
        if agent_type:
            self.logger.debug(f"Retrieved agent type definition for '{agent_type_id}'")
            return agent_type.copy()
        else:
            self.logger.warning(f"Agent type '{agent_type_id}' not found in registry.")
            return None

    def list_agent_types(self) -> list[dict[str, Any]]:
        self.logger.debug(f"Listing all {len(self._agent_types)} agent types.")
        return [agent_type.copy() for agent_type in self._agent_types.values()]

    def find_agent_for_capability(self, capability: str) -> str | None:
        self.logger.debug(f"Searching for agent supporting capability: '{capability}'")
        capability_lower = capability.lower()
        for agent_type_id, agent_info in self._agent_types.items():
            supported_actions = agent_info.get("supported_actions", [])
            if any(action.lower() == capability_lower for action in supported_actions):
                self.logger.debug(
                    f"Found agent type '{agent_type_id}' for capability '{capability}'"
                )
                return agent_type_id
        self.logger.warning(
            f"No agent type found supporting capability: '{capability}'"
        )
        return None


class ActiveAgentInstanceRegistry(AbstractAgentInstanceRegistry):
    """
    Simple in-memory registry for managing active agent instances.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self._active_instances: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.logger.info("ActiveAgentInstanceRegistry initialized.")

    async def register_instance(
        self, instance_id: str, type_id: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        async with self._lock:
            if instance_id in self._active_instances:
                self.logger.warning(
                    f"Attempted to register existing instance ID: {instance_id}"
                )
                # Or raise an error? For now, return existing.
                return self._active_instances[instance_id]

            now_ts = Timestamp()
            now_ts.FromDatetime(datetime.datetime.now(datetime.timezone.utc))
            instance_info = {
                "instance_id": instance_id,
                "type_id": type_id,
                "status": daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE,
                "last_heartbeat": now_ts,
                "config": config,
                "current_task_info": None,
            }
            self._active_instances[instance_id] = instance_info
            self.logger.info(
                f"Registered new agent instance: {instance_id} (type: {type_id})"
            )
            return instance_info

    async def unregister_instance(self, instance_id: str) -> bool:
        async with self._lock:
            if instance_id in self._active_instances:
                del self._active_instances[instance_id]
                self.logger.info(f"Unregistered agent instance: {instance_id}")
                return True
            else:
                self.logger.warning(
                    f"Attempted to unregister non-existent instance ID: {instance_id}"
                )
                return False

    async def get_instance(self, instance_id: str) -> dict[str, Any] | None:
        async with self._lock:
            instance = self._active_instances.get(instance_id)
            return instance.copy() if instance else None

    async def list_instances(self, type_id: str | None = None) -> list[dict[str, Any]]:
        async with self._lock:
            if type_id:
                return [
                    inst.copy()
                    for inst in self._active_instances.values()
                    if inst["type_id"] == type_id
                ]
            else:
                return [inst.copy() for inst in self._active_instances.values()]

    async def update_instance_status(
        self, instance_id: str, status: Any, task_info: dict[str, Any] | None = None
    ) -> bool:
        async with self._lock:
            instance = self._active_instances.get(instance_id)
            if instance:
                instance["status"] = status
                instance["current_task_info"] = task_info
                now_ts = Timestamp()
                now_ts.FromDatetime(datetime.datetime.now(datetime.timezone.utc))
                instance["last_heartbeat"] = now_ts
                self.logger.debug(
                    f"Updated status for instance {instance_id} to {status}"
                )
                return True
            else:
                self.logger.warning(
                    f"Attempted to update status for non-existent instance ID: {instance_id}"
                )
                return False

    async def find_idle_instance(self, type_id: str) -> str | None:
        async with self._lock:
            for instance_id, instance_info in self._active_instances.items():
                if (
                    instance_info["type_id"] == type_id
                    and instance_info["status"]
                    == daf_pb2.AgentInstanceMessage.AgentStatus.AGENT_STATUS_IDLE
                ):
                    self.logger.debug(
                        f"Found idle instance {instance_id} for type {type_id}"
                    )
                    return instance_id
            self.logger.debug(f"No idle instance found for type {type_id}")
            return None


# --- Backward Compatibility Aliases ---
# Tests and older code may reference these names
SimpleAgentRegistry = StaticAgentTypeRegistry
AbstractAgentRegistry = AbstractAgentTypeRegistry
