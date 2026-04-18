"""
Prometheus Skill Marketplace - Foundation for shareable agent skills

Provides infrastructure for defining, packaging, and sharing agent skills.
Skills are modular capabilities that can be installed, configured, and invoked.
"""
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SkillCategory(Enum):
    """Categories for organizing skills."""

    PRODUCTIVITY = "productivity"
    COMMUNICATION = "communication"
    DATA_ANALYSIS = "data_analysis"
    WEB_AUTOMATION = "web_automation"
    CODE_GENERATION = "code_generation"
    CONTENT_CREATION = "content_creation"
    INTEGRATIONS = "integrations"
    UTILITIES = "utilities"
    CUSTOM = "custom"


@dataclass
class SkillParameter:
    """Defines a parameter for a skill."""

    name: str
    type: str  # string, number, boolean, array, object
    description: str = ""
    required: bool = False
    default: Any = None
    enum: list[Any] | None = None  # If provided, value must be one of these


@dataclass
class SkillDefinition:
    """Defines a skill that can be installed and invoked."""

    id: str  # Unique identifier like "predacore.web-scraper" or "community.email-summarizer"
    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    category: SkillCategory = SkillCategory.CUSTOM
    parameters: list[SkillParameter] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    icon: str = "🔧"  # Emoji or icon URL
    examples: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[str] = field(
        default_factory=list
    )  # Other skill IDs this depends on

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "category": self.category.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum,
                }
                for p in self.parameters
            ],
            "tags": self.tags,
            "icon": self.icon,
            "examples": self.examples,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillDefinition":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author", ""),
            category=SkillCategory(data.get("category", "custom")),
            parameters=[
                SkillParameter(
                    name=p["name"],
                    type=p["type"],
                    description=p.get("description", ""),
                    required=p.get("required", False),
                    default=p.get("default"),
                    enum=p.get("enum"),
                )
                for p in data.get("parameters", [])
            ],
            tags=data.get("tags", []),
            icon=data.get("icon", "🔧"),
            examples=data.get("examples", []),
            dependencies=data.get("dependencies", []),
        )


@dataclass
class InstalledSkill:
    """Represents an installed skill with configuration."""

    definition: SkillDefinition
    installed_at: datetime = field(default_factory=datetime.utcnow)
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)  # User-specific configuration
    invocation_count: int = 0
    last_invoked: datetime | None = None
    user_id: str | None = None


class SkillExecutor(ABC):
    """Base class for skill execution."""

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the skill with given parameters. Returns result dict."""
        pass


class SkillMarketplace:
    """
    Manages skill discovery, installation, and invocation.

    This is the foundation layer - actual skill implementations are registered
    as executors at runtime.
    """

    def __init__(self, data_path: str | None = None):
        self._data_path = Path(data_path) if data_path else Path("data/skills")
        self._data_path.mkdir(parents=True, exist_ok=True)

        # Available skills in the marketplace (could be fetched from remote)
        self._available: dict[str, SkillDefinition] = {}

        # Installed skills per user
        self._installed: dict[
            str, dict[str, InstalledSkill]
        ] = {}  # user_id -> {skill_id -> skill}

        # Registered executors
        self._executors: dict[str, SkillExecutor] = {}

        # Load persisted data
        self._load()

        # Register built-in skills
        self._register_builtin_skills()

        logger.info(
            f"SkillMarketplace initialized with {len(self._available)} available skills"
        )

    def _load(self):
        """Load persisted skills from disk."""
        installed_file = self._data_path / "installed_skills.json"
        if installed_file.exists():
            try:
                with installed_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    for user_id, skills in data.items():
                        self._installed[user_id] = {}
                        for skill_id, skill_data in skills.items():
                            definition = SkillDefinition.from_dict(
                                skill_data["definition"]
                            )
                            self._installed[user_id][skill_id] = InstalledSkill(
                                definition=definition,
                                enabled=skill_data.get("enabled", True),
                                config=skill_data.get("config", {}),
                                invocation_count=skill_data.get("invocation_count", 0),
                                user_id=user_id,
                            )
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Failed to load installed skills: {e}")

    def _save(self):
        """Persist installed skills to disk."""
        installed_file = self._data_path / "installed_skills.json"
        try:
            data = {}
            for user_id, skills in self._installed.items():
                data[user_id] = {}
                for skill_id, skill in skills.items():
                    data[user_id][skill_id] = {
                        "definition": skill.definition.to_dict(),
                        "enabled": skill.enabled,
                        "config": skill.config,
                        "invocation_count": skill.invocation_count,
                    }
            with installed_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save installed skills: {e}")

    def _register_builtin_skills(self):
        """Register built-in skills available out of the box."""
        builtins = [
            SkillDefinition(
                id="predacore.web-scraper",
                name="Web Scraper",
                description="Extract content from web pages including text, links, and structured data",
                category=SkillCategory.WEB_AUTOMATION,
                icon="🌐",
                parameters=[
                    SkillParameter("url", "string", "URL to scrape", required=True),
                    SkillParameter(
                        "selector", "string", "CSS selector to target specific elements"
                    ),
                    SkillParameter(
                        "format",
                        "string",
                        "Output format",
                        enum=["text", "markdown", "json"],
                    ),
                ],
                tags=["web", "scraping", "content"],
            ),
            SkillDefinition(
                id="predacore.code-executor",
                name="Code Executor",
                description="Execute code in a sandboxed environment",
                category=SkillCategory.CODE_GENERATION,
                icon="⚡",
                parameters=[
                    SkillParameter("code", "string", "Code to execute", required=True),
                    SkillParameter(
                        "language",
                        "string",
                        "Programming language",
                        required=True,
                        enum=["python", "javascript", "go", "rust"],
                    ),
                    SkillParameter(
                        "timeout", "number", "Timeout in seconds", default=30
                    ),
                ],
                tags=["code", "execution", "sandbox"],
            ),
            SkillDefinition(
                id="predacore.email-sender",
                name="Email Sender",
                description="Send emails with templating support",
                category=SkillCategory.COMMUNICATION,
                icon="📧",
                parameters=[
                    SkillParameter("to", "string", "Recipient email", required=True),
                    SkillParameter("subject", "string", "Email subject", required=True),
                    SkillParameter("body", "string", "Email body", required=True),
                    SkillParameter("html", "boolean", "Send as HTML", default=False),
                ],
                tags=["email", "communication"],
            ),
            SkillDefinition(
                id="predacore.file-manager",
                name="File Manager",
                description="Read, write, and manage files in the allowed workspace",
                category=SkillCategory.UTILITIES,
                icon="📁",
                parameters=[
                    SkillParameter(
                        "operation",
                        "string",
                        "Operation to perform",
                        required=True,
                        enum=["read", "write", "list", "delete"],
                    ),
                    SkillParameter(
                        "path", "string", "File or directory path", required=True
                    ),
                    SkillParameter("content", "string", "Content for write operations"),
                ],
                tags=["files", "storage"],
            ),
            SkillDefinition(
                id="predacore.daily-briefing",
                name="Daily Briefing Generator",
                description="Generate personalized daily briefings with calendar, tasks, and news",
                category=SkillCategory.PRODUCTIVITY,
                icon="🌅",
                parameters=[
                    SkillParameter(
                        "include_calendar",
                        "boolean",
                        "Include calendar events",
                        default=True,
                    ),
                    SkillParameter(
                        "include_tasks",
                        "boolean",
                        "Include pending tasks",
                        default=True,
                    ),
                    SkillParameter(
                        "include_news",
                        "boolean",
                        "Include relevant news",
                        default=False,
                    ),
                    SkillParameter(
                        "time_range_hours", "number", "Hours ahead to look", default=24
                    ),
                ],
                tags=["briefing", "productivity", "summary"],
            ),
            SkillDefinition(
                id="predacore.data-analyzer",
                name="Data Analyzer",
                description="Analyze data from CSV, JSON, or API responses",
                category=SkillCategory.DATA_ANALYSIS,
                icon="📊",
                parameters=[
                    SkillParameter(
                        "data",
                        "string",
                        "Data to analyze (JSON or file path)",
                        required=True,
                    ),
                    SkillParameter(
                        "analysis_type",
                        "string",
                        "Type of analysis",
                        enum=["summary", "trends", "anomalies", "correlations"],
                    ),
                    SkillParameter(
                        "output_format",
                        "string",
                        "Output format",
                        enum=["text", "json", "chart_data"],
                    ),
                ],
                tags=["data", "analysis", "statistics"],
            ),
        ]

        for skill in builtins:
            self._available[skill.id] = skill

    def register_executor(self, skill_id: str, executor: SkillExecutor):
        """Register an executor for a skill."""
        self._executors[skill_id] = executor
        logger.info(f"Registered executor for skill {skill_id}")

    def register_skill(
        self,
        definition: SkillDefinition,
        *,
        overwrite: bool = False,
    ) -> bool:
        """Register a skill definition in the available catalog."""
        if definition.id in self._available and not overwrite:
            return False
        self._available[definition.id] = definition
        return True

    def get_skill(self, skill_id: str) -> SkillDefinition | None:
        """Get a skill definition by ID from the available catalog."""
        return self._available.get(skill_id)

    def list_available(
        self,
        category: SkillCategory | None = None,
        search: str | None = None,
    ) -> list[SkillDefinition]:
        """List available skills, optionally filtered."""
        skills = list(self._available.values())

        if category:
            skills = [s for s in skills if s.category == category]

        if search:
            search = search.lower()
            skills = [
                s
                for s in skills
                if search in s.id.lower()
                or search in s.name.lower()
                or search in s.description.lower()
                or any(search in tag for tag in s.tags)
            ]

        return skills

    def install(
        self, skill_id: str, user_id: str, config: dict[str, Any] | None = None
    ) -> bool:
        """Install a skill for a user."""
        if skill_id not in self._available:
            logger.warning(f"Skill {skill_id} not found in marketplace")
            return False

        definition = self._available[skill_id]

        # Check dependencies
        for dep_id in definition.dependencies:
            if dep_id not in self._installed.get(user_id, {}):
                logger.warning(f"Missing dependency {dep_id} for skill {skill_id}")
                return False

        # Install
        if user_id not in self._installed:
            self._installed[user_id] = {}

        self._installed[user_id][skill_id] = InstalledSkill(
            definition=definition,
            config=config or {},
            user_id=user_id,
        )

        self._save()
        logger.info(f"Installed skill {skill_id} for user {user_id}")
        return True

    def uninstall(self, skill_id: str, user_id: str) -> bool:
        """Uninstall a skill for a user."""
        if user_id not in self._installed or skill_id not in self._installed[user_id]:
            return False

        del self._installed[user_id][skill_id]
        self._save()
        logger.info(f"Uninstalled skill {skill_id} for user {user_id}")
        return True

    def list_installed(self, user_id: str) -> list[InstalledSkill]:
        """List installed skills for a user."""
        return list(self._installed.get(user_id, {}).values())

    async def invoke(
        self,
        skill_id: str,
        user_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke an installed skill."""
        # Check if installed
        if user_id not in self._installed or skill_id not in self._installed[user_id]:
            return {
                "error": f"Skill {skill_id} not installed for user",
                "success": False,
            }

        installed = self._installed[user_id][skill_id]

        if not installed.enabled:
            return {"error": "Skill is disabled", "success": False}

        # Validate parameters
        definition = installed.definition
        for param in definition.parameters:
            if param.required and param.name not in params:
                return {
                    "error": f"Missing required parameter: {param.name}",
                    "success": False,
                }
            if param.name not in params and param.default is not None:
                params[param.name] = param.default
            if (
                param.enum
                and param.name in params
                and params[param.name] not in param.enum
            ):
                return {
                    "error": f"Invalid value for {param.name}: must be one of {param.enum}",
                    "success": False,
                }

        # Execute
        executor = self._executors.get(skill_id)
        if not executor:
            return {
                "error": f"No executor registered for skill {skill_id}",
                "success": False,
            }

        try:
            result = await executor.execute(params)

            # Update invocation stats
            installed.invocation_count += 1
            installed.last_invoked = datetime.utcnow()
            self._save()

            return {"result": result, "success": True}

        except Exception as e:
            logger.error(f"Skill {skill_id} execution failed: {e}", exc_info=True)
            return {"error": str(e), "success": False}
