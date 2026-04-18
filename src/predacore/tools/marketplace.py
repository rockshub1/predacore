"""
PredaCore Marketplace Integration — OpenClaw skill import + executor wiring.

Extracted from executor.py during Phase 6.3 refactoring. Handles:
  - Resolving OpenClaw skills directory from env/config/discovery
  - Parsing SKILL.md frontmatter into SkillDefinition catalog entries
  - Building async skill executors (describe, search, run_script)
  - Wiring built-in Prometheus skills (web-scraper, code-executor, etc.)

Usage:
    marketplace = MarketplaceManager(config, tool_ctx)
    marketplace.initialize()   # registers all skills
    marketplace.skill_marketplace  # → SkillMarketplace instance
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import PredaCoreConfig
    from .handlers._context import ToolContext

logger = logging.getLogger(__name__)

# Lazy imports for marketplace (only needed if marketplace enabled)
try:
    from predacore.tools.skill_marketplace import (
        SkillCategory,
        SkillDefinition,
        SkillExecutor,
        SkillMarketplace,
        SkillParameter,
    )
except ImportError:  # pragma: no cover
    from src.predacore.tools.skill_marketplace import (  # type: ignore
        SkillCategory,
        SkillDefinition,
        SkillExecutor,
        SkillMarketplace,
        SkillParameter,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class _CallableSkillExecutor(SkillExecutor):
    """Adapter to execute marketplace skills via async callbacks."""

    def __init__(self, func: Callable[[dict[str, Any]], Any]) -> None:
        self._func = func

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool by name with the given arguments."""
        result = await self._func(params)
        return {"output": result}


# ---------------------------------------------------------------------------
# MarketplaceManager
# ---------------------------------------------------------------------------


class MarketplaceManager:
    """Manages the skill marketplace lifecycle — init, OpenClaw import, executor wiring.

    Extracted from ToolExecutor to keep the executor as a thin facade.
    """

    def __init__(self, config: "PredaCoreConfig", tool_ctx: "ToolContext") -> None:
        self._config = config
        self._tool_ctx = tool_ctx
        self._skill_marketplace: SkillMarketplace | None = None
        self._openclaw_marketplace_skills: dict[str, dict[str, Any]] = {}
        self._openclaw_skills_dir: Path | None = None

    # -- Public API --------------------------------------------------------

    @property
    def skill_marketplace(self) -> SkillMarketplace | None:
        """Return the underlying SkillMarketplace instance (or None)."""
        return self._skill_marketplace

    @property
    def openclaw_skills(self) -> dict[str, dict[str, Any]]:
        """Return the imported OpenClaw skill metadata."""
        return self._openclaw_marketplace_skills

    def initialize(self) -> None:
        """Initialize the marketplace, register executors, import OpenClaw skills."""
        try:
            marketplace_path = str(
                Path(self._config.skills_dir) / "marketplace"
            )
            self._skill_marketplace = SkillMarketplace(data_path=marketplace_path)
            self._tool_ctx.skill_marketplace = self._skill_marketplace

            self._register_builtin_executors()
            imported_openclaw = self._register_openclaw_skills()

            default_user = str(os.getenv("USER") or "default")
            for skill in self._skill_marketplace.list_available():
                self._skill_marketplace.install(skill.id, default_user)

            logger.info(
                "Skill marketplace enabled (%d skills, imported_openclaw=%d, user=%s)",
                len(self._skill_marketplace.list_available()),
                imported_openclaw,
                default_user,
            )
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning("Skill marketplace initialization failed: %s", exc)

    # -- OpenClaw directory resolution ------------------------------------

    def _resolve_openclaw_skills_dir(self) -> Path | None:
        """Resolve a readable OpenClaw skills directory, if available."""
        configured = [
            str(os.getenv("PREDACORE_OPENCLAW_SKILLS_DIR") or "").strip(),
            str(os.getenv("OPENCLAW_SKILLS_DIR") or "").strip(),
            str(getattr(self._config.openclaw, "skills_dir", "") or "").strip(),
        ]
        repo_root = Path(__file__).resolve().parents[3]
        discovered = self._discover_installed_openclaw_skills_dirs()

        candidates: list[Path] = []
        for raw in configured:
            if not raw:
                continue
            p = Path(raw).expanduser()
            candidates.append(p)
            if not p.is_absolute():
                candidates.append(Path.cwd() / p)
                candidates.append(repo_root / p)
        candidates.extend(discovered)

        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()
        return None

    def _discover_installed_openclaw_skills_dirs(self) -> list[Path]:
        """Discover skills dirs from a normal installed `openclaw` CLI, if present."""
        executable = shutil.which("openclaw")
        if not executable:
            return []

        candidates: list[Path] = []
        try:
            resolved = Path(executable).expanduser().resolve()
        except OSError:
            resolved = Path(executable).expanduser()

        for root in (
            resolved.parent,
            resolved.parent.parent,
        ):
            skills_dir = root / "skills"
            if skills_dir.exists() and skills_dir.is_dir():
                candidates.append(skills_dir)

        seen: set[str] = set()
        deduped: list[Path] = []
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    # -- OpenClaw skill executor factory ----------------------------------

    def _make_openclaw_skill_executor(
        self, skill_id: str
    ) -> Callable[[dict[str, Any]], Any]:
        """Create an async executor for a single OpenClaw skill."""

        async def _execute(params: dict[str, Any]) -> dict[str, Any]:
            info = self._openclaw_marketplace_skills.get(skill_id)
            if info is None:
                return {
                    "success": False,
                    "error": f"Unknown OpenClaw skill: {skill_id}",
                }

            action = str(params.get("action") or "describe").strip().lower()
            skill_dir = Path(info["skill_dir"]).resolve()
            scripts = list(info.get("scripts", []))

            if action in {"describe", "search"}:
                query = str(params.get("query") or "").strip().lower()
                matches: list[str] = []
                if query:
                    for line in str(info.get("body", "")).splitlines():
                        candidate = line.strip()
                        if candidate and query in candidate.lower():
                            matches.append(candidate)
                            if len(matches) >= 20:
                                break
                return {
                    "success": True,
                    "skill_id": skill_id,
                    "name": info.get("name"),
                    "description": info.get("description"),
                    "skill_file": info.get("skill_file"),
                    "scripts": scripts,
                    "commands": list(info.get("commands", [])),
                    "preview": info.get("preview", ""),
                    "query": query or None,
                    "matches": matches,
                }

            if action == "run_script":
                return await self._run_skill_script(
                    skill_id, skill_dir, scripts, params
                )

            return {
                "success": False,
                "error": "unsupported action",
                "allowed_actions": ["describe", "search", "run_script"],
            }

        return _execute

    async def _run_skill_script(
        self,
        skill_id: str,
        skill_dir: Path,
        scripts: list[str],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a script from an OpenClaw skill."""
        script_rel = str(params.get("script") or "").strip()
        if not script_rel:
            if len(scripts) == 1:
                script_rel = scripts[0]
            else:
                return {
                    "success": False,
                    "error": "script is required (multiple scripts available)",
                    "available_scripts": scripts,
                }

        script_path = (skill_dir / script_rel).resolve()
        try:
            script_path.relative_to(skill_dir)
        except ValueError:
            return {"success": False, "error": "script path escapes skill directory"}

        if not script_path.exists() or not script_path.is_file():
            return {
                "success": False,
                "error": f"script not found: {script_rel}",
                "available_scripts": scripts,
            }

        raw_args = params.get("args", [])
        if isinstance(raw_args, str):
            raw_args = [raw_args]
        if not isinstance(raw_args, list):
            return {"success": False, "error": "args must be a string array"}
        arg_tokens = [shlex.quote(str(item)) for item in raw_args]

        timeout = params.get("timeout_seconds", 120)
        try:
            timeout_seconds = max(1, min(int(timeout), 3600))
        except (ValueError, TypeError):
            timeout_seconds = 120

        ext = script_path.suffix.lower()
        if ext == ".py":
            command = f"python3 {shlex.quote(str(script_path))}"
        elif ext in {".sh", ".bash"}:
            command = f"bash {shlex.quote(str(script_path))}"
        else:
            command = shlex.quote(str(script_path))
        if arg_tokens:
            command = f"{command} {' '.join(arg_tokens)}"

        from .handlers import handle_run_command as _handle_run_command

        output = await _handle_run_command(
            {
                "command": command,
                "cwd": str(skill_dir),
                "timeout_seconds": timeout_seconds,
            },
            self._tool_ctx,
        )
        return {
            "success": True,
            "skill_id": skill_id,
            "script": script_rel,
            "cwd": str(skill_dir),
            "command": command,
            "output": output,
        }

    # -- OpenClaw skill import --------------------------------------------

    def _register_openclaw_skills(self) -> int:
        """Import OpenClaw SKILL.md entries into the marketplace catalog."""
        if self._skill_marketplace is None:
            return 0
        if not bool(getattr(self._config.openclaw, "auto_import_skills", True)):
            return 0

        from ..prompts import (
            _extract_openclaw_command_samples,
            _normalize_openclaw_skill_slug,
            _parse_openclaw_skill_document,
            _summarize_openclaw_markdown,
        )

        skills_root = self._resolve_openclaw_skills_dir()
        if skills_root is None:
            return 0

        imported = 0
        self._openclaw_skills_dir = skills_root
        for skill_dir in sorted(skills_root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists() or not skill_file.is_file():
                continue

            try:
                frontmatter, body = _parse_openclaw_skill_document(skill_file)
            except (OSError, ValueError, KeyError) as exc:
                logger.warning(
                    "Failed to parse OpenClaw skill doc %s: %s", skill_file, exc
                )
                continue

            name_raw = str(frontmatter.get("name") or skill_dir.name).strip()
            slug = _normalize_openclaw_skill_slug(skill_dir.name)
            skill_id = f"openclaw.{slug}"
            description = str(frontmatter.get("description") or "").strip()
            if not description:
                description = f"Imported OpenClaw skill '{name_raw}'."

            metadata = frontmatter.get("metadata")
            openclaw_meta: dict[str, Any] = {}
            if isinstance(metadata, dict):
                candidate = metadata.get("openclaw")
                if isinstance(candidate, dict):
                    openclaw_meta = candidate

            emoji = str(openclaw_meta.get("emoji") or "\U0001f9e9")
            required_bins: list[str] = []
            requires = openclaw_meta.get("requires")
            if isinstance(requires, dict):
                bins = requires.get("bins")
                if isinstance(bins, list):
                    required_bins = [
                        str(item).strip() for item in bins if str(item).strip()
                    ]

            scripts: list[str] = []
            scripts_dir = skill_dir / "scripts"
            if scripts_dir.exists() and scripts_dir.is_dir():
                for file_path in sorted(scripts_dir.rglob("*")):
                    if file_path.is_file():
                        scripts.append(str(file_path.relative_to(skill_dir)))

            commands = _extract_openclaw_command_samples(body)
            preview = _summarize_openclaw_markdown(body)

            definition = SkillDefinition(
                id=skill_id,
                name=f"OpenClaw: {name_raw}",
                description=description,
                author="OpenClaw",
                category=SkillCategory.INTEGRATIONS,
                parameters=[
                    SkillParameter(
                        name="action",
                        type="string",
                        description="describe | search | run_script",
                        default="describe",
                        enum=["describe", "search", "run_script"],
                    ),
                    SkillParameter(
                        name="query",
                        type="string",
                        description="Optional text filter for action=search",
                    ),
                    SkillParameter(
                        name="script",
                        type="string",
                        description="Relative script path (required when action=run_script and multiple scripts exist)",
                    ),
                    SkillParameter(
                        name="args",
                        type="array",
                        description="Optional script args for action=run_script",
                        default=[],
                    ),
                    SkillParameter(
                        name="timeout_seconds",
                        type="number",
                        description="Optional timeout for action=run_script",
                        default=120,
                    ),
                ],
                tags=sorted(
                    {
                        "openclaw",
                        "external",
                        "skill",
                        *[f"bin:{item}" for item in required_bins[:8]],
                    }
                ),
                icon=emoji,
                examples=[
                    {"action": "describe"},
                    {"action": "search", "query": "setup"},
                ],
            )

            self._skill_marketplace.register_skill(definition, overwrite=True)
            self._openclaw_marketplace_skills[skill_id] = {
                "skill_id": skill_id,
                "name": name_raw,
                "description": description,
                "skill_dir": str(skill_dir),
                "skill_file": str(skill_file),
                "scripts": scripts,
                "commands": commands,
                "preview": preview,
                "body": body,
            }
            self._skill_marketplace.register_executor(
                skill_id,
                _CallableSkillExecutor(
                    self._make_openclaw_skill_executor(skill_id)
                ),
            )
            imported += 1

        if imported:
            logger.info(
                "Imported %d OpenClaw skills from %s into marketplace",
                imported,
                skills_root,
            )
        return imported

    # -- Built-in skill executors -----------------------------------------

    def _register_builtin_executors(self) -> None:
        """Bind marketplace skill IDs to concrete runtime executors."""
        if self._skill_marketplace is None:
            return

        from .handlers import (
            handle_execute_code as _h_execute_code,
            handle_list_directory as _h_list_directory,
            handle_read_file as _h_read_file,
            handle_web_scrape as _h_web_scrape,
            handle_write_file as _h_write_file,
        )

        ctx = self._tool_ctx

        async def _exec_web_scraper(params: dict[str, Any]) -> dict[str, Any]:
            """Execute web scraping skill."""
            url = str(params.get("url", "")).strip()
            if not url:
                return {"success": False, "error": "Missing required parameter: url"}
            content = await _h_web_scrape({"url": url}, ctx)
            return {"success": True, "url": url, "content": content}

        async def _exec_code(params: dict[str, Any]) -> dict[str, Any]:
            """Execute code execution skill."""
            language = str(params.get("language") or "python").strip().lower()
            runtime_map = {"javascript": "node", "js": "node", "ts": "typescript"}
            runtime = runtime_map.get(language, language)
            timeout = int(params.get("timeout") or 30)
            result = await _h_execute_code(
                {
                    "code": str(params.get("code") or ""),
                    "runtime": runtime,
                    "timeout": timeout,
                    "network_allowed": bool(params.get("network_allowed", False)),
                },
                ctx,
            )
            return {"success": True, "runtime": runtime, "result": result}

        async def _exec_file_manager(params: dict[str, Any]) -> dict[str, Any]:
            """Execute file manager skill."""
            operation = str(params.get("operation") or "").strip().lower()
            path = str(params.get("path") or "").strip()
            if not operation or not path:
                return {"success": False, "error": "operation and path are required"}

            if operation == "read":
                result = await _h_read_file({"path": path}, ctx)
                return {"success": True, "result": result}
            if operation == "write":
                result = await _h_write_file(
                    {"path": path, "content": str(params.get("content") or "")}, ctx
                )
                return {"success": True, "result": result}
            if operation == "list":
                result = await _h_list_directory(
                    {"path": path, "recursive": bool(params.get("recursive", False))},
                    ctx,
                )
                return {"success": True, "result": result}
            if operation == "delete":
                return self._safe_delete(path)
            return {"success": False, "error": f"unsupported operation: {operation}"}

        async def _exec_email_sender(params: dict[str, Any]) -> dict[str, Any]:
            """Execute email sender skill (queues to outbox)."""
            to = str(params.get("to") or "").strip()
            subject = str(params.get("subject") or "").strip()
            body = str(params.get("body") or "")
            if not (to and subject):
                return {"success": False, "error": "to and subject are required"}
            outbox = Path(self._config.logs_dir) / "marketplace_outbox"
            outbox.mkdir(parents=True, exist_ok=True)
            msg_id = f"email_{int(time.time() * 1000)}"
            path = outbox / f"{msg_id}.json"
            path.write_text(
                json.dumps(
                    {"to": to, "subject": subject, "body": body},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return {"success": True, "message_id": msg_id, "saved_to": str(path)}

        async def _exec_daily_briefing(params: dict[str, Any]) -> dict[str, Any]:
            """Execute daily briefing skill."""
            include_tasks = bool(params.get("include_tasks", True))
            include_calendar = bool(params.get("include_calendar", True))
            include_news = bool(params.get("include_news", False))
            memory_count = 0
            memory_svc = getattr(ctx, "memory_service", None)
            if memory_svc is not None:
                try:
                    memory_count = int(memory_svc._count_all())
                except (ValueError, TypeError):
                    memory_count = 0
            sections = [
                f"Memory items: {memory_count}",
                f"Session cache items: {len(ctx.memory or {})}",
            ]
            if include_tasks:
                sections.append(
                    "Tasks: use run_command/list_directory to inspect queues."
                )
            if include_calendar:
                sections.append(
                    "Calendar: integrate calendar_integration tool to enable live sync."
                )
            if include_news:
                sections.append("News: use web_search for fresh external updates.")
            return {"success": True, "briefing": "\n".join(sections)}

        async def _exec_data_analyzer(params: dict[str, Any]) -> dict[str, Any]:
            """Execute data analyzer skill."""
            raw_data = params.get("data")
            if raw_data is None:
                return {"success": False, "error": "data is required"}
            text = str(raw_data)
            source_path = Path(text).expanduser().resolve()
            # Only read files from the current working directory or home
            _safe_roots = [Path.cwd(), Path.home()]
            if (
                source_path.exists()
                and source_path.is_file()
                and any(str(source_path).startswith(str(r)) for r in _safe_roots)
                and source_path.stat().st_size < 10_000_000  # 10MB cap
            ):
                text = source_path.read_text(errors="replace")

            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return {"success": True, "format": "json", "rows": len(parsed)}
                if isinstance(parsed, dict):
                    return {
                        "success": True,
                        "format": "json",
                        "keys": sorted(parsed.keys())[:100],
                    }
            except (json.JSONDecodeError, ValueError):
                pass

            try:
                import csv
                import io
                reader = csv.DictReader(io.StringIO(text))
                rows = list(reader)
                return {
                    "success": True,
                    "format": "csv",
                    "rows": len(rows),
                    "columns": reader.fieldnames or [],
                }
            except (ValueError, OSError, KeyError):
                pass

            return {
                "success": True,
                "format": "text",
                "length": len(text),
                "preview": text[:500],
            }

        self._skill_marketplace.register_executor(
            "predacore.web-scraper", _CallableSkillExecutor(_exec_web_scraper)
        )
        self._skill_marketplace.register_executor(
            "predacore.code-executor", _CallableSkillExecutor(_exec_code)
        )
        self._skill_marketplace.register_executor(
            "predacore.file-manager", _CallableSkillExecutor(_exec_file_manager)
        )
        self._skill_marketplace.register_executor(
            "predacore.email-sender", _CallableSkillExecutor(_exec_email_sender)
        )
        self._skill_marketplace.register_executor(
            "predacore.daily-briefing", _CallableSkillExecutor(_exec_daily_briefing)
        )
        self._skill_marketplace.register_executor(
            "predacore.data-analyzer", _CallableSkillExecutor(_exec_data_analyzer)
        )

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _safe_delete(path: str) -> dict[str, Any]:
        """Safely delete a file/directory with protection against dangerous paths."""
        target = Path(path).expanduser().resolve()
        if not target.exists():
            return {"success": False, "error": f"path not found: {target}"}

        if ".." in path:
            return {"success": False, "error": "Blocked: path traversal ('..') is not allowed"}

        dangerous_prefixes = [
            "/etc", "/var", "/sys", "/proc", "/dev", "/boot",
            "/sbin", "/usr/bin", "/usr/lib",
            "/.ssh", "/.gnupg", "/.gitconfig", "/.aws", "/.kube",
            "/Library", "/System", "/.config", "/.env",
        ]
        target_str = str(target)
        home = str(Path.home())
        for dp in dangerous_prefixes:
            if target_str.startswith(dp) or target_str.startswith(home + dp):
                return {
                    "success": False,
                    "error": f"Blocked: cannot delete sensitive path: {target_str}",
                }

        if target.is_dir():
            import shutil
            shutil.rmtree(target)
        else:
            target.unlink()
        return {"success": True, "result": f"deleted: {target}"}
