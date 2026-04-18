"""
PredaCore Subsystem Factory — Single source of truth for initializing all subsystems.

ToolExecutor and other consumers need the same subsystems
(desktop, memory, MCTS, voice, OpenClaw, LLM). This factory creates them
once, consistently, eliminating the duplicate init code.

Usage:
    from .subsystem_init import SubsystemFactory, SubsystemBundle
    bundle = SubsystemFactory.create_all(config)
    # bundle.desktop_operator, bundle.unified_memory, etc.
"""
from __future__ import annotations

import copy
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# CLI-based providers that can't be used recursively from within sub-agents
# (nested CLI processes deadlock waiting for stdin).
_CLI_PROVIDERS = {"gemini-cli"}


@dataclass
class SubsystemBundle:
    """Container for all initialized subsystems.

    Every field starts as None and gets populated by SubsystemFactory.
    Subsystems that fail to initialize remain None — callers should
    handle gracefully.
    """
    db_adapter: Any = None
    desktop_operator: Any = None
    unified_memory: Any = None
    memory_service: Any = None
    mcts_planner: Any = None
    llm_for_collab: Any = None
    voice: Any = None
    openclaw_runtime: Any = None
    openclaw_enabled: bool = False
    sandbox: Any = None
    docker_sandbox: Any = None
    sandbox_pool: Any = None
    # Track what's available for logging
    available: list[str] = field(default_factory=list)


class SubsystemFactory:
    """Single place to initialize all PredaCore subsystems from config.

    Each init method is independent — failures are logged but don't
    block other subsystems from initializing.
    """

    @classmethod
    def create_all(
        cls,
        config: Any,
        *,
        skip_cli_providers: bool = False,
        home_dir: str | None = None,
    ) -> SubsystemBundle:
        """Initialize all subsystems and return a bundle.

        Args:
            config: PredaCoreConfig instance.
            skip_cli_providers: If True, exclude CLI-based LLM providers
                (currently just gemini-cli) from the collaboration LLM chain.
                Set to True when running inside a sub-agent context where a
                nested CLI process would deadlock on stdin.
            home_dir: Override home directory (defaults to config.home_dir or ~/.predacore).
        """
        bundle = SubsystemBundle()
        _home = home_dir or getattr(config, "home_dir", os.path.expanduser("~/.predacore"))

        # DB adapter first — stores use it if available
        cls._init_db_adapter(bundle, config, _home)

        # Order doesn't matter — each is independent
        cls._init_desktop(bundle, config)
        cls._init_unified_memory(bundle, config, _home)
        cls._init_legacy_memory(bundle, config)
        cls._init_mcts(bundle)
        cls._init_voice(bundle)
        cls._init_llm_collab(bundle, config, skip_cli_providers)
        cls._init_openclaw(bundle, config)
        cls._init_sandbox(bundle, config)

        logger.info(
            "SubsystemFactory: initialized [%s] (%d/%d)",
            ", ".join(bundle.available),
            len(bundle.available),
            8,
        )
        return bundle

    # ── Individual Initializers ──────────────────────────────────

    @staticmethod
    def _init_db_adapter(bundle: SubsystemBundle, config: Any, home_dir: str) -> None:
        """DB adapter placeholder — disabled for now.

        SQLite in-process (WAL + 5s busy_timeout) is sufficient for current
        workloads. The socket server infrastructure (db_server.py, db_client.py,
        db_adapter.py) is available if future daemon architectures need it.
        """
        return

    @staticmethod
    def _init_desktop(bundle: SubsystemBundle, config: Any) -> None:
        desktop_enabled = str(
            os.getenv("PREDACORE_ENABLE_DESKTOP_CONTROL", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        if not desktop_enabled:
            return

        try:
            try:
                from predacore.operators.desktop import MacDesktopOperator
            except ImportError:
                from src.predacore.operators.desktop import MacDesktopOperator  # type: ignore

            ops_cfg = getattr(config, "operators", None)
            op = MacDesktopOperator(log=logger, operators_config=ops_cfg)
            if op.available:
                bundle.desktop_operator = op
                bundle.available.append("desktop")
                logger.info("Desktop operator active")
            else:
                logger.info("Desktop operator initialized (inactive on non-macOS)")
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning("Desktop control initialization failed: %s", exc)

    @staticmethod
    def _init_unified_memory(bundle: SubsystemBundle, config: Any, home_dir: str) -> None:
        try:
            try:
                from predacore.memory import UnifiedMemoryStore
            except ImportError:
                from src.predacore.memory import UnifiedMemoryStore  # type: ignore

            try:
                from predacore.services.embedding import get_default_embedding_client
            except ImportError:
                from src.predacore.services.embedding import get_default_embedding_client  # type: ignore

            um_db = str(Path(home_dir) / "memory" / "unified_memory.db")
            bundle.unified_memory = UnifiedMemoryStore(
                db_path=um_db,
                embedding_client=get_default_embedding_client(),
                db_adapter=bundle.db_adapter,
            )
            bundle.available.append("memory")
            logger.info("Unified memory enabled (db=%s)", um_db)
        except (ImportError, OSError, ValueError, RuntimeError, sqlite3.OperationalError) as exc:
            if "database is locked" in str(exc).lower():
                logger.info("Unified memory busy (db lock) — will use session memory only.")
            else:
                logger.warning("Unified memory init failed (non-fatal): %s", exc)

    @staticmethod
    def _init_legacy_memory(bundle: SubsystemBundle, config: Any) -> None:
        try:
            try:
                from predacore.services.embedding import get_default_embedding_client
            except ImportError:
                from src.predacore.services.embedding import get_default_embedding_client  # type: ignore
            try:
                from predacore._vendor.common.memory_service import MemoryService
            except ImportError:
                from predacore._vendor.common.memory_service import MemoryService  # type: ignore

            bundle.memory_service = MemoryService(
                data_path=config.memory.persistence_dir,
                embedding_client=get_default_embedding_client(),
            )
            logger.info(
                "Legacy memory service enabled (dir=%s)",
                config.memory.persistence_dir,
            )
        except (ImportError, OSError, ValueError, RuntimeError, AttributeError, sqlite3.OperationalError) as exc:
            if "database is locked" in str(exc).lower():
                logger.info("Legacy memory busy (db lock) — session memory only.")
            else:
                logger.warning("Legacy memory init failed: %s", exc)

    @staticmethod
    def _init_mcts(bundle: SubsystemBundle) -> None:
        # The MCTS planner transitively pulls spaCy → thinc → PyTorch (~2GB)
        # via _vendor/core_strategic_engine/planner.py's top-level `import spacy`.
        # It's only used by the rarely-invoked `strategic_plan` tool, so we
        # skip the eager init unless explicitly enabled. Set
        # PREDACORE_ENABLE_MCTS_PLANNER=1 to bring it back.
        enabled = str(
            os.getenv("PREDACORE_ENABLE_MCTS_PLANNER", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not enabled:
            logger.debug(
                "MCTS planner skipped (PREDACORE_ENABLE_MCTS_PLANNER unset). "
                "Set to 1 to enable; the tool will error until then."
            )
            return

        try:
            try:
                from predacore._vendor.core_strategic_engine.planner_mcts import ABMCTSPlanner
            except ImportError:
                from predacore._vendor.core_strategic_engine.planner_mcts import ABMCTSPlanner  # type: ignore

            bundle.mcts_planner = ABMCTSPlanner(kn_stub=None, egm_stub=None)
            bundle.available.append("mcts_planner")
            logger.info("MCTS strategic planner enabled")
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning("MCTS planner init failed (non-fatal): %s", exc)

    @staticmethod
    def _init_voice(bundle: SubsystemBundle) -> None:
        try:
            try:
                from predacore.services.voice import VoiceInterface
            except ImportError:
                from src.predacore.services.voice import VoiceInterface  # type: ignore

            bundle.voice = VoiceInterface()
            bundle.available.append("voice")
            logger.info("Voice interface enabled (TTS/STT)")
        except (ImportError, OSError, RuntimeError) as exc:
            logger.debug("Voice interface unavailable: %s", exc)

    @staticmethod
    def _init_llm_collab(
        bundle: SubsystemBundle, config: Any, skip_cli: bool
    ) -> None:
        try:
            try:
                from predacore.llm_providers.router import LLMInterface
            except ImportError:
                from src.predacore.llm_providers.router import LLMInterface  # type: ignore

            collab_config = copy.deepcopy(config)
            agent_llm = getattr(config, "agent_llm", None)
            if getattr(agent_llm, "provider", "").strip():
                collab_config.llm = copy.deepcopy(agent_llm)

            if skip_cli:
                # Build a config that skips CLI-based providers to avoid recursion
                all_providers = [collab_config.llm.provider] + list(
                    collab_config.llm.fallback_providers or []
                )
                direct_providers = [
                    p for p in all_providers if p and p not in _CLI_PROVIDERS
                ]

                if direct_providers:
                    collab_config.llm.provider = direct_providers[0]
                    collab_config.llm.fallback_providers = direct_providers[1:]
                    if collab_config.llm.provider != all_providers[0]:
                        collab_config.llm.model = ""
                else:
                    # Try common fallbacks with available API keys
                    for fallback in ["sambanova", "groq", "openai", "anthropic"]:
                        key_map = {
                            "sambanova": "SAMBANOVA_API_KEY",
                            "groq": "GROQ_API_KEY",
                            "openai": "OPENAI_API_KEY",
                            "anthropic": "ANTHROPIC_API_KEY",
                        }
                        if os.environ.get(key_map.get(fallback, "")):
                            collab_config.llm.provider = fallback
                            collab_config.llm.fallback_providers = []
                            collab_config.llm.model = ""
                            break
                    else:
                        logger.warning(
                            "No direct-API providers available for collaboration"
                        )
                        return

                bundle.llm_for_collab = LLMInterface(collab_config)
            else:
                bundle.llm_for_collab = LLMInterface(collab_config)

            bundle.available.append("multi_agent")
            logger.info(
                "LLM for collaboration enabled (provider=%s)",
                collab_config.llm.provider,
            )
        except (ImportError, OSError, ValueError, RuntimeError, AttributeError) as exc:
            logger.warning("LLM for collaboration init failed (non-fatal): %s", exc)

    @staticmethod
    def _init_openclaw(bundle: SubsystemBundle, config: Any) -> None:
        openclaw_enabled = os.environ.get("PREDACORE_ENABLE_OPENCLAW_BRIDGE", "0") == "1" or bool(
            getattr(getattr(config, "launch", None), "enable_openclaw_bridge", False)
        )
        bundle.openclaw_enabled = openclaw_enabled
        if not openclaw_enabled:
            return

        try:
            try:
                from predacore.agents.autonomy import OpenClawBridgeRuntime
            except ImportError:
                from src.predacore.agents.autonomy import OpenClawBridgeRuntime  # type: ignore

            bundle.openclaw_runtime = OpenClawBridgeRuntime(config)
            bundle.available.append("openclaw")
            logger.info("OpenClaw bridge runtime enabled")
        except (ImportError, OSError, RuntimeError, AttributeError, sqlite3.OperationalError) as exc:
            logger.warning("OpenClaw bridge init failed (non-fatal): %s", exc)

    @staticmethod
    def _init_sandbox(bundle: SubsystemBundle, config: Any) -> None:
        try:
            try:
                from predacore.auth.sandbox import (
                    DockerSandboxManager,
                    SubprocessSandboxManager,
                )
            except ImportError:
                from src.predacore.auth.sandbox import (  # type: ignore
                    DockerSandboxManager,
                    SubprocessSandboxManager,
                )

            bundle.sandbox = SubprocessSandboxManager(logger=logger)
            if getattr(config.security, "docker_sandbox", False):
                try:
                    bundle.docker_sandbox = DockerSandboxManager(logger=logger)
                    logger.info("Docker sandbox enabled")
                except (OSError, ConnectionError, RuntimeError) as e:
                    logger.warning("Docker sandbox unavailable (%s) — using subprocess", e)
            bundle.available.append("sandbox")
        except ImportError:
            logger.warning("Sandbox module not available — raw subprocess fallback")

        # Sandbox pool
        try:
            from predacore.auth.sandbox import SessionSandboxPool
            bundle.sandbox_pool = SessionSandboxPool(max_sessions=50)
        except (ImportError, OSError, RuntimeError, sqlite3.OperationalError):
            pass
