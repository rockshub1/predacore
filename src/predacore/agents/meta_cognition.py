"""
PredaCore Meta-Cognition Engine — Self-evaluation and loop detection.

Evaluates agent responses before sending, detects when the agent is stuck
in a loop, and decides when to ask the user for help.

Usage:
    engine = MetaCognitionEngine(config)
    evaluation = engine.evaluate_response(user_msg, response, tools_used, tool_results)
    if engine.detect_loop(tool_history):
        ...
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response evaluation
# ---------------------------------------------------------------------------


@dataclass
class ResponseEvaluation:
    """Result of evaluating an agent response before sending."""

    confidence: float = 1.0  # 0.0 (no confidence) to 1.0 (fully confident)
    issues: list[str] = field(default_factory=list)
    grounded: bool = True  # Response is grounded in tool results
    is_echo: bool = False  # Response just restates the question
    has_fabrication_risk: bool = False

    @property
    def should_send(self) -> bool:
        """Whether the response is good enough to send."""
        return self.confidence >= 0.3 and not self.is_echo


# Patterns that suggest fabrication (claiming specific data without tool evidence)
_FABRICATION_INDICATORS = (
    re.compile(r"\b(the (current|latest) (price|value|rate|stock) is)\b", re.I),
    re.compile(r"\b(as of (today|now|this moment),?\s+(the|it))\b", re.I),
    re.compile(
        r"\b(i (found|see|noticed) that the (file|directory|code|error))\b", re.I
    ),
    re.compile(r"\b(the (output|result|response) (shows?|was|is|returned))\b", re.I),
)

# Patterns suggesting the response is just restating the question
_ECHO_PATTERNS = (
    re.compile(r"^(sure[,!]?\s*)?(you ('d like|want|asked)|you're asking)", re.I),
    re.compile(r"^(i understand|got it)[,.]?\s*you", re.I),
)


def evaluate_response(
    user_message: str,
    response: str,
    tools_used: list[str],
    tool_results: list[str],
) -> ResponseEvaluation:
    """
    Evaluate an agent response for quality before sending.

    Checks:
    1. Grounding: If response references specific data, was a tool used?
    2. Echo detection: Is response just restating the question?
    3. Fabrication risk: Claims about current data without tool evidence
    4. Empty or trivially short responses
    """
    eval_result = ResponseEvaluation()
    resp = (response or "").strip()
    user_msg = (user_message or "").strip()

    if not resp:
        eval_result.confidence = 0.0
        eval_result.issues.append("empty_response")
        return eval_result

    # 1. Echo detection (skip very short messages — too noisy)
    for pattern in _ECHO_PATTERNS:
        if pattern.search(resp) and len(user_msg) >= 20:
            # Check if the response is mostly restating the question
            if len(resp) < len(user_msg) * 2 and _text_similarity(user_msg, resp) > 0.5:
                eval_result.is_echo = True
                eval_result.confidence -= 0.3
                eval_result.issues.append("echo_detected")
                break

    # 2. Fabrication risk
    if not tools_used:
        for pattern in _FABRICATION_INDICATORS:
            if pattern.search(resp):
                eval_result.has_fabrication_risk = True
                eval_result.grounded = False
                eval_result.confidence -= 0.2
                eval_result.issues.append("fabrication_risk_no_tools")
                break

    # 3. Grounding check — if tools were used, does response relate to results?
    if tools_used and tool_results:
        combined_results = " ".join(str(r)[:500] for r in tool_results).lower()
        resp_lower = resp.lower()

        # Extract key terms from response (words > 5 chars, not common)
        resp_terms = set(re.findall(r"\b[a-z]{6,}\b", resp_lower))
        result_terms = set(re.findall(r"\b[a-z]{6,}\b", combined_results))

        if resp_terms and result_terms:
            overlap = resp_terms & result_terms
            ratio = len(overlap) / max(len(resp_terms), 1)
            if ratio < 0.05 and len(resp) > 200:
                eval_result.grounded = False
                eval_result.confidence -= 0.15
                eval_result.issues.append("low_grounding_overlap")

    # 4. Trivially short response to a complex question
    if len(resp) < 20 and len(user_msg) > 100:
        eval_result.confidence -= 0.1
        eval_result.issues.append("response_too_brief")

    eval_result.confidence = max(0.0, min(1.0, eval_result.confidence))
    return eval_result


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


# Strips leading `cd <path> && ` prefix — the operative command is after the &&.
_CD_PREFIX_RE = re.compile(r"^\s*cd\s+[^\s&|;]+\s*&&\s*")


def _shell_command_fingerprint(command: str) -> str:
    """
    Produce a "meaning" fingerprint for a shell command so loop detection
    can tell `cat a.json` apart from `cat b.json` apart from `npm run build`.

    Rules:
      - Strip leading `cd <path> && ` (it's navigation, not the action)
      - Strip trailing pipes/redirects (`| head`, `2>&1`, `>out.log`)
      - Return up to first 3 tokens of what's left.

    Examples:
      `cd ~/proj && cat package.json | head -5`  → `cat package.json`
      `cd X && npm run build 2>&1 | tail -30`    → `npm run build`
      `cat X/tsconfig.json`                       → `cat X/tsconfig.json`
      `cd X && npx next build 2>&1 | head -100`  → `npx next build`
    """
    if not command:
        return ""
    cmd = command.strip()
    cmd = _CD_PREFIX_RE.sub("", cmd)
    # Cut at pipes or redirects so `cat x | head` == `cat x > y`
    for sep in (" | ", " 2>&1", " 2>/dev/null", " >", " &&"):
        idx = cmd.find(sep)
        if idx > 0:
            cmd = cmd[:idx]
    tokens = cmd.split()
    return " ".join(tokens[:3])


@dataclass
class ToolCall:
    """A single tool call in the agent loop history."""

    name: str
    args_hash: str  # stable hash of arguments for comparison
    iteration: int


def detect_loop(
    tool_history: list[tuple[str, dict[str, Any]]],
    max_repeats: int = 3,
    max_iterations: int = 0,
) -> bool:
    """
    Detect if the agent is stuck in a tool-calling loop.

    Checks:
    1. Same tool called with same args N+ times
    2. Oscillating between two tools (A→B→A→B)
    3. More than max_repeats consecutive identical calls
    4. Same tool called with varied args too many times (thrashing)
    5. Too many total tool calls without progress (stall)
    """
    if len(tool_history) < max_repeats:
        return False

    # Build fingerprints: (tool_name, action_or_args_key)
    # For action-based tools, use the action as part of the fingerprint
    # so navigate + get_page_tree + read_text are NOT the same fingerprint
    _ACTION_TOOLS = frozenset({"browser_control", "desktop_control", "screen_vision"})
    fingerprints = []
    for idx, (name, args) in enumerate(tool_history):
        if name in _ACTION_TOOLS:
            # Use tool_name + action as fingerprint (different actions = different tools)
            action = str(args.get("action", ""))
            url = str(args.get("url", ""))[:50]
            # For read-only actions (get_page_tree, read_text, get_page_links),
            # include the iteration index so repeated reads after different navigations
            # are NOT flagged as the same call
            if action in ("get_page_tree", "read_text", "get_page_links", "get_page_images"):
                fingerprints.append((name, f"{action}:iter{idx}"))
            elif url:
                fingerprints.append((name, f"{action}:{url}"))
            else:
                fingerprints.append((name, action))
        else:
            args_key = _stable_args_key(args)
            fingerprints.append((name, args_key))

    # Check 1: Same exact call repeated max_repeats times
    from collections import Counter

    counts = Counter(fingerprints)
    for fp, count in counts.items():
        if count >= max_repeats:
            logger.warning(
                "Loop detected: %s(%s) called %d times with same args", fp[0], fp[1][:30], count
            )
            return True

    # Check 2: Oscillation pattern (A,B,A,B or A,B,A)
    if len(fingerprints) >= 4:
        recent = fingerprints[-4:]
        if recent[0] == recent[2] and recent[1] == recent[3] and recent[0] != recent[1]:
            logger.warning(
                "Loop detected: oscillation between %s and %s",
                recent[0][0],
                recent[1][0],
            )
            return True

    # Check 3: Last N calls are all identical
    if len(fingerprints) >= max_repeats:
        last_n = fingerprints[-max_repeats:]
        if len(set(last_n)) == 1:
            logger.warning(
                "Loop detected: %d consecutive identical calls to %s",
                max_repeats,
                last_n[0][0],
            )
            return True

    # Check 4: Same tool called 8+ times (raised from 5 — browser workflows need more calls)
    # Exception: if the tool is being called with diverse arguments, it's exploring
    # or doing legitimate bulk work (scaffolding many files, running different
    # commands), not thrashing. We check 3+ unique action/path/command values.
    _PATH_TOOLS = frozenset({
        "write_file", "read_file", "edit_file", "delete_file",
        "create_directory", "list_directory",
    })
    tool_name_counts = Counter(name for name, _ in tool_history)
    for tool_name, count in tool_name_counts.items():
        if count >= 8:
            if tool_name in _ACTION_TOOLS:
                unique_actions = set()
                for name, args in tool_history:
                    if name == tool_name:
                        unique_actions.add(str(args.get("action", "")))
                if len(unique_actions) >= 3:
                    continue  # 3+ different actions = exploring, not thrashing
            elif tool_name in _PATH_TOOLS:
                unique_paths = set()
                for name, args in tool_history:
                    if name == tool_name:
                        p = (
                            args.get("path")
                            or args.get("file_path")
                            or args.get("filename")
                            or ""
                        )
                        unique_paths.add(str(p))
                if len(unique_paths) >= 3:
                    continue  # 3+ different paths = bulk file work, not thrashing
            elif tool_name == "run_command":
                unique_cmds = set()
                for name, args in tool_history:
                    if name == tool_name:
                        fingerprint = _shell_command_fingerprint(
                            str(args.get("command", ""))
                        )
                        unique_cmds.add(fingerprint)
                if len(unique_cmds) >= 3:
                    continue  # 3+ different commands = legitimate shell work
            # Check if the last 8 calls are dominated by this tool (>75%)
            recent_8 = [name for name, _ in tool_history[-8:]]
            recent_ratio = recent_8.count(tool_name) / len(recent_8)
            if recent_ratio >= 0.75:
                logger.warning(
                    "Loop detected: %s called %d times total, %.0f%% of recent calls (thrashing)",
                    tool_name, count, recent_ratio * 100,
                )
                return True

    # Check 5: Stall detection — too many tool calls is suspicious regardless of variety
    # Use 80% of max_iterations as threshold (or min 15) to avoid false positives
    # on agents with low max_steps (e.g. desktop_agent with max_steps=15)
    stall_threshold = max(15, int(max_iterations * 0.8)) if max_iterations else 15
    if len(tool_history) >= stall_threshold:
        logger.warning(
            "Loop detected: %d tool calls (threshold %d) without completing — likely stalled",
            len(tool_history),
            stall_threshold,
        )
        return True

    return False


def should_ask_for_help(
    tool_errors: list[str],
    iteration_count: int,
    max_iterations: int,
    tool_history: list[tuple[str, dict[str, Any]]],
) -> str | None:
    """
    Determine if the agent should stop and ask the user for help.

    Returns a suggestion message if help is needed, None otherwise.
    """
    # Multiple consecutive tool errors
    if len(tool_errors) >= 3:
        recent_errors = tool_errors[-3:]
        return (
            f"I've encountered {len(tool_errors)} tool errors. "
            f"Recent issues: {'; '.join(e[:100] for e in recent_errors)}. "
            "Would you like to try a different approach?"
        )

    # Loop detected (check BEFORE iteration count — catches thrashing early).
    # Pass max_iterations so the stall threshold scales with the configured cap
    # instead of falling back to its 15-call default.
    if detect_loop(tool_history, max_iterations=max_iterations):
        return (
            "I appear to be stuck in a loop calling the same tools repeatedly. "
            "Could you provide more context or try rephrasing your request?"
        )

    # Running out of iterations — trigger at 80% instead of max-1
    # This gives the agent a chance to wrap up rather than slamming into the wall
    threshold = max(2, int(max_iterations * 0.8))
    if iteration_count >= threshold and max_iterations > 1:
        return (
            f"I've used {iteration_count} of {max_iterations} allowed iterations "
            "without completing the task. Should I try a different approach?"
        )

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_similarity(a: str, b: str) -> float:
    """Quick similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower()[:500], b.lower()[:500]).ratio()


def _stable_args_key(args: dict[str, Any]) -> str:
    """Create a stable string key from tool arguments for comparison."""
    if not args:
        return ""
    try:
        import json

        return json.dumps(args, sort_keys=True, default=str)[:200]
    except (TypeError, ValueError):
        return str(sorted(args.items()))[:200]
