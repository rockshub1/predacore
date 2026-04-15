"""
LLM-guided scheduler for the Dynamic Agent Fabric (DAF).

This module proposes spawn/retire actions based on simple metrics by asking the
configured LLM to emit a strictly-structured JSON plan. All actions must pass
EGM checks and are executed deterministically by the DAF service.

Enabled only when LLM_SCHEDULER_ENABLED=1 to avoid test interference.
"""
from __future__ import annotations

import json
import os
from typing import Any

from jarvis._vendor.common.llm import default_params, get_default_llm_client
from jarvis._vendor.common.protos import egm_pb2


def _fmt_state(
    agent_types: list[dict[str, Any]], instances: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "agent_types": agent_types,
        "instances": [
            {
                "instance_id": inst.get("instance_id"),
                "type_id": inst.get("type_id"),
                "status": int(inst.get("status", 0)),
            }
            for inst in instances
        ],
    }


async def propose_roster_diff(
    llm_client, state: dict[str, Any], limits: dict[str, Any]
) -> dict[str, Any]:
    """Ask the LLM to propose spawn/retire operations in strict JSON.

    limits: {"max_total": int, "max_per_type": {type_id: int}}
    """
    system = (
        "You are an operations planner. You must output only JSON. "
        "Given the current agents and limits, propose spawn/retire actions to optimize throughput. "
        "Constraints: never exceed per-type or total limits; prefer keeping at least one python_executor."
    )
    user = json.dumps({"state": state, "limits": limits}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    schema_hint = (
        "Respond with: {\n"
        '  "actions": [ {"op": "spawn|retire", "agent_type_id": string, "count": int } ],\n'
        '  "justification": string\n'
        "}"
    )
    content = await llm_client.generate(
        messages + [{"role": "system", "content": schema_hint}],
        params=default_params(temperature=0.1, max_tokens=400),
    )
    try:
        plan = json.loads(content)
        if not isinstance(plan, dict) or "actions" not in plan:
            raise ValueError("Missing actions in response")
        # sanitize actions
        actions = []
        for a in plan.get("actions", []):
            op = str(a.get("op", "")).lower()
            t = str(a.get("agent_type_id", ""))
            c = int(a.get("count", 0) or 0)
            if op in {"spawn", "retire"} and t and c > 0:
                actions.append({"op": op, "agent_type_id": t, "count": c})
        return {"actions": actions, "justification": plan.get("justification", "")}
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return {"actions": [], "justification": "parse_error"}


async def scheduler_tick(daf_service, egm_stub, logger) -> None:
    """One scheduling iteration. Safe to no-op if disabled or misconfigured."""
    if os.getenv("LLM_SCHEDULER_ENABLED") != "1":
        return

    try:
        llm = get_default_llm_client(logger=logger)
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        logger.debug(f"LLM scheduler disabled (no client): {e}")
        return

    # Gather state
    agent_types = daf_service.agent_type_registry.list_agent_types()
    instances = await daf_service.agent_instance_registry.list_instances()
    state = _fmt_state(agent_types, instances)

    # Limits (basic defaults; later make configurable)
    per_type_limit = {t["agent_type_id"]: 3 for t in agent_types}
    total_limit = 3 * min(3, len(agent_types)) if agent_types else 3
    limits = {"max_total": total_limit, "max_per_type": per_type_limit}

    plan = await propose_roster_diff(llm, state, limits)
    for action in plan.get("actions", []):
        op = action["op"]
        t = action["agent_type_id"]
        count = action["count"]
        try:
            # EGM check
            from google.protobuf.struct_pb2 import Struct

            desc = Struct()
            desc.update({"action_type": op.upper(), "agent_type_id": t, "count": count})
            req = egm_pb2.CheckActionComplianceRequest(action_description=desc)
            res = await egm_stub.CheckActionCompliance(req)
            if not res.is_compliant:
                logger.info(f"Scheduler action blocked by EGM: {op} {t} x{count}")
                continue

            if op == "spawn":
                from jarvis._vendor.common.protos import daf_pb2 as _daf

                for _ in range(min(count, per_type_limit.get(t, 0))):
                    await daf_service.SpawnAgent(
                        _daf.SpawnAgentRequest(agent_type_id=t), None
                    )
            elif op == "retire":
                # retire idle instances first
                all_instances = (
                    await daf_service.agent_instance_registry.list_instances()
                )
                cand = [i for i in all_instances if i["type_id"] == t]
                for inst in cand[:count]:
                    from jarvis._vendor.common.protos import daf_pb2 as _daf

                    await daf_service.RetireAgent(
                        _daf.RetireAgentRequest(agent_instance_id=inst["instance_id"]),
                        None,
                    )
        except Exception as e:
            logger.debug(f"Scheduler failed action {action}: {e}")
