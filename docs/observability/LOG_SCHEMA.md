# Unified JSON Log Schema

All components emit structured logs via `log_json(logger, level, event, **fields)`.

## Common fields
- event: string (e.g., `csc.process_goal.start`, `wil.execute_tool.end`)
- trace_id: string (propagated end-to-end)
- ts: implicit (logger timestamp)
- component: implicit by logger name

## CSC
- csc.process_goal.start: goal_id
- csc.plan.generated: goal_id, plan_id
- csc.plan.egm_checked: goal_id, plan_id, is_compliant (bool)
- csc.plan.score: goal_id, score
- csc.plan.mcts_improve: goal_id, depth, best_score

## DAF
- daf.dispatch.start: trace_id, task_id, tool_id, agent_instance_id
- daf.dispatch.end: trace_id, task_id, status

## WIL
- wil.execute_tool.start: trace_id, request_id, tool_id
- wil.execute_tool.end: trace_id, tool_id, status
- wil.execute_code.start: trace_id, request_id
- wil.execute_code.end: trace_id, status

## Notes
- All logs should be single-line JSON and avoid large blobs; include identifiers and status.
- For errors, use `level=ERROR` and add `error_message` if relevant. Keep PII out of logs.

