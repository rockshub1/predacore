# Using MCP servers with PredaCore

PredaCore ships with an **MCP (Model Context Protocol) client** — the
inbound side of the spec. Every MCP server in the ecosystem
(Anthropic's first-party ones, community-built, and your own) becomes
callable by the agent without writing a PredaCore tool wrapper.

PredaCore does **not** run an MCP server. All of its built-in tools go
through the PredaCore SDK directly. MCP is the "reach out and use other
people's tools" layer — not "expose our own tools over yet another
protocol."

## How an MCP tool becomes a PredaCore tool

For every configured MCP server, the daemon:

1. Spawns the server as a subprocess.
2. Sends `initialize` → `tools/list`.
3. Mounts each discovered tool into `HANDLER_MAP` under the name
   `mcp_<server>_<tool>`.
4. Adds a matching schema entry to the LLM's tool-definitions list.

The LLM sees `mcp_filesystem_read_file`, `mcp_github_search_issues`,
etc. as normal tool names — no meta-"call the MCP system" step.

## Add a server (two ways)

### During setup (`config.yaml`)

```yaml
mcp_servers:
  - name: filesystem
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/Users/YOU"]

  - name: github
    command: ["npx", "-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: ${GITHUB_TOKEN}

  - name: slack
    command: ["npx", "-y", "@modelcontextprotocol/server-slack"]
    env:
      SLACK_BOT_TOKEN: ${SLACK_BOT_TOKEN}
      SLACK_TEAM_ID:   ${SLACK_TEAM_ID}
```

`${VAR}` gets expanded from the main process environment, so put your
secrets in `~/.predacore/.env` (or use `secret_set` in chat) and
reference them here.

### Mid-conversation (via the agent)

```
You: add the filesystem MCP rooted at /Users/YOU
PredaCore:
  [mcp_add name=filesystem command=["npx","-y","@modelcontextprotocol/server-filesystem","/Users/YOU"]]
  → installed + 8 tools mounted (mcp_filesystem_read_file,
    mcp_filesystem_write_file, mcp_filesystem_list_directory, ...)

You: add github, here's my token ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
PredaCore:
  [secret_set name=GITHUB_TOKEN]
  [mcp_add name=github command=["npx","-y","@modelcontextprotocol/server-github"]
           env={"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"}]
  → 27 tools mounted.
```

Both `mcp_add` and `secret_set` persist to the right files
(`config.yaml` and `.env`, chmod 600 where appropriate) so restarts
keep working.

### Available management tools

| Tool | Purpose |
|---|---|
| `mcp_list` | Show every configured server, whether it's running, and the tools it currently exposes. |
| `mcp_add` | Register and launch a new server. Optional `install.npm`/`install.pip` runs the backing package installer first. Persists to config.yaml by default. |
| `mcp_remove` | Tear down a server + drop it from config. |
| `mcp_restart` | Bounce a server (after a token change, crash, or config edit). |

## Trust and security

- `mcp_add` with an `install` step is **blocked in paranoid trust
  mode** and prompts for approval in normal mode. Every `npx ... -y`
  you'd run manually is the same thing PredaCore runs for you —
  inspect the package before saying yes.
- Each subprocess inherits the main process environment merged with
  the `env` dict — it can read the same files and call the same
  network endpoints you can. There is no sandbox; consider the Docker
  sandbox pattern (`docker/sandbox/Dockerfile`) if you need isolation.
- MCP servers can crash. The registry logs the failure and keeps
  PredaCore running — the mounted tools simply disappear until the
  server is back. Use `mcp_restart` to bring one back without a full
  daemon restart.

## Writing your own MCP server

If you want PredaCore to have a new capability, two options:

1. **Write an MCP server** → every MCP-aware client (Claude Desktop,
   Cline, PredaCore, etc.) gets it for free. Use Anthropic's TypeScript
   or Python SDK; ship to npm or PyPI; register via `mcp_add`.
2. **Write a PredaCore tool** → only PredaCore uses it, but it lives
   inside the project with full access to the config and other tools.
   See `src/predacore/tools/handlers/` for the pattern.

Both work; the MCP path is better when the capability is generic and
you want reuse across agents.

## Debugging

- Server stderr is forwarded to PredaCore's logs at DEBUG level. Run
  `predacore logs --follow` to watch.
- `mcp_list` reports which servers are running; a server that
  disappears from `running: true` → `false` likely crashed. Check
  logs, then `mcp_restart`.
- Protocol mismatches show up as `MCPClientError` with the server's
  JSON-RPC error message passed through verbatim.
