# LayerCore MCP Adapters

LayerCore bundles the infrastructure-facing MCP adapters the platform still relies on after retiring legacy workflow tools. These adapters expose a narrow, audited surface area into core services so agent runtimes (nWorkflow, nOpenaiServer, toolorchestra) can reuse the same primitives.

## Maintained adapters

| Adapter | Path | Primary capability |
|---------|------|--------------------|
| Filesystem | `./filesystem` | Read/write access to project artifacts, build outputs, and generated documentation for automation jobs. |
| Git | `./git` | Safe repository interactions against the internal Gitea remote (clone, commit, diff, push) used by CI bots and release tooling. |
| Memory | `./memory` | Conversation/agent scratchpad persisted in Dragonfly/Postgres so long-running orchestration chains can resume contextually. |
| Postgres | `./postgres` | Parameterized SQL interface into the operational databases (Keycloak, Registry, Gitea, Marquez). Used for inspections, migrations, and health automation. |

All other adapters that previously targeted removed vendors (Langflow, n8n, etc.) have been deleted. New adapters should be added only when there is a concrete first-party consumer and corresponding audit plan.

## Usage patterns

1. **Service automation** – backend jobs and nWorkflow plans call the MCP adapters through `rust_cli_adapter.py`, ensuring a consistent auth/logging surface.
2. **Operational tooling** – maintenance scripts (e.g., registry migrations, database integrity checks) invoke the same adapters instead of using bespoke shell commands.
3. **Agent runtimes** – when exposing MCPs directly to external agent hosts, point them to these adapters to guarantee parity with production automation.

## Configuration

- Environment variables for connection strings live in the root `.env*` files and are injected via Docker Compose.
- Adapter-specific policies (e.g., paths accessible by Filesystem MCP) are defined next to each adapter in this directory; update them when changing directory structure.

## Contribution notes

1. Prefer extending existing adapters over adding new ones.
2. Document every new command or endpoint in this README so consumers know what is supported.
3. When deprecating an adapter, remove it here, delete the underlying code, and update any orchestration scripts that referenced it.