# Vendor Directory

This directory contains vendored dependencies and integrations, organized by layer.

## Layered Layout

- **layerCore**: `apisix`, `gitea`, `keycloak`, `markitdown`
- **layerData**: `dragonflydb`, `marquez`, `memgraph`, `postgres`, `qdrant`
- **layerAutomation**: `langflow`
- **layerIntelligence**: `shimmy-ai`, `lean4`, `hyperbooklm`, `n8n`, `memgraph-ai-toolkit`
- **layerUi**: `open-canvas`
- **layerModels**: `folderRepos`, `huggingFace`

## Integration Status

| Project | Language | Location | Status |
|---------|----------|----------|--------|
| APISIX | Lua/OpenResty | `vendor/layerCore/apisix/` | ✅ Vendored |
| Keycloak | Java | `vendor/layerCore/keycloak/` | ✅ Vendored |
| Gitea | Go | `vendor/layerCore/gitea/` | ✅ Vendored |
| MarkItDown | Python | `vendor/layerCore/markitdown/` | ✅ Vendored |
| Memgraph | C++ | `vendor/layerData/memgraph/` | ✅ Vendored |
| Qdrant | Rust | `vendor/layerData/qdrant/` | ✅ Vendored |
| DragonflyDB | C++ | `vendor/layerData/dragonflydb/` | ✅ Vendored |
| Postgres | C | `vendor/layerData/postgres/` | ✅ Vendored |
| Marquez | Java | `vendor/layerData/marquez/` | ✅ Vendored |
| Langflow | Python | `vendor/layerAutomation/langflow/` | ✅ Vendored |
| Shimmy | Rust | `vendor/layerIntelligence/shimmy-ai/` | ✅ Vendored |
| Lean4 | Lean/C++ | `vendor/layerIntelligence/lean4/` | ✅ Vendored |
| HyperbookLM | TypeScript/Next.js | `vendor/layerIntelligence/hyperbooklm/` | ✅ Vendored |
| n8n | TypeScript/Node | `vendor/layerIntelligence/n8n/` | ✅ Vendored |
| Memgraph AI Toolkit | Python | `vendor/layerIntelligence/memgraph-ai-toolkit/` | ✅ Vendored |
| Open Canvas | TypeScript/React | `vendor/layerUi/open-canvas/` | ✅ Vendored |

## Updating Vendored Projects

Update within each layer directory. Examples:

```bash
cd vendor/layerUi/open-canvas
git pull origin main

cd vendor/layerIntelligence/shimmy-ai
git pull origin main

cd vendor/layerData/qdrant
git pull origin master
```

## Notes

- **A2UI**: Specification and samples live under `vendor/layerData/memgraph/a2ui/` and Shimmy includes A2UI adapters.
- **ToolOrchestra**: Vendor source lives under `vendor/layerModels/folderRepos/toolorchestra/`.
