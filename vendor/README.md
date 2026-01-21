# Vendor Directory

This directory contains vendored dependencies and integrations, organized by layer.

## Layered Layout

- **layerCore**: `apisix`, `gitea`, `keycloak`
- **layerData**: `dragonflydb`, `marquez`, `memgraph`, `postgres`, `qdrant`
- **layerIntelligence**: `shimmy-ai`, `hyperbooklm`, `n8n`
- **layerUi**: `open-canvas`
- **layerModels**: `folderRepos`, `huggingFace`

## Integration Status

| Project | Language | Location | Status |
|---------|----------|----------|--------|
| APISIX | Lua/OpenResty | `vendor/layerCore/apisix/` | ✅ Vendored |
| Keycloak | Java | `vendor/layerCore/keycloak/` | ✅ Vendored |
| Gitea | Go | `vendor/layerCore/gitea/` | ✅ Vendored |
| Memgraph | C++ | `vendor/layerData/memgraph/` | ✅ Vendored |
| Qdrant | Rust | `vendor/layerData/qdrant/` | ✅ Vendored |
| DragonflyDB | C++ | `vendor/layerData/dragonflydb/` | ✅ Vendored |
| Postgres | C | `vendor/layerData/postgres/` | ✅ Vendored |
| Marquez | Java | `vendor/layerData/marquez/` | ✅ Vendored |
| Shimmy | Rust | `vendor/layerIntelligence/shimmy-ai/` | ✅ Vendored |
| HyperbookLM | TypeScript/Next.js | `vendor/layerIntelligence/hyperbooklm/` | ✅ Vendored |
| n8n | TypeScript/Node | `vendor/layerIntelligence/n8n/` | ✅ Vendored |
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
