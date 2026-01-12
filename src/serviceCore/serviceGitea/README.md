# serviceGitea - Auto-Generated Gitea Feature Services

## Overview

This directory contains auto-generated Gitea feature implementations in Rust (and Go where needed), created by the forward engineering pipeline.

```
Design (n8n) → Specification (MD) → Generation → Deploy Here
```

## Directory Structure

```
src/serviceCore/serviceGitea/
├── README.md                    # This file
├── Cargo.toml                   # Workspace Cargo.toml
├── features/                    # Generated feature implementations
│   ├── {feature-name}/
│   │   ├── Cargo.toml          # Feature-specific dependencies
│   │   ├── README.md           # Feature documentation
│   │   ├── src/
│   │   │   ├── lib.rs          # Rust implementation
│   │   │   ├── handlers.rs     # HTTP handlers
│   │   │   ├── models.rs       # Data models
│   │   │   └── validation.rs   # Validation logic
│   │   ├── proofs/
│   │   │   └── feature.lean    # Lean4 formal proofs
│   │   ├── specs/
│   │   │   └── scip.json       # SCIP compliance spec
│   │   └── tests/
│   │       └── integration.rs  # Tests
│   └── .gitkeep
├── shared/                      # Shared utilities
│   ├── mod.rs
│   ├── auth.rs                 # Authentication helpers
│   ├── db.rs                   # Database utilities
│   └── types.rs                # Common types
└── templates/                   # Code generation templates
    ├── rust_service.hbs
    ├── lean4_proof.hbs
    └── scip_spec.hbs
```

## Generated Feature Example

### Input: n8n Workflow
```json
{
  "name": "PR Review Automation",
  "nodes": [...]
}
```

### Output: Rust Service

**`features/pr-review/src/lib.rs`:**
```rust
//! PR Review Automation Feature
//! Auto-generated from n8n workflow design

use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct PrReview {
    pub pr_id: i64,
    pub status: ReviewStatus,
    pub score: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ReviewStatus {
    Pending,
    Approved,
    Rejected,
}

// REQ-001: MUST validate all incoming PRs
pub async fn validate_pr(pr_id: i64) -> Result<bool> {
    // Implementation
    Ok(true)
}

// REQ-002: MUST pass lint checks
pub async fn run_lint_checks(pr_id: i64) -> Result<bool> {
    // Implementation
    Ok(true)
}
```

**`features/pr-review/proofs/feature.lean`:**
```lean
-- Lean4 Formal Verification
namespace PRReviewAutomation

axiom MUST_validate_all_incoming_PRs : True
axiom MUST_pass_lint_checks : True

structure PrReview where
  pr_id : Nat
  status : String
  score : Nat

theorem PRReviewAutomation_complete : True := by
  trivial

end PRReviewAutomation
```

**`features/pr-review/specs/scip.json`:**
```json
{
  "scip_version": "1.0",
  "feature": "PR Review Automation",
  "elements": [
    {
      "id": "REQ-001",
      "element_type": "requirement",
      "title": "MUST validate all incoming PRs",
      "severity": "high",
      "formal_spec": "axiom MUST_validate_all_incoming_PRs : True",
      "verification_method": "formal_proof"
    }
  ]
}
```

## Integration with Gitea

Generated Rust services integrate with Gitea through:

1. **HTTP API** - RESTful endpoints for Gitea webhooks
2. **Database** - Direct database access for Gitea data
3. **Git Operations** - Git CLI or libgit2 for repository operations
4. **Authentication** - JWT/OAuth integration with Gitea auth

## Deployment Process

### Automated by serviceN8n:

1. **Generate** - n8n workflow → Lean4 Parser → Rust code
2. **Verify** - Lean4 proof checker validates requirements
3. **Deploy** - Write to `features/{feature-name}/`
4. **Build** - `cargo build` in feature directory
5. **Test** - `cargo test` runs generated tests
6. **Integrate** - Add to main Cargo workspace
7. **Start** - Feature becomes available immediately

### Manual Steps (if needed):

```bash
# Navigate to feature
cd src/serviceCore/serviceGitea/features/{feature-name}

# Build
cargo build --release

# Test
cargo test

# Run standalone
cargo run --bin {feature-name}-service
```

## Feature Registry

Generated features are registered in `Cargo.toml`:

```toml
[workspace]
members = [
    "features/pr-review",
    "features/issue-triage",
    "features/release-automation",
]

[dependencies]
# Shared dependencies
actix-web = "4.4"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.35", features = ["full"] }
```

## API Server

A main server aggregates all feature endpoints:

**`src/main.rs`:**
```rust
use actix_web::{web, App, HttpServer};

mod features {
    pub mod pr_review;
    pub mod issue_triage;
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/api/pr/review", web::post().to(features::pr_review::handle))
            .route("/api/issue/triage", web::post().to(features::issue_triage::handle))
    })
    .bind("0.0.0.0:8004")?
    .run()
    .await
}
```

## Configuration

**`.env`:**
```bash
# Gitea Integration
GITEA_URL=http://localhost:3000
GITEA_TOKEN=your-gitea-token

# Database
DATABASE_URL=postgresql://user:pass@localhost/gitea

# Service
SERVICE_PORT=8004
LOG_LEVEL=info

# Lean4 Verification
LEAN4_PATH=/usr/local/bin/lean
VERIFY_ON_DEPLOY=true
```

## Monitoring

Each generated feature includes:

- **Health endpoint** - `/health/{feature-name}`
- **Metrics endpoint** - `/metrics/{feature-name}`
- **Logs** - Structured JSON logging
- **Tracing** - OpenTelemetry integration

## Development Workflow

### For Generated Features:

1. **Review** - Check generated code in PR
2. **Customize** - Add business logic if needed
3. **Test** - Ensure tests pass
4. **Deploy** - Merge to trigger deployment

### For Manual Features:

1. **Create** - Use template or copy structure
2. **Implement** - Write Rust code
3. **Prove** - Write Lean4 proofs
4. **Spec** - Create SCIP specification
5. **Test** - Write comprehensive tests
6. **Deploy** - Add to workspace

## Quality Checks

All features must pass:

- ✅ **Rust compiler** - `cargo check`
- ✅ **Linter** - `cargo clippy`
- ✅ **Tests** - `cargo test`
- ✅ **Lean4 verification** - `lean --check proofs/feature.lean`
- ✅ **SCIP validation** - Schema validation
- ✅ **Security scan** - `cargo audit`

## Migration from Go

For existing Gitea Go code:

1. **Extract** - Identify Go handlers/logic
2. **Specify** - Write n8n workflow or MD spec
3. **Generate** - Use forward engineering pipeline
4. **Compare** - Validate Rust matches Go behavior
5. **Replace** - Gradually switch to Rust services
6. **Deprecate** - Remove old Go code

## Roadmap

- [x] Design directory structure
- [ ] Create Cargo workspace
- [ ] Implement shared utilities
- [ ] Create service templates
- [ ] Build feature registry
- [ ] Add health/metrics endpoints
- [ ] Integrate with CI/CD
- [ ] Add monitoring dashboards

---

**Status:** Architecture defined, ready for first feature
**Language:** Rust (primary), Go (legacy/compatibility)
**Port:** 8004
**Integration:** Gitea webhooks, database, Git operations