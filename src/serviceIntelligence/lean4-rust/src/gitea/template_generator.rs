/// Template Generator
/// Creates markdown templates for Gitea feature specifications
/// that can be converted to Lean4 proofs and SCIP specifications

use chrono::Utc;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConfig {
    pub feature_name: String,
    pub include_api_section: bool,
    pub include_data_model: bool,
    pub include_validation: bool,
    pub include_security: bool,
    pub include_examples: bool,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            feature_name: "New Feature".to_string(),
            include_api_section: true,
            include_data_model: true,
            include_validation: true,
            include_security: false,
            include_examples: true,
        }
    }
}

pub struct TemplateGenerator;

impl TemplateGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate a complete feature specification template
    pub fn generate_feature_template(&self, config: &TemplateConfig) -> String {
        let mut template = Vec::new();

        // Header
        template.push(self.generate_header(&config.feature_name));
        template.push("".to_string());

        // Table of Contents
        template.push(self.generate_toc(config));
        template.push("".to_string());

        // Overview
        template.push(self.generate_overview_section());
        template.push("".to_string());

        // Requirements
        template.push(self.generate_requirements_section());
        template.push("".to_string());

        // API Endpoints
        if config.include_api_section {
            template.push(self.generate_api_section());
            template.push("".to_string());
        }

        // Data Model
        if config.include_data_model {
            template.push(self.generate_data_model_section());
            template.push("".to_string());
        }

        // Validation
        if config.include_validation {
            template.push(self.generate_validation_section());
            template.push("".to_string());
        }

        // Security
        if config.include_security {
            template.push(self.generate_security_section());
            template.push("".to_string());
        }

        // Examples
        if config.include_examples {
            template.push(self.generate_examples_section());
            template.push("".to_string());
        }

        // Implementation Notes
        template.push(self.generate_implementation_notes());
        template.push("".to_string());

        // Metadata
        template.push(self.generate_metadata());

        template.join("\n")
    }

    fn generate_header(&self, feature_name: &str) -> String {
        format!("# {}\n\n> Feature Specification for Gitea\n> Auto-generated template - Replace with actual content", feature_name)
    }

    fn generate_toc(&self, config: &TemplateConfig) -> String {
        let mut toc = vec![
            "## Table of Contents".to_string(),
            "".to_string(),
            "- [Overview](#overview)".to_string(),
            "- [Requirements](#requirements)".to_string(),
        ];

        if config.include_api_section {
            toc.push("- [API Endpoints](#api-endpoints)".to_string());
        }
        if config.include_data_model {
            toc.push("- [Data Model](#data-model)".to_string());
        }
        if config.include_validation {
            toc.push("- [Validation Rules](#validation-rules)".to_string());
        }
        if config.include_security {
            toc.push("- [Security Considerations](#security-considerations)".to_string());
        }
        if config.include_examples {
            toc.push("- [Examples](#examples)".to_string());
        }
        toc.push("- [Implementation Notes](#implementation-notes)".to_string());

        toc.join("\n")
    }

    fn generate_overview_section(&self) -> String {
        r#"## Overview

**Purpose:** [Describe the main purpose of this feature]

**Scope:** [Define what is included and excluded from this feature]

**Target Users:** [Who will use this feature?]

**Success Criteria:** [How do we know this feature is successful?]"#.to_string()
    }

    fn generate_requirements_section(&self) -> String {
        r#"## Requirements

### Functional Requirements

> Use MUST, SHOULD, or MAY to indicate requirement levels:
> - **MUST** / **SHALL** / **REQUIRED** = Mandatory (High Priority)
> - **SHOULD** / **RECOMMENDED** = Strongly suggested (Medium Priority)
> - **MAY** / **OPTIONAL** = Nice to have (Low Priority)

#### Core Functionality
- MUST [describe core requirement 1]
- MUST [describe core requirement 2]
- SHOULD [describe recommended feature 1]
- MAY [describe optional feature 1]

#### User Interface
- MUST [describe UI requirement]
- SHOULD [describe UI enhancement]

#### Data Management
- MUST [describe data requirement]
- MUST NOT [describe constraint]

### Non-Functional Requirements

#### Performance
- MUST respond within [X] seconds for [operation]
- SHOULD handle [N] concurrent users

#### Reliability
- MUST have [X]% uptime
- SHOULD gracefully handle errors

#### Scalability
- MUST support [N] records
- SHOULD scale horizontally

### Constraints
- MUST NOT [describe what should not happen]
- MUST NOT [describe another constraint]"#.to_string()
    }

    fn generate_api_section(&self) -> String {
        r#"## API Endpoints

### Endpoint Format
For each endpoint, specify: METHOD /path - Description

### Read Operations
- GET /api/v1/[resource] - List all [resources]
- GET /api/v1/[resource]/:id - Get specific [resource]

### Write Operations
- POST /api/v1/[resource] - Create new [resource]
- PUT /api/v1/[resource]/:id - Update [resource]
- PATCH /api/v1/[resource]/:id - Partial update [resource]
- DELETE /api/v1/[resource]/:id - Delete [resource]

### Request/Response Examples

#### Create Resource
```json
POST /api/v1/[resource]
Content-Type: application/json

{
  "field1": "value1",
  "field2": "value2"
}
```

#### Response
```json
{
  "id": 123,
  "field1": "value1",
  "field2": "value2",
  "created_at": "2026-01-09T00:00:00Z"
}
```"#.to_string()
    }

    fn generate_data_model_section(&self) -> String {
        r#"## Data Model

### Primary Entity

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| id | integer | yes | auto-generated | Unique identifier |
| name | string | yes | 1-100 chars | Entity name |
| description | text | no | max 5000 chars | Detailed description |
| status | enum | yes | active/inactive | Current status |
| created_at | datetime | yes | auto-generated | Creation timestamp |
| updated_at | datetime | yes | auto-generated | Last update timestamp |
| created_by | integer | yes | foreign key | User who created |

### Related Entities

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| parent_id | integer | no | foreign key | Parent entity reference |
| metadata | json | no | valid json | Additional metadata |

### Relationships
- One-to-Many: [Entity A] → [Entity B]
- Many-to-Many: [Entity X] ↔ [Entity Y] (via junction table)

### Indexes
- Primary: `id`
- Unique: `name`
- Index: `status`, `created_at`"#.to_string()
    }

    fn generate_validation_section(&self) -> String {
        r#"## Validation Rules

### Input Validation

#### Field: name
- MUST be a non-empty string
- MUST be between 1 and 100 characters
- MUST contain only alphanumeric characters and spaces
- MUST NOT contain special characters except hyphen and underscore

#### Field: email
- MUST be a valid email format
- MUST be unique in the system
- MUST NOT exceed 255 characters

#### Field: status
- MUST be one of: [draft, active, inactive, archived]
- MUST transition according to state diagram

### Business Rules
- MUST validate X before Y
- MUST ensure uniqueness of Z
- MUST NOT allow deletion if dependencies exist

### Error Messages
- Invalid field: "[Field] is required"
- Format error: "[Field] must be in format X"
- Business rule: "[Specific business rule violated]""#.to_string()
    }

    fn generate_security_section(&self) -> String {
        r#"## Security Considerations

### Authentication
- MUST require authentication for all endpoints except [list public endpoints]
- MUST use JWT/OAuth tokens for API access
- MUST validate token expiration

### Authorization
- MUST check user permissions before allowing operations
- MUST implement role-based access control (RBAC)
- MUST log all authorization failures

### Data Protection
- MUST encrypt sensitive data at rest
- MUST use HTTPS for all communications
- MUST sanitize all user inputs
- MUST NOT log sensitive information

### Rate Limiting
- MUST implement rate limiting: [N] requests per [time period]
- MUST return 429 (Too Many Requests) when exceeded

### Audit Trail
- MUST log all create/update/delete operations
- MUST include: user_id, timestamp, action, resource
- MUST retain logs for [X] days"#.to_string()
    }

    fn generate_examples_section(&self) -> String {
        r#"## Examples

### Use Case 1: [Common Scenario]

**Scenario:** [Describe the scenario]

**Steps:**
1. User performs action A
2. System validates B
3. System creates/updates C
4. System returns response D

**Expected Outcome:**
- [What should happen]
- [What should be created/updated]

### Use Case 2: [Error Scenario]

**Scenario:** [Describe error case]

**Steps:**
1. User attempts invalid operation
2. System detects error
3. System returns appropriate error message

**Expected Outcome:**
- Error code: 400/403/404/409
- Clear error message explaining the issue

### Use Case 3: [Edge Case]

**Scenario:** [Describe edge case]

**Expected Behavior:**
- [How system should handle this edge case]"#.to_string()
    }

    fn generate_implementation_notes(&self) -> String {
        r#"## Implementation Notes

### Technical Considerations
- **Database:** [Specify database requirements, migrations needed]
- **Dependencies:** [List external libraries or services]
- **Configuration:** [Environment variables or config needed]

### Development Tasks
- [ ] Create database migration
- [ ] Implement data models
- [ ] Create API handlers
- [ ] Add validation logic
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update API documentation
- [ ] Add to Swagger/OpenAPI spec

### Testing Strategy
- **Unit Tests:** [What to test]
- **Integration Tests:** [What to test]
- **E2E Tests:** [What to test]
- **Performance Tests:** [Load testing requirements]

### Documentation Updates
- [ ] Update user documentation
- [ ] Update API documentation
- [ ] Update developer guide
- [ ] Add to changelog

### Deployment Considerations
- **Backward Compatibility:** [Yes/No - explain breaking changes]
- **Migration Path:** [How to migrate from old version]
- **Rollback Plan:** [How to rollback if needed]"#.to_string()
    }

    fn generate_metadata(&self) -> String {
        format!(
            r#"---

## Metadata

**Created:** {}
**Status:** Draft
**Version:** 0.1.0
**Author:** [Your Name]
**Reviewers:** [List reviewers]
**Target Release:** [Version number]

## Related Documents
- Architecture Decision Record: [Link]
- Design Document: [Link]
- API Specification: [Link]

## Change Log
- {} - Initial draft created"#,
            Utc::now().format("%Y-%m-%d"),
            Utc::now().format("%Y-%m-%d")
        )
    }

    /// Generate a minimal quick-start template
    pub fn generate_quick_template(&self, feature_name: &str) -> String {
        format!(
            r#"# {}

## Requirements
- MUST [core requirement 1]
- MUST [core requirement 2]
- SHOULD [optional enhancement]

## API Endpoints
- POST /api/v1/[resource] - Create
- GET /api/v1/[resource] - List
- GET /api/v1/[resource]/:id - Get one

## Data Model

| Field | Type | Required |
|-------|------|----------|
| id | integer | yes |
| name | string | yes |
| created_at | datetime | yes |

## Validation
- name: MUST be 1-100 characters
- name: MUST be unique

---
Created: {}
"#,
            feature_name,
            Utc::now().format("%Y-%m-%d")
        )
    }

    /// Generate a template with best practices guide
    pub fn generate_guided_template(&self) -> String {
        r#"# Feature Specification Template - Guided Version

## 📋 How to Use This Template

This template helps you create a specification that can be:
1. **Automatically converted** to Gitea Go code
2. **Formally verified** with Lean4 proofs
3. **Tracked for compliance** with SCIP specifications

### Tips for Success:
- Use **MUST/SHOULD/MAY** keywords for requirements (they're automatically detected!)
- Define API endpoints with **METHOD /path** format (automatically parsed!)
- Use **markdown tables** for data models (converted to Go structs!)
- Be specific and clear - this spec drives code generation!

---

## Overview

**What does this feature do?**
[One paragraph explaining the feature's purpose]

**Who is it for?**
[Target users or use cases]

---

## Requirements

> 💡 **Tip:** Requirements starting with MUST/SHOULD/MAY are automatically:
> - Converted to Lean4 axioms for formal verification
> - Tagged with severity levels in SCIP (MUST=High, SHOULD=Medium, MAY=Low)
> - Added as comments in generated Go code

### Core Requirements
- MUST [mandatory requirement - will be marked as HIGH severity]
- MUST [another mandatory requirement]
- SHOULD [recommended feature - will be MEDIUM severity]
- MAY [optional feature - will be LOW severity]

### Constraints  
- MUST NOT [thing that should never happen - will be CRITICAL severity]

---

## API Endpoints

> 💡 **Tip:** Use format "METHOD /path - Description"
> Supported methods: GET, POST, PUT, PATCH, DELETE

### Endpoints
- POST /api/v1/resource - Create new resource
- GET /api/v1/resource - List all resources
- GET /api/v1/resource/:id - Get specific resource
- PUT /api/v1/resource/:id - Update resource
- DELETE /api/v1/resource/:id - Delete resource

---

## Data Model

> 💡 **Tip:** This table will be converted to:
> - Go struct with appropriate types and tags
> - Lean4 structure definition
> - SCIP data model specification

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| id | integer | yes | auto | Unique identifier |
| name | string | yes | 1-100 chars | Resource name |
| status | string | yes | enum | One of: active, inactive |
| created_at | datetime | yes | auto | Creation time |

---

## Validation Rules

> 💡 **Tip:** Use format "field: rule" for automatic parsing

- name: MUST be unique
- name: MUST be 1-100 characters
- status: MUST be one of [active, inactive, archived]
- email: MUST be valid email format

---

## Next Steps

After completing this spec:
1. **Save as** `feature-name.md`
2. **POST to** `/generate` endpoint
3. **Receive:**
   - `feature.go` - Gitea implementation scaffolding
   - `feature_proofs.lean` - Formal verification
   - `scip_spec.json` - Compliance tracking
4. **Implement** the business logic in Go
5. **Verify** with Lean4 proofs
6. **Track** with SCIP

---

## Metadata

Created: [DATE]
Status: Draft
Author: [YOUR NAME]
"#.to_string()
    }

    /// Generate template with real-world example
    pub fn generate_example_template(&self) -> String {
        r#"# Pull Request Review Feature

## Overview

This feature adds automated code review checks to pull requests in Gitea, helping teams maintain code quality and catch issues before merging.

## Requirements

### Core Functionality
- MUST run automated checks on every pull request
- MUST block merging if critical checks fail
- SHOULD provide inline comments on issues found
- SHOULD integrate with CI/CD pipeline
- MAY support custom check configurations

### User Interface
- MUST display check status prominently on PR page
- MUST show detailed results for each check
- SHOULD update status in real-time as checks complete

### Performance
- MUST complete checks within 5 minutes for typical PR
- SHOULD process checks in parallel
- MUST NOT block other PR operations

### Constraints
- MUST NOT merge PR if any MUST checks fail
- MUST NOT expose sensitive information in check results

## API Endpoints

- POST /api/v1/repos/:owner/:repo/pulls/:index/reviews - Submit code review
- GET /api/v1/repos/:owner/:repo/pulls/:index/reviews - List reviews
- GET /api/v1/repos/:owner/:repo/pulls/:index/checks - Get check status
- POST /api/v1/repos/:owner/:repo/pulls/:index/checks/:id/rerun - Rerun specific check

## Data Model

### PullRequestReview

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| id | integer | yes | auto-generated | Unique identifier |
| pr_id | integer | yes | foreign key | Pull request reference |
| reviewer_id | integer | yes | foreign key | User who reviewed |
| status | enum | yes | pending/approved/rejected | Review status |
| comment | text | no | max 10000 chars | Review comment |
| created_at | datetime | yes | auto-generated | Review timestamp |

### CheckRun

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| id | integer | yes | auto-generated | Unique identifier |
| pr_id | integer | yes | foreign key | Pull request reference |
| name | string | yes | 1-100 chars | Check name |
| status | enum | yes | queued/running/completed | Check status |
| conclusion | enum | no | success/failure/neutral | Check result |
| started_at | datetime | no | - | Start time |
| completed_at | datetime | no | - | Completion time |

## Validation Rules

- status: MUST be one of [pending, approved, rejected, changes_requested]
- comment: MUST be provided if status is rejected or changes_requested
- reviewer_id: MUST NOT be the same as PR author
- check timeout: MUST fail if running longer than 30 minutes

## Security Considerations

- MUST require write access to submit reviews
- MUST validate reviewer has repository access
- MUST NOT allow self-approval
- MUST audit all review actions

## Examples

### Use Case 1: Successful Review

**Scenario:** Developer submits PR, all checks pass, reviewer approves

**Steps:**
1. POST /api/v1/repos/myorg/myrepo/pulls/123/reviews with status=approved
2. System validates reviewer permissions
3. System records review
4. System updates PR status to "ready to merge"

**Expected Outcome:**
- Review saved with status "approved"
- PR shows green checkmark
- Merge button becomes enabled

### Use Case 2: Failed Checks

**Scenario:** Automated checks find issues

**Steps:**
1. POST to /checks endpoint triggers checks
2. Lint check finds style violations
3. System creates check run with conclusion=failure
4. System blocks PR merge

**Expected Outcome:**
- PR shows red X for failed check
- Detailed error messages displayed
- Merge button disabled

---

**Created:** 2026-01-09
**Status:** Draft - Ready for Implementation
**Author:** Development Team
**Target Release:** v1.23.0
"#.to_string()
    }
}

impl Default for TemplateGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_feature_template() {
        let generator = TemplateGenerator::new();
        let config = TemplateConfig::default();
        let template = generator.generate_feature_template(&config);
        
        assert!(template.contains("# New Feature"));
        assert!(template.contains("## Requirements"));
        assert!(template.contains("MUST"));
    }

    #[test]
    fn test_generate_quick_template() {
        let generator = TemplateGenerator::new();
        let template = generator.generate_quick_template("Test Feature");
        
        assert!(template.contains("# Test Feature"));
        assert!(template.contains("## Requirements"));
    }

    #[test]
    fn test_minimal_sections() {
        let generator = TemplateGenerator::new();
        let config = TemplateConfig {
            feature_name: "Minimal".to_string(),
            include_api_section: false,
            include_data_model: false,
            include_validation: false,
            include_security: false,
            include_examples: false,
        };
        let template = generator.generate_feature_template(&config);
        
        assert!(template.contains("## Requirements"));
        assert!(!template.contains("## API Endpoints"));
    }
}