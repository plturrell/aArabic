# Process Apps - Common Library

Reusable components, utilities, and patterns for building process applications.

## Overview

This library provides shared functionality used across all process apps in the `/src/nApps/Process/` directory. It includes backend workflow engines, UI components, HANA utilities, and integration helpers.

## Structure

```
_Common/
├── backend/
│   ├── workflow/         # Workflow engines
│   ├── models/           # Common data models
│   ├── utils/            # Utilities
│   └── integration/      # Integration helpers
└── webapp/
    ├── components/       # Reusable UI5 components
    ├── utils/            # Frontend utilities
    └── fragments/        # UI fragments
```

## Backend Components

### Workflow

#### approval_engine.zig
Reusable approval workflow engine supporting:
- Two-tier approval (Checker → Manager)
- Three-tier approval (Maker → Checker → Manager)
- Parallel approval (all must approve)
- Timeout and escalation
- Approval history tracking

**Usage:**
```zig
const ApprovalEngine = @import("_Common/backend/workflow/approval_engine.zig").ApprovalEngine;
const createThreeTierChain = @import("_Common/backend/workflow/approval_engine.zig").createThreeTierChain;

var approval_engine = try ApprovalEngine.init(allocator, createThreeTierChain());
defer approval_engine.deinit();

// Process approval
const result = try approval_engine.processApproval(
    "instance-001",
    "user-123",
    .checker,
    .approve,
    "Approved"
);
```

### Utils

Common utilities for:
- HANA Cloud connections
- Audit logging
- Data validation
- Date/time formatting
- Currency handling

### Integration

Helper modules for:
- nAgentFlow client
- nLocalModels client (AI narratives)
- nGrounding client (context)
- SAP Destination service wrapper

## Frontend Components

### Reusable UI5 Components

Components that can be used across all process apps:

1. **ApprovalWorkflow** - Visual approval workflow component
   - Shows approval chain
   - Displays current status
   - Action buttons (Approve/Reject)
   - Comments section

2. **ProcessTimeline** - Timeline visualization
   - Shows process history
   - Displays state transitions
   - User actions with timestamps
   - Audit trail viewer

3. **StatusBadge** - Standardized status indicators
   - Color-coded by state
   - Configurable icons
   - Tooltip with details

4. **WorklistTable** - Generic worklist table
   - Filterable by state/role
   - Sortable columns
   - Bulk actions
   - Export functionality

### Fragments

Reusable XML fragments:
- UserMenu.fragment.xml
- ApprovalDialog.fragment.xml
- CommentDialog.fragment.xml
- FilterBar.fragment.xml

## Usage in Process Apps

### Backend Integration

In your process app's `backend/build.zig`:

```zig
const common_workflow = b.addModule("common-workflow", .{
    .root_source_file = b.path("../../_Common/backend/workflow/approval_engine.zig"),
});

exe.root_module.addImport("common-workflow", common_workflow);
```

In your source code:

```zig
const ApprovalEngine = @import("common-workflow").ApprovalEngine;
```

### Frontend Integration

In your UI5 app's `Component.js`:

```javascript
// Register custom component library
sap.ui.loader.config({
    paths: {
        "process.common": "../_Common/webapp"
    }
});
```

In your views:

```xml
<mvc:View
    xmlns:common="process.common.components"
    ...>
    <common:ApprovalWorkflow
        instance="{/currentInstance}"
        onApprove="handleApprove"
        onReject="handleReject" />
</mvc:View>
```

## Development Guidelines

### Adding New Common Components

1. **Identify reusable pattern** across multiple process apps
2. **Extract to _Common** with generic interface
3. **Parameterize** app-specific behavior
4. **Document** usage with examples
5. **Test** with at least two different process apps
6. **Version** if breaking changes needed

### Backward Compatibility

- Maintain API stability for shared components
- Use versioning for breaking changes
- Deprecate before removing functionality
- Provide migration guides

## Testing

```bash
# Test backend components
cd _Common/backend
zig build test

# Test UI5 components
cd _Common/webapp
npm test
```

## Examples

See these process apps for usage examples:
- `src/nApps/Process/TrialBalance`
- `src/nApps/Process/AccountsPayable`

## Best Practices

1. **Keep it generic** - Don't add app-specific logic
2. **Configuration over code** - Use config parameters
3. **Document thoroughly** - Include usage examples
4. **Test extensively** - Components are used by multiple apps
5. **Performance matters** - Common code affects all apps
6. **Error handling** - Provide clear error messages

## Contributing

When adding new common components:
1. Discuss with team first
2. Ensure it's truly reusable (2+ apps need it)
3. Write comprehensive tests
4. Update this README
5. Add usage examples

## Related Documentation

- [Process Engine Library](../../../nLang/n-c-sdk/lib/process/README.md)
- [nAgentFlow Integration](../../../serviceCore/nAgentFlow/README.md)
- [UI5 Best Practices](https://sapui5.hana.ondemand.com/)