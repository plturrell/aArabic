# Trial Balance App Extensions - Phase 1

## Overview

Phase 1 of the App Extensions architecture provides the foundational framework for extending the Trial Balance application with custom functionality. This enables modular development of new features without modifying core code.

## What's Included

### Frontend Components

1. **ExtensionManager.js** (`webapp/extensions/`)
   - Central registry for managing extensions
   - Hook execution with priority support
   - Dependency resolution
   - Lifecycle management

2. **ComponentExtension.js** (`webapp/extensions/`)
   - Base class for creating component extensions
   - Standardized extension interface
   - Component wrapping and enhancement
   - Built-in lifecycle hooks

3. **EnhancedNetworkGraph.js** (`webapp/extensions/examples/`)
   - Example extension demonstrating all features
   - Data transformation
   - Custom rendering
   - Backend integration

### Backend Components

1. **extension_registry.zig** (`backend/src/extensions/`)
   - Extension registration and lifecycle
   - Hook execution by type
   - Priority-based ordering
   - Statistics and monitoring

2. **enhanced_metrics.zig** (`backend/src/extensions/examples/`)
   - Example backend extension
   - Custom API endpoints
   - Metric calculations
   - Frontend integration

## Quick Start

### 1. Using the Example Extension

#### Frontend Setup

```javascript
// In your Component.js or controller
sap.ui.define([
    "trialbalance/extensions/ExtensionManager",
    "trialbalance/extensions/examples/EnhancedNetworkGraph"
], function(ExtensionManager, EnhancedNetworkGraph) {
    
    // Create extension manager
    const oExtManager = new ExtensionManager();
    
    // Create and register extension
    const oExtension = new EnhancedNetworkGraph();
    oExtManager.registerExtension(oExtension.getExtensionConfig());
    
    // Initialize extensions
    oExtManager.initialize().then(function() {
        console.log("Extensions initialized");
    });
    
    // Use extension hooks
    const transformedData = oExtManager.executeHook("data.received", {
        data: myData,
        metadata: { source: "backend" }
    });
});
```

#### Backend Setup

```zig
// In your main.zig
const ExtensionRegistry = @import("extensions/extension_registry.zig").ExtensionRegistry;
const EnhancedMetrics = @import("extensions/examples/enhanced_metrics.zig").EnhancedMetricsExtension;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    // Create extension registry
    var registry = try ExtensionRegistry.init(allocator);
    defer registry.deinit();
    
    // Register enhanced metrics extension
    const metrics_ext = try EnhancedMetrics.create(allocator);
    try registry.register(metrics_ext);
    
    // Initialize all extensions
    try registry.initializeExtensions();
    
    // Use extensions in API handlers
    // ... (see integration example below)
}
```

### 2. Creating Your Own Extension

#### Frontend Extension

```javascript
sap.ui.define([
    "trialbalance/extensions/ComponentExtension"
], function(ComponentExtension) {
    
    return ComponentExtension.extend("my.custom.Extension", {
        
        constructor: function() {
            ComponentExtension.call(this);
            
            this.setId("my-custom-extension");
            this.setName("My Custom Extension");
            this.setVersion("1.0.0");
            this.setTargetComponents(["trialbalance.control.NetworkGraphControl"]);
        },
        
        init: function() {
            // Initialize your extension
            console.log("My extension initialized");
        },
        
        onDataReceived: function(vData, mContext) {
            // Transform data
            return vData.map(item => ({
                ...item,
                customField: this._calculateCustomValue(item)
            }));
        },
        
        onAfterRender: function(oComponent) {
            // Add custom behavior after rendering
            this._addCustomFeatures(oComponent);
        },
        
        _calculateCustomValue: function(item) {
            // Your custom logic
            return item.value * 2;
        },
        
        _addCustomFeatures: function(oComponent) {
            // Add visual enhancements, event handlers, etc.
        }
    });
});
```

#### Backend Extension

```zig
const std = @import("std");
const Extension = @import("../extension_registry.zig").Extension;
const ExtensionType = @import("../extension_registry.zig").ExtensionType;

pub fn create(allocator: std.mem.Allocator) !*Extension {
    const ext = try allocator.create(Extension);
    
    ext.* = Extension.init(
        "my-custom-extension",
        "My Custom Extension",
        "1.0.0",
        .calculator,
    );
    
    ext.hooks.handleRequest = handleRequest;
    
    return ext;
}

fn handleRequest(
    allocator: std.mem.Allocator,
    path: []const u8,
    method: []const u8,
    body: []const u8,
) ![]const u8 {
    // Handle your custom endpoints
    if (std.mem.endsWith(u8, path, "/my-endpoint")) {
        return try allocator.dupe(u8, 
            \\{"result": "success"}
        );
    }
    
    return error.EndpointNotFound;
}
```

## Extension Hooks

### Frontend Hooks

| Hook Name | Parameters | Description |
|-----------|------------|-------------|
| `component.beforeExtend` | `oComponent` | Before extending a component |
| `component.afterExtend` | `oComponent` | After extending a component |
| `data.received` | `vData, mContext` | When data is received |
| `component.beforeRender` | `vData, mContext` | Before component renders |
| `component.afterRender` | `oComponent` | After component renders |
| `user.action` | `sAction, mParams` | User interaction |
| `component.error` | `oError, mContext` | Error occurred |

### Backend Hooks

| Hook Type | Function | Description |
|-----------|----------|-------------|
| `init` | `fn(allocator) !void` | Initialize extension |
| `deinit` | `fn() void` | Cleanup extension |
| `onDataLoad` | `fn(allocator, data) ![]u8` | Transform loaded data |
| `onCalculate` | `fn(allocator, input) ![]u8` | Perform calculations |
| `handleRequest` | `fn(allocator, path, method, body) ![]u8` | Handle HTTP requests |

## Extension API Endpoints

Extensions can expose custom API endpoints:

```
GET  /api/v1/extensions/{extension-id}/config
GET  /api/v1/extensions/{extension-id}/status
POST /api/v1/extensions/{extension-id}/action
GET  /api/v1/extensions/{extension-id}/{custom-path}
```

Example with enhanced-metrics extension:

```javascript
// Frontend call
fetch('/api/v1/extensions/enhanced-metrics/config')
    .then(r => r.json())
    .then(config => console.log(config));

fetch('/api/v1/extensions/enhanced-metrics/node/ACC001')
    .then(r => r.json())
    .then(details => console.log(details));
```

## Testing Extensions

### Frontend Test

```javascript
QUnit.test("Extension transforms data correctly", function(assert) {
    const oExtension = new MyExtension();
    const input = [{ value: 10 }];
    const result = oExtension.onDataReceived(input, {});
    
    assert.equal(result[0].customField, 20, "Custom field calculated");
});
```

### Backend Test

```zig
test "Extension handles request" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const response = try handleRequest(allocator, "/test", "GET", "");
    defer allocator.free(response);
    
    try testing.expect(std.mem.indexOf(u8, response, "success") != null);
}
```

## Best Practices

1. **Naming Conventions**
   - Use kebab-case for extension IDs: `enhanced-network-graph`
   - Use PascalCase for class names: `EnhancedNetworkGraph`

2. **Error Handling**
   - Always implement `onError` hook
   - Provide graceful degradation
   - Log errors appropriately

3. **Performance**
   - Use caching for expensive operations
   - Set appropriate priorities
   - Avoid blocking operations in hooks

4. **Dependencies**
   - Declare dependencies explicitly
   - Check for required extensions
   - Handle missing dependencies gracefully

5. **Documentation**
   - Document all hooks
   - Provide usage examples
   - List configuration options

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│          Frontend (UI5)                 │
├─────────────────────────────────────────┤
│  ExtensionManager                       │
│  ├─ ComponentExtension (base)           │
│  ├─ EnhancedNetworkGraph (example)      │
│  └─ Custom Extensions                   │
└────────────────┬────────────────────────┘
                 │ HTTP/REST
┌────────────────▼────────────────────────┐
│          Backend (Zig)                  │
├─────────────────────────────────────────┤
│  ExtensionRegistry                      │
│  ├─ enhanced_metrics (example)          │
│  └─ Custom Extensions                   │
└─────────────────────────────────────────┘
```

## Extension Lifecycle

```
1. Register    → Extension added to manager/registry
2. Initialize  → init() hook called
3. Use         → Hooks executed as needed
4. Disable     → Extension temporarily disabled
5. Enable      → Extension re-enabled
6. Unregister  → Extension removed
7. Destroy     → deinit() hook called, cleanup
```

## Next Steps (Phase 2+)

- [ ] Extension marketplace/discovery
- [ ] Hot-reload capabilities
- [ ] Extension sandboxing
- [ ] Advanced dependency management
- [ ] Extension versioning and migration
- [ ] Performance monitoring
- [ ] Extension debugging tools

## Support

For questions or issues:
1. Check the [Architecture Document](../BusDocs/docs/APP_EXTENSIONS_ARCHITECTURE.md)
2. Review example extensions in `examples/`
3. Run tests: `zig test` (backend) or QUnit (frontend)

## License

See main project LICENSE file.