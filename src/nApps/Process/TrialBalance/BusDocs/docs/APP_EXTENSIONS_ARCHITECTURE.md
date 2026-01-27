# Trial Balance App Extensions Architecture

## Overview

This document outlines the App Extensions architecture for integrating custom UI5 components with the Zig backend, enabling a modular, extensible system for the Trial Balance application.

## Current Architecture

### Frontend Components
- **NetworkGraphControl** - Visualizes account relationships and data flows
- **ProcessFlowControl** - Shows trial balance workflow steps
- Located in: `webapp/control/`

### Backend Services (Zig)
- **Port**: 8091
- **API Endpoints**:
  - `/api/v1/accounts` - Trial balance data
  - `/api/v1/trial-balance/calculate` - Calculation engine
  - `/api/v1/trial-balance/summary` - Summary data
  - `/api/v1/trial-balance/process-review` - DOI process
  - `/api/v1/trial-balance/narrative` - Narrative generation
- **Features**: CORS-enabled, static file serving

## Extension Architecture

### 1. Extension Points Framework

#### Frontend Extension Points

```javascript
// ExtensionPoint Interface
{
  id: string,
  type: 'component' | 'controller' | 'service',
  lifecycle: {
    beforeInit: Function,
    afterInit: Function,
    beforeRender: Function,
    afterRender: Function,
    onDestroy: Function
  },
  hooks: {
    onDataReceived: Function,
    onDataTransform: Function,
    onUserAction: Function,
    onError: Function
  }
}
```

#### Backend Extension Points

```zig
// Extension Interface
pub const Extension = struct {
    id: []const u8,
    version: []const u8,
    type: ExtensionType,
    
    // Lifecycle hooks
    init: ?*const fn(allocator: Allocator) anyerror!void,
    deinit: ?*const fn() void,
    
    // Data hooks
    onDataLoad: ?*const fn(data: []const u8) anyerror![]const u8,
    onCalculate: ?*const fn(input: []const u8) anyerror![]const u8,
    
    // API hooks
    handleRequest: ?*const fn(path: []const u8, method: []const u8) anyerror!Response,
};
```

### 2. Directory Structure

```
src/nApps/Process/TrialBalance/
├── webapp/
│   ├── extensions/
│   │   ├── ExtensionManager.js          # Main extension registry
│   │   ├── ComponentExtension.js        # Base class for component extensions
│   │   ├── manifest.json                # Extension manifest
│   │   └── examples/
│   │       ├── CustomNetworkGraph/
│   │       │   ├── Extension.js
│   │       │   ├── config.json
│   │       │   └── README.md
│   │       └── CustomProcessFlow/
│   │           ├── Extension.js
│   │           ├── config.json
│   │           └── README.md
│   ├── control/
│   │   ├── NetworkGraphControl.js       # Modified to support extensions
│   │   └── ProcessFlowControl.js        # Modified to support extensions
│   └── bridge/
│       ├── ComponentBridge.js           # UI5 ↔ Zig bridge
│       └── ExtensionBridge.js           # Extension ↔ Backend bridge
├── backend/src/
│   ├── extensions/
│   │   ├── extension_loader.zig         # Dynamic extension loading
│   │   ├── extension_registry.zig       # Extension registry
│   │   ├── extension_manager.zig        # Lifecycle management
│   │   └── interfaces/
│   │       ├── data_source.zig          # Custom data source interface
│   │       ├── calculator.zig           # Custom calculator interface
│   │       ├── transformer.zig          # Data transformer interface
│   │       └── api_handler.zig          # Custom API handler interface
│   └── api/
│       └── extension_api.zig            # Extension API endpoints
└── extensions/
    ├── README.md                         # Extension development guide
    └── examples/
        ├── advanced-variance/
        │   ├── frontend/
        │   │   └── AdvancedVarianceExtension.js
        │   ├── backend/
        │   │   └── advanced_variance.zig
        │   └── extension.json
        └── custom-visualizer/
            ├── frontend/
            │   └── CustomVisualizerExtension.js
            ├── backend/
            │   └── custom_visualizer.zig
            └── extension.json
```

### 3. Extension Manifest Format

```json
{
  "id": "trialbalance.extension.advanced-variance",
  "name": "Advanced Variance Analysis",
  "version": "1.0.0",
  "author": "Your Organization",
  "description": "Provides advanced variance calculation and visualization",
  "type": ["frontend", "backend"],
  
  "frontend": {
    "main": "frontend/AdvancedVarianceExtension.js",
    "extends": ["NetworkGraphControl"],
    "provides": {
      "visualizations": ["3d-variance", "monte-carlo"],
      "metrics": ["cashFlow", "variance", "trend"]
    },
    "requires": {
      "ui5Version": ">=1.120.0",
      "libraries": ["sap.m", "sap.f", "sap.tnt"]
    }
  },
  
  "backend": {
    "main": "backend/advanced_variance.zig",
    "endpoints": [
      {
        "path": "/api/v1/extensions/advanced-variance/calculate",
        "method": "POST",
        "handler": "calculateAdvancedVariance"
      }
    ],
    "dataLoaders": ["customCsvLoader"],
    "calculators": ["advancedVarianceCalc"],
    "hooks": {
      "calculation.pre": "preCalculationHook",
      "calculation.post": "postCalculationHook"
    }
  },
  
  "dependencies": {
    "extensions": [],
    "libraries": []
  },
  
  "configuration": {
    "schema": {
      "calculationMethod": {
        "type": "string",
        "enum": ["monteCarlo", "regression", "arima"],
        "default": "monteCarlo"
      },
      "iterations": {
        "type": "integer",
        "min": 100,
        "max": 10000,
        "default": 1000
      }
    }
  }
}
```

### 4. Extension Manager (Frontend)

```javascript
// webapp/extensions/ExtensionManager.js
sap.ui.define([
    "sap/ui/base/Object"
], function(BaseObject) {
    "use strict";

    return BaseObject.extend("trialbalance.extensions.ExtensionManager", {
        constructor: function() {
            this._extensions = new Map();
            this._hooks = new Map();
        },

        /**
         * Register an extension
         * @param {Object} oExtension - Extension configuration
         */
        registerExtension: function(oExtension) {
            if (!oExtension.id) {
                throw new Error("Extension must have an id");
            }
            
            this._extensions.set(oExtension.id, oExtension);
            this._registerHooks(oExtension);
            
            return this;
        },

        /**
         * Get registered extension
         * @param {string} sId - Extension ID
         */
        getExtension: function(sId) {
            return this._extensions.get(sId);
        },

        /**
         * Execute hook
         * @param {string} sHookName - Hook name
         * @param {Object} oContext - Hook context
         */
        executeHook: function(sHookName, oContext) {
            const aHooks = this._hooks.get(sHookName) || [];
            
            return aHooks.reduce(function(result, fnHook) {
                return fnHook.call(null, result, oContext);
            }, oContext.data);
        },

        _registerHooks: function(oExtension) {
            if (!oExtension.hooks) return;
            
            Object.keys(oExtension.hooks).forEach(function(sHook) {
                if (!this._hooks.has(sHook)) {
                    this._hooks.set(sHook, []);
                }
                this._hooks.get(sHook).push(oExtension.hooks[sHook]);
            }.bind(this));
        }
    });
});
```

### 5. Extension Registry (Backend)

```zig
// backend/src/extensions/extension_registry.zig
const std = @import("std");

pub const ExtensionType = enum {
    data_source,
    calculator,
    transformer,
    api_handler,
};

pub const Extension = struct {
    id: []const u8,
    version: []const u8,
    type: ExtensionType,
    
    // Function pointers for hooks
    init: ?*const fn(allocator: std.mem.Allocator) anyerror!void = null,
    deinit: ?*const fn() void = null,
    handleRequest: ?*const fn(path: []const u8, method: []const u8, body: []const u8) anyerror![]const u8 = null,
};

pub const ExtensionRegistry = struct {
    allocator: std.mem.Allocator,
    extensions: std.StringHashMap(*Extension),
    
    pub fn init(allocator: std.mem.Allocator) !ExtensionRegistry {
        return ExtensionRegistry{
            .allocator = allocator,
            .extensions = std.StringHashMap(*Extension).init(allocator),
        };
    }
    
    pub fn deinit(self: *ExtensionRegistry) void {
        self.extensions.deinit();
    }
    
    pub fn register(self: *ExtensionRegistry, extension: *Extension) !void {
        try self.extensions.put(extension.id, extension);
        
        // Call init hook if present
        if (extension.init) |initFn| {
            try initFn(self.allocator);
        }
    }
    
    pub fn get(self: *ExtensionRegistry, id: []const u8) ?*Extension {
        return self.extensions.get(id);
    }
    
    pub fn handleExtensionRequest(self: *ExtensionRegistry, extension_id: []const u8, path: []const u8, method: []const u8, body: []const u8) ![]const u8 {
        const extension = self.get(extension_id) orelse return error.ExtensionNotFound;
        
        if (extension.handleRequest) |handler| {
            return try handler(path, method, body);
        }
        
        return error.HandlerNotImplemented;
    }
};
```

## Integration Benefits

### 1. Modularity
- **Isolated Development**: Extensions developed independently
- **Version Control**: Each extension has its own version
- **Testing**: Extensions tested in isolation before integration

### 2. Flexibility
- **Custom Visualizations**: Add new chart types without modifying core
- **Advanced Calculations**: Plug in specialized algorithms
- **Integration**: Connect to external systems via extensions

### 3. Performance
- **Lazy Loading**: Load extensions only when needed
- **Parallel Execution**: Backend extensions can run concurrently
- **Caching**: Extension results can be cached independently

### 4. Maintainability
- **Clear Separation**: Core vs extension code clearly separated
- **Easy Updates**: Update extensions without touching core
- **Backwards Compatibility**: Extensions declare version requirements

## Example Extension Use Cases

### 1. Advanced Variance Calculator
```
Purpose: Monte Carlo simulation for variance analysis
Frontend: Custom visualization component
Backend: Statistical calculation engine
Integration: Real-time data streaming
```

### 2. 3D Balance Sheet Viewer
```
Purpose: Interactive 3D visualization of balance sheet
Frontend: WebGL-based renderer
Backend: Data transformation for 3D mapping
Integration: NetworkGraphControl extension
```

### 3. Predictive Analytics
```
Purpose: ML-based prediction of future balances
Frontend: Trend visualization
Backend: ARIMA/Prophet models in Zig
Integration: Time series data loader
```

### 4. External System Connector
```
Purpose: Real-time data from SAP S/4HANA
Frontend: Status indicators and alerts
Backend: REST/OData client
Integration: Custom data source extension
```

## Development Workflow

### Creating a New Extension

1. **Create Extension Directory**
   ```bash
   mkdir -p extensions/my-extension/{frontend,backend}
   ```

2. **Define Extension Manifest**
   ```bash
   touch extensions/my-extension/extension.json
   ```

3. **Implement Frontend Extension**
   ```javascript
   // extensions/my-extension/frontend/MyExtension.js
   sap.ui.define([
       "trialbalance/extensions/ComponentExtension"
   ], function(ComponentExtension) {
       return ComponentExtension.extend("my.Extension", {
           // Implementation
       });
   });
   ```

4. **Implement Backend Extension**
   ```zig
   // extensions/my-extension/backend/my_extension.zig
   const Extension = @import("../../../backend/src/extensions/extension_registry.zig").Extension;
   
   pub fn init(allocator: std.mem.Allocator) !void {
       // Initialization
   }
   
   pub fn handleRequest(path: []const u8, method: []const u8, body: []const u8) ![]const u8 {
       // Handle requests
   }
   ```

5. **Register Extension**
   ```javascript
   // In app initialization
   const oExtManager = new ExtensionManager();
   oExtManager.registerExtension(oMyExtension);
   ```

## Security Considerations

1. **Sandboxing**: Extensions run in isolated contexts
2. **Validation**: Extension manifests validated before loading
3. **Permissions**: Extensions declare required permissions
4. **Code Signing**: Production extensions must be signed
5. **API Rate Limiting**: Extension API calls are rate-limited

## Next Steps

1. Implement ExtensionManager (frontend)
2. Implement ExtensionRegistry (backend)
3. Create ComponentBridge for UI5 ↔ Zig communication
4. Develop example extensions
5. Create extension development guide
6. Set up extension marketplace/registry

## References

- UI5 Component Extensions: https://ui5.sap.com
- Zig Plugin Systems: https://ziglang.org/documentation/
- Extension Architecture Patterns: Enterprise Integration Patterns