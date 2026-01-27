sap.ui.define([
    "trialbalance/extensions/ComponentExtension",
    "sap/base/Log"
], function(ComponentExtension, Log) {
    "use strict";

    /**
     * WorkflowBuilder Extension
     * Wraps the TypeScript WorkflowBuilder component (D3.js based).
     * Provides drag-and-drop workflow creation capabilities.
     * 
     * Features:
     * - Visual drag-and-drop workflow editing
     * - Node templates for different workflow steps
     * - Connection validation
     * - Workflow import/export
     * - Zoom and pan support
     * 
     * @class
     * @extends trialbalance.extensions.ComponentExtension
     */
    return ComponentExtension.extend("trialbalance.extensions.components.WorkflowBuilderExtension", {
        
        constructor: function(mSettings) {
            ComponentExtension.call(this, mSettings);
            
            this.setId("workflow-builder-core");
            this.setName("Workflow Builder Core");
            this.setVersion("1.0.0");
            this.setTargetComponents(["trialbalance.control.WorkflowBuilderControl"]);
            this.setPriority(100);
            
            this._workflowBuilder = null;
            this._config = null;
            this._workflowCache = null;
            this._eventHandlers = new Map();
            this._extensionManager = null;
        },

        init: function() {
            Log.info("Initializing WorkflowBuilder Extension", "trialbalance.extensions.components.WorkflowBuilderExtension");
            
            return Promise.all([
                this._loadWorkflowBuilderScript(),
                this._loadConfiguration()
            ]).then(function() {
                Log.info("WorkflowBuilder Extension initialized", "trialbalance.extensions.components.WorkflowBuilderExtension");
            }.bind(this));
        },

        setExtensionManager: function(oManager) {
            this._extensionManager = oManager;
        },

        onBeforeExtend: function(oComponent) {
            this._ui5Control = oComponent;
            oComponent.addStyleClass("workflow-builder-extended");
        },

        onAfterExtend: function(oComponent) {
            setTimeout(function() {
                this._initializeWorkflowBuilder(oComponent);
            }.bind(this), 100);
        },

        onDataReceived: function(vData, mContext) {
            if (!vData) return vData;
            
            if (this._extensionManager) {
                vData = this._extensionManager.executeHook("workflowbuilder.data.transform", {
                    data: vData,
                    metadata: mContext
                });
            }
            
            if (vData && vData.nodes && vData.connections) {
                this._workflowCache = vData;
            }
            
            return vData;
        },

        onUserAction: function(sAction, mParams) {
            Log.debug("User action: " + sAction, "trialbalance.extensions.components.WorkflowBuilderExtension");
            
            if (this._extensionManager) {
                const bContinue = this._extensionManager.executeHook("workflowbuilder.action.before", {
                    action: sAction,
                    params: mParams
                });
                if (bContinue === false) return false;
            }
            
            switch (sAction) {
                case "nodeSelect":
                    this._handleNodeSelect(mParams);
                    break;
                case "workflowChange":
                    this._handleWorkflowChange(mParams);
                    break;
            }
            
            if (this._extensionManager) {
                this._extensionManager.executeHook("workflowbuilder.action.after", { action: sAction, params: mParams });
            }
            
            return true;
        },

        // ========== Public API ==========

        getWorkflowBuilder: function() {
            return this._workflowBuilder;
        },

        addNode: function(sType, nX, nY) {
            if (!this._workflowBuilder) {
                Log.warning("WorkflowBuilder not initialized", "trialbalance.extensions.components.WorkflowBuilderExtension");
                return null;
            }
            
            if (this._extensionManager) {
                const result = this._extensionManager.executeHook("workflowbuilder.node.beforeAdd", {
                    type: sType, x: nX, y: nY
                });
                if (result === false) return null;
            }
            
            const node = this._workflowBuilder.addNode(sType, nX, nY);
            
            if (this._extensionManager) {
                this._extensionManager.executeHook("workflowbuilder.node.added", { node: node });
            }
            
            return node;
        },

        connect: function(sSourceNodeId, sSourcePortId, sTargetNodeId, sTargetPortId) {
            if (!this._workflowBuilder) return null;
            return this._workflowBuilder.connect(sSourceNodeId, sSourcePortId, sTargetNodeId, sTargetPortId);
        },

        selectNode: function(sNodeId) {
            if (this._workflowBuilder) {
                this._workflowBuilder.selectNode(sNodeId);
            }
        },

        deleteSelected: function() {
            if (this._workflowBuilder) {
                this._workflowBuilder.deleteSelected();
            }
        },

        validate: function() {
            if (!this._workflowBuilder) {
                return { valid: false, errors: [{ type: "not_initialized", message: "WorkflowBuilder not initialized" }], warnings: [] };
            }
            
            const result = this._workflowBuilder.validate();
            
            if (this._extensionManager) {
                this._extensionManager.executeHook("workflowbuilder.validated", { result: result });
            }
            
            return result;
        },

        exportWorkflow: function() {
            return this._workflowBuilder ? this._workflowBuilder.exportWorkflow() : null;
        },

        loadWorkflow: function(oWorkflow) {
            if (!this._workflowBuilder) {
                this._workflowCache = oWorkflow;
                return;
            }
            
            if (this._extensionManager) {
                oWorkflow = this._extensionManager.executeHook("workflowbuilder.workflow.beforeLoad", {
                    data: oWorkflow, metadata: {}
                }) || oWorkflow;
            }
            
            this._workflowBuilder.loadWorkflow(oWorkflow);
        },

        clear: function() {
            if (this._workflowBuilder) {
                this._workflowBuilder.clear();
            }
        },

        getNodeTemplates: function() {
            return this._workflowBuilder ? this._workflowBuilder.getNodeTemplates() : [];
        },

        on: function(sEvent, fnHandler) {
            if (this._workflowBuilder) {
                if (sEvent === "change") this._workflowBuilder.onChange(fnHandler);
                else if (sEvent === "select") this._workflowBuilder.onSelect(fnHandler);
            }
            if (!this._eventHandlers.has(sEvent)) this._eventHandlers.set(sEvent, []);
            this._eventHandlers.get(sEvent).push(fnHandler);
        },

        // ========== Private Methods ==========

        _loadWorkflowBuilderScript: function() {
            return new Promise(function(resolve, reject) {
                if (window.WorkflowBuilder) {
                    resolve();
                    return;
                }
                
                const sScriptPath = sap.ui.require.toUrl("trialbalance/components/dist/WorkflowBuilder/WorkflowBuilder.min.js");
                
                const script = document.createElement("script");
                script.type = "module";
                script.textContent = `
                    import { WorkflowBuilder } from '${sScriptPath}';
                    window.WorkflowBuilder = WorkflowBuilder;
                    window.dispatchEvent(new CustomEvent('workflowbuilder-loaded'));
                `;
                document.head.appendChild(script);
                
                window.addEventListener('workflowbuilder-loaded', function() {
                    Log.info("WorkflowBuilder script loaded", "trialbalance.extensions.components.WorkflowBuilderExtension");
                    resolve();
                }, { once: true });
                
                setTimeout(function() {
                    if (!window.WorkflowBuilder) reject(new Error("WorkflowBuilder script load timeout"));
                }, 10000);
            });
        },

        _loadConfiguration: function() {
            return fetch('/api/v1/extensions/workflow-builder-core/config')
                .then(function(response) { return response.json(); })
                .then(function(config) { this._config = config; }.bind(this))
                .catch(function() {
                    this._config = { showGrid: true, snapToGrid: true, gridSize: 20 };
                }.bind(this));
        },

        _initializeWorkflowBuilder: function(oComponent) {
            if (!window.WorkflowBuilder) {
                Log.error("WorkflowBuilder not loaded", "trialbalance.extensions.components.WorkflowBuilderExtension");
                return;
            }
            
            const domRef = oComponent.getDomRef();
            if (!domRef) return;
            
            const canvas = domRef.querySelector(".workflowbuilder-canvas") || domRef;
            
            try {
                this._workflowBuilder = new window.WorkflowBuilder(canvas, this._config);
                
                this._workflowBuilder.onChange(function(workflow) {
                    this.onUserAction("workflowChange", { workflow: workflow });
                }.bind(this));
                
                this._workflowBuilder.onSelect(function(node) {
                    this.onUserAction("nodeSelect", { node: node });
                }.bind(this));
                
                this._wireEventHandlers();
                
                if (this._workflowCache) {
                    this._workflowBuilder.loadWorkflow(this._workflowCache);
                }
                
                Log.info("WorkflowBuilder instance created", "trialbalance.extensions.components.WorkflowBuilderExtension");
                
            } catch (e) {
                Log.error("Failed to create WorkflowBuilder", e.message, "trialbalance.extensions.components.WorkflowBuilderExtension");
            }
        },

        _wireEventHandlers: function() {
            if (!this._workflowBuilder) return;
            
            this._eventHandlers.forEach(function(handlers, event) {
                handlers.forEach(function(handler) {
                    if (event === "change") this._workflowBuilder.onChange(handler);
                    else if (event === "select") this._workflowBuilder.onSelect(handler);
                }.bind(this));
            }.bind(this));
        },

        _handleNodeSelect: function(mParams) {
            if (this._extensionManager) {
                this._extensionManager.executeHook("workflowbuilder.node.selected", mParams);
            }
        },

        _handleWorkflowChange: function(mParams) {
            if (this._extensionManager) {
                this._extensionManager.executeHook("workflowbuilder.workflow.changed", mParams);
            }
        },

        destroy: function() {
            Log.info("Destroying WorkflowBuilder Extension", "trialbalance.extensions.components.WorkflowBuilderExtension");
            
            if (this._workflowBuilder) {
                this._workflowBuilder.destroy();
                this._workflowBuilder = null;
            }
            
            this._eventHandlers.clear();
            this._workflowCache = null;
            this._config = null;
            this._extensionManager = null;
            
            ComponentExtension.prototype.destroy.call(this);
        }
    });
});