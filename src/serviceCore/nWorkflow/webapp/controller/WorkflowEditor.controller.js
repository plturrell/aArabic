sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/routing/History",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/m/Input",
    "sap/m/TextArea",
    "sap/m/Select",
    "sap/m/CheckBox",
    "sap/m/Label",
    "sap/ui/core/Item",
    "serviceCore/nWorkflow/model/NodeTypes",
    "serviceCore/nWorkflow/util/WorkflowCanvas"
], function (Controller, History, JSONModel, MessageToast, MessageBox, Input, TextArea, Select, CheckBox, Label, Item, NodeTypes, WorkflowCanvas) {
    "use strict";

    return Controller.extend("serviceCore.nWorkflow.controller.WorkflowEditor", {

        onInit: function () {
            // Initialize editor model
            var oEditorModel = new JSONModel({
                selectedNode: null,
                zoomLevel: 100,
                canUndo: false,
                canRedo: false
            });
            this.getView().setModel(oEditorModel, "editor");

            // Router
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("workflowEditor").attachPatternMatched(this._onRouteMatched, this);

            // Canvas will be initialized after view is rendered
            this._canvas = null;
            this._jointjsLoaded = false;
        },

        onAfterRendering: function () {
            if (!this._jointjsLoaded) {
                this._loadJointJS();
            }
        },

        /**
         * Load JointJS library from CDN
         */
        _loadJointJS: function () {
            var that = this;

            // Check if already loaded
            if (window.joint) {
                this._jointjsLoaded = true;
                this._initCanvas();
                return;
            }

            // Load JointJS CSS
            var link = document.createElement("link");
            link.rel = "stylesheet";
            link.href = "https://cdnjs.cloudflare.com/ajax/libs/jointjs/3.7.7/joint.min.css";
            document.head.appendChild(link);

            // Load JointJS JS
            var script = document.createElement("script");
            script.src = "https://cdnjs.cloudflare.com/ajax/libs/jointjs/3.7.7/joint.min.js";
            script.onload = function () {
                that._jointjsLoaded = true;
                that._initCanvas();
            };
            script.onerror = function () {
                MessageBox.error("Failed to load JointJS library");
            };
            document.head.appendChild(script);
        },

        /**
         * Initialize the workflow canvas
         */
        _initCanvas: function () {
            var that = this;
            var oContainer = document.getElementById("workflow-canvas");

            if (!oContainer) {
                // Retry after a short delay if container not ready
                setTimeout(function () {
                    that._initCanvas();
                }, 100);
                return;
            }

            this._canvas = new WorkflowCanvas({
                container: oContainer,
                onNodeSelect: function (sNodeId, sNodeType, oNodeData) {
                    that._onNodeSelected(sNodeId, sNodeType, oNodeData);
                },
                onNodeDeselect: function () {
                    that._onNodeDeselected();
                },
                onConnectionCreate: function (oConnection) {
                    that._onConnectionCreated(oConnection);
                }
            });

            this._canvas.init();
            this._setupDragAndDrop();

            // Load existing workflow if any
            if (this._pendingWorkflow) {
                this._canvas.loadWorkflowJSON(this._pendingWorkflow);
                this._pendingWorkflow = null;
            }
        },

        /**
         * Setup drag and drop from palette to canvas
         */
        _setupDragAndDrop: function () {
            var that = this;
            var oCanvasElement = document.getElementById("workflow-canvas");

            // Make palette items draggable
            var aDraggableItems = document.querySelectorAll(".draggableNode");
            aDraggableItems.forEach(function (item) {
                item.setAttribute("draggable", "true");
                item.addEventListener("dragstart", function (e) {
                    var sNodeType = this.getAttribute("data-nodetype") ||
                                   this.querySelector("[data-nodetype]")?.getAttribute("data-nodetype") ||
                                   this.closest("[data-nodetype]")?.getAttribute("data-nodetype");
                    // Try to get from custom data
                    if (!sNodeType) {
                        var oControl = sap.ui.getCore().byId(this.id);
                        if (oControl && oControl.data) {
                            sNodeType = oControl.data("nodeType");
                        }
                    }
                    e.dataTransfer.setData("nodeType", sNodeType || "task");
                    e.dataTransfer.effectAllowed = "copy";
                });
            });

            // Handle drop on canvas
            if (oCanvasElement) {
                oCanvasElement.addEventListener("dragover", function (e) {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = "copy";
                });

                oCanvasElement.addEventListener("drop", function (e) {
                    e.preventDefault();
                    var sNodeType = e.dataTransfer.getData("nodeType");
                    if (sNodeType && that._canvas) {
                        var rect = oCanvasElement.getBoundingClientRect();
                        var x = e.clientX - rect.left;
                        var y = e.clientY - rect.top;
                        that._canvas.createNode(sNodeType, { x: x, y: y });
                    }
                });
            }
        },

        _onRouteMatched: function (oEvent) {
            var sWorkflowId = oEvent.getParameter("arguments").workflowId;
            this._workflowId = sWorkflowId;

            if (sWorkflowId === "new") {
                this.getOwnerComponent().getModel().setProperty("/currentWorkflow", {
                    id: null,
                    name: "New Workflow",
                    nodes: [],
                    connections: []
                });
                if (this._canvas) {
                    this._canvas.clear();
                }
            } else {
                this._loadWorkflow(sWorkflowId);
            }
        },

        _loadWorkflow: function (sWorkflowId) {
            var oModel = this.getOwnerComponent().getModel();
            var aWorkflows = oModel.getProperty("/workflows");
            var oWorkflow = aWorkflows.find(function (w) {
                return w.id === sWorkflowId;
            });

            if (oWorkflow) {
                oModel.setProperty("/currentWorkflow", oWorkflow);
                if (this._canvas) {
                    this._canvas.loadWorkflowJSON(oWorkflow);
                } else {
                    this._pendingWorkflow = oWorkflow;
                }
            }
        },

        /**
         * Handle node selection
         */
        _onNodeSelected: function (sNodeId, sNodeType, oNodeData) {
            var oNodeDef = NodeTypes.getNodeType(sNodeType);
            var oEditorModel = this.getView().getModel("editor");

            oEditorModel.setProperty("/selectedNode", {
                id: sNodeId,
                type: sNodeType,
                typeName: oNodeDef ? oNodeDef.name : sNodeType,
                data: oNodeData || {}
            });

            this._buildPropertiesForm(sNodeType, oNodeData);
        },

        /**
         * Handle node deselection
         */
        _onNodeDeselected: function () {
            this.getView().getModel("editor").setProperty("/selectedNode", null);
            this._clearPropertiesForm();
        },

        /**
         * Handle connection created
         */
        _onConnectionCreated: function (oConnection) {
            // Connection tracking if needed
        },

        /**
         * Build dynamic properties form based on node type
         */
        _buildPropertiesForm: function (sNodeType, oNodeData) {
            var oContainer = this.byId("dynamicPropertiesContainer");
            if (!oContainer) return;

            oContainer.destroyItems();

            var oNodeDef = NodeTypes.getNodeType(sNodeType);
            if (!oNodeDef || !oNodeDef.properties) return;

            var that = this;
            oNodeData = oNodeData || {};

            oNodeDef.properties.forEach(function (prop) {
                // Add label
                oContainer.addItem(new Label({ text: prop.name }));

                // Add input based on type
                var oInput;
                switch (prop.type) {
                    case "select":
                        oInput = new Select({
                            selectedKey: oNodeData[prop.id] || prop.default,
                            change: function (oEvent) {
                                that._updateNodeProperty(prop.id, oEvent.getParameter("selectedItem").getKey());
                            }
                        });
                        (prop.options || []).forEach(function (option) {
                            oInput.addItem(new Item({ key: option, text: option }));
                        });
                        break;
                    case "text":
                    case "code":
                    case "expression":
                        oInput = new TextArea({
                            value: oNodeData[prop.id] || prop.default,
                            rows: 4,
                            width: "100%",
                            change: function (oEvent) {
                                that._updateNodeProperty(prop.id, oEvent.getParameter("value"));
                            }
                        });
                        break;
                    case "number":
                        oInput = new Input({
                            type: "Number",
                            value: oNodeData[prop.id] !== undefined ? oNodeData[prop.id] : prop.default,
                            change: function (oEvent) {
                                that._updateNodeProperty(prop.id, parseFloat(oEvent.getParameter("value")));
                            }
                        });
                        break;
                    case "boolean":
                        oInput = new CheckBox({
                            selected: oNodeData[prop.id] !== undefined ? oNodeData[prop.id] : prop.default,
                            select: function (oEvent) {
                                that._updateNodeProperty(prop.id, oEvent.getParameter("selected"));
                            }
                        });
                        break;
                    default:
                        oInput = new Input({
                            value: oNodeData[prop.id] || prop.default,
                            width: "100%",
                            change: function (oEvent) {
                                that._updateNodeProperty(prop.id, oEvent.getParameter("value"));
                            }
                        });
                }
                oContainer.addItem(oInput);
            });
        },

        /**
         * Clear dynamic properties form
         */
        _clearPropertiesForm: function () {
            var oContainer = this.byId("dynamicPropertiesContainer");
            if (oContainer) {
                oContainer.destroyItems();
            }
        },

        /**
         * Update a property on the selected node
         */
        _updateNodeProperty: function (sPropertyId, vValue) {
            var oEditorModel = this.getView().getModel("editor");
            var oSelectedNode = oEditorModel.getProperty("/selectedNode");

            if (oSelectedNode && this._canvas) {
                var oNodeData = oSelectedNode.data || {};
                oNodeData[sPropertyId] = vValue;
                oEditorModel.setProperty("/selectedNode/data", oNodeData);
                this._canvas.updateNodeData(oSelectedNode.id, oNodeData);
            }
        },

        /**
         * Handle click on node type in palette
         */
        onNodeTypePress: function (oEvent) {
            var oSource = oEvent.getSource();
            var sNodeType = oSource.data("nodeType");

            if (sNodeType && this._canvas) {
                // Add node at center of visible canvas area
                this._canvas.createNode(sNodeType, { x: 300, y: 200 });
            }
        },

        onNavBack: function () {
            var oHistory = History.getInstance();
            var sPreviousHash = oHistory.getPreviousHash();

            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                this.getOwnerComponent().getRouter().navTo("dashboard", {}, true);
            }
        },

        onSavePress: function () {
            if (this._canvas) {
                var oWorkflowData = this._canvas.getWorkflowJSON();
                var oModel = this.getOwnerComponent().getModel();
                var oCurrentWorkflow = oModel.getProperty("/currentWorkflow") || {};

                oCurrentWorkflow.nodes = oWorkflowData.nodes;
                oCurrentWorkflow.connections = oWorkflowData.connections;
                oModel.setProperty("/currentWorkflow", oCurrentWorkflow);

                // In production, save to backend here
                console.log("Workflow saved:", JSON.stringify(oWorkflowData, null, 2));
            }
            MessageToast.show(this.getView().getModel("i18n").getResourceBundle().getText("workflowSaved"));
        },

        onRunPress: function () {
            MessageToast.show(this.getView().getModel("i18n").getResourceBundle().getText("workflowStarted"));
        },

        onUndoPress: function () {
            if (this._canvas) {
                this._canvas.undo();
            }
        },

        onRedoPress: function () {
            if (this._canvas) {
                this._canvas.redo();
            }
        },

        onZoomInPress: function () {
            if (this._canvas) {
                this._canvas.zoom(1.2);
                this._updateZoomLevel();
            }
        },

        onZoomOutPress: function () {
            if (this._canvas) {
                this._canvas.zoom(0.8);
                this._updateZoomLevel();
            }
        },

        onFitContentPress: function () {
            if (this._canvas) {
                this._canvas.fitContent();
                this._updateZoomLevel();
            }
        },

        _updateZoomLevel: function () {
            if (this._canvas) {
                var fZoom = this._canvas.getZoom();
                this.getView().getModel("editor").setProperty("/zoomLevel", Math.round(fZoom * 100));
            }
        },

        onDeletePress: function () {
            if (this._canvas) {
                this._canvas.deleteSelected();
            }
        },

        onExit: function () {
            if (this._canvas) {
                this._canvas.destroy();
                this._canvas = null;
            }
        }
    });
});