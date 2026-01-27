/**
 * ============================================================================
 * Checklist Controller
 * Maker/Checker workflow management for trial balance close process
 * ============================================================================
 *
 * [CODE:file=Checklist.controller.js]
 * [CODE:module=controller]
 * [CODE:language=javascript]
 *
 * [ODPS:product=checklist-items]
 *
 * [DOI:controls=MKR-CHK-001,MKR-CHK-002]
 *
 * [PETRI:stages=S01,S02,S03,S04,S05,S06,S07]
 * [PETRI:process=TB_PROCESS_petrinet.pnml]
 *
 * [VIEW:binding=Checklist.view.xml]
 *
 * [API:consumes=/api/v1/workflow]
 * [API:consumes=/api/v1/workflow/checklist]
 * [API:consumes=/api/v1/workflow/submit]
 * [API:consumes=/api/v1/workflow/approve]
 *
 * [RELATION:uses=CODE:ApiService.js]
 * [RELATION:calls=CODE:maker_checker.zig]
 * [RELATION:calls=CODE:odps_petrinet_bridge.zig]
 *
 * This controller implements the maker/checker workflow for trial balance
 * close process, tracking checklist items and approval status.
 */
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "trialbalance/service/ApiService"
], function (Controller, JSONModel, MessageToast, MessageBox, ApiService) {
    "use strict";

    return Controller.extend("trialbalance.controller.Checklist", {

        /**
         * Controller initialization
         */
        onInit: function () {
            // Initialize API service
            this._oApiService = new ApiService();
            
            // Create view model
            var oViewModel = new JSONModel({
                busy: true,
                workflowStatus: {
                    currentStage: "S01",
                    stageName: "Data Loading",
                    status: "in_progress",
                    makerUser: null,
                    checkerUser: null,
                    submittedAt: null,
                    approvedAt: null
                },
                stages: [
                    { id: "S01", name: "Data Loading", status: "completed", icon: "sap-icon://download" },
                    { id: "S02", name: "Data Validation", status: "completed", icon: "sap-icon://accept" },
                    { id: "S03", name: "FX Conversion", status: "completed", icon: "sap-icon://money-bills" },
                    { id: "S04", name: "TB Calculation", status: "in_progress", icon: "sap-icon://calculator" },
                    { id: "S05", name: "Variance Analysis", status: "pending", icon: "sap-icon://compare" },
                    { id: "S06", name: "Commentary", status: "pending", icon: "sap-icon://edit" },
                    { id: "S07", name: "Approval", status: "pending", icon: "sap-icon://approvals" }
                ],
                checklistItems: [],
                summary: {
                    total: 0,
                    completed: 0,
                    inProgress: 0,
                    pending: 0,
                    completionPercent: 0
                },
                canSubmit: false,
                canApprove: false,
                lastUpdated: null
            });
            this.getView().setModel(oViewModel, "view");
            
            // Load data when route is matched
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("checklist").attachPatternMatched(this._onRouteMatched, this);
        },

        /**
         * Route matched handler
         * @private
         */
        _onRouteMatched: function () {
            this._loadWorkflowData();
        },

        /**
         * Load workflow and checklist data from API
         * @private
         */
        _loadWorkflowData: function () {
            var that = this;
            var oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/busy", true);
            
            Promise.all([
                this._oApiService.getWorkflowStatus(),
                this._oApiService.getChecklistItems()
            ]).then(function (aResults) {
                var oWorkflow = aResults[0] || {};
                var aItems = aResults[1] || [];
                
                // Update workflow status
                if (oWorkflow.current_stage) {
                    oViewModel.setProperty("/workflowStatus", {
                        currentStage: oWorkflow.current_stage,
                        stageName: oWorkflow.stage_name || "",
                        status: oWorkflow.status || "in_progress",
                        makerUser: oWorkflow.maker_user,
                        checkerUser: oWorkflow.checker_user,
                        submittedAt: oWorkflow.submitted_at ? new Date(oWorkflow.submitted_at) : null,
                        approvedAt: oWorkflow.approved_at ? new Date(oWorkflow.approved_at) : null
                    });
                }
                
                // Transform checklist items
                var aTransformed = aItems.map(function (item) {
                    return {
                        id: item.id,
                        stageId: item.stage_id,
                        name: item.name,
                        description: item.description || "",
                        status: item.status || "pending",
                        assignee: item.assignee,
                        dueDate: item.due_date ? new Date(item.due_date) : null,
                        completedAt: item.completed_at ? new Date(item.completed_at) : null,
                        completedBy: item.completed_by,
                        notes: item.notes || ""
                    };
                });
                
                oViewModel.setProperty("/checklistItems", aTransformed);
                
                // Calculate summary
                var iCompleted = aTransformed.filter(function(i) { return i.status === "completed"; }).length;
                var iInProgress = aTransformed.filter(function(i) { return i.status === "in_progress"; }).length;
                var iPending = aTransformed.filter(function(i) { return i.status === "pending"; }).length;
                
                oViewModel.setProperty("/summary", {
                    total: aTransformed.length,
                    completed: iCompleted,
                    inProgress: iInProgress,
                    pending: iPending,
                    completionPercent: aTransformed.length > 0 ? (iCompleted / aTransformed.length * 100) : 0
                });
                
                // Determine if can submit/approve
                oViewModel.setProperty("/canSubmit", iCompleted === aTransformed.length && aTransformed.length > 0);
                oViewModel.setProperty("/canApprove", oWorkflow.status === "pending_approval");
                
                oViewModel.setProperty("/lastUpdated", new Date());
                oViewModel.setProperty("/busy", false);
            }).catch(function (oError) {
                // Use fallback static data
                oViewModel.setProperty("/checklistItems", that._getStaticChecklistItems());
                oViewModel.setProperty("/busy", false);
            });
        },

        /**
         * Get static checklist items for offline mode
         * @private
         */
        _getStaticChecklistItems: function () {
            return [
                { id: "1", stageId: "S01", name: "Load ACDOCA data", description: "Import journal entries from S/4HANA", status: "completed" },
                { id: "2", stageId: "S01", name: "Load exchange rates", description: "Import FX rates from Treasury", status: "completed" },
                { id: "3", stageId: "S02", name: "Run TB001-TB006 validations", description: "Execute trial balance validation rules", status: "completed" },
                { id: "4", stageId: "S03", name: "Apply FX conversion", description: "Convert local currency to group currency", status: "completed" },
                { id: "5", stageId: "S04", name: "Calculate trial balance", description: "Compute opening/closing balances", status: "in_progress" },
                { id: "6", stageId: "S05", name: "Analyze variances", description: "Compare with prior period", status: "pending" },
                { id: "7", stageId: "S05", name: "Flag material variances", description: "Apply DOI thresholds", status: "pending" },
                { id: "8", stageId: "S06", name: "Add commentary", description: "Explain material variances", status: "pending" },
                { id: "9", stageId: "S06", name: "Verify 90% coverage", description: "Meet commentary requirement", status: "pending" },
                { id: "10", stageId: "S07", name: "Checker review", description: "Review and approve", status: "pending" }
            ];
        },

        /**
         * Get status icon
         */
        getStatusIcon: function (sStatus) {
            var oIcons = {
                "completed": "sap-icon://accept",
                "in_progress": "sap-icon://process",
                "pending": "sap-icon://pending"
            };
            return oIcons[sStatus] || "sap-icon://pending";
        },

        /**
         * Get status state
         */
        getStatusState: function (sStatus) {
            var oStates = {
                "completed": "Success",
                "in_progress": "Warning",
                "pending": "None"
            };
            return oStates[sStatus] || "None";
        },

        /**
         * Handle checklist item completion toggle
         */
        onToggleComplete: function (oEvent) {
            var that = this;
            var oSource = oEvent.getSource();
            var oContext = oSource.getBindingContext("view");
            var oItem = oContext.getObject();
            
            var sNewStatus = oItem.status === "completed" ? "pending" : "completed";
            
            this._oApiService.updateChecklistItem(oItem.id, {
                status: sNewStatus,
                completed_at: sNewStatus === "completed" ? new Date().toISOString() : null
            }).then(function () {
                MessageToast.show("Item updated");
                that._loadWorkflowData();
            }).catch(function (oError) {
                MessageBox.error("Failed to update item: " + oError.message);
            });
        },

        /**
         * Submit for review
         */
        onSubmitForReview: function () {
            var that = this;
            var oViewModel = this.getView().getModel("view");
            
            if (!oViewModel.getProperty("/canSubmit")) {
                MessageBox.warning("Please complete all checklist items before submitting.");
                return;
            }
            
            MessageBox.confirm("Submit trial balance for checker approval?", {
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        that._oApiService.submitForReview({
                            submitted_at: new Date().toISOString()
                        }).then(function () {
                            MessageToast.show("Submitted for review");
                            that._loadWorkflowData();
                        }).catch(function (oError) {
                            MessageBox.error("Failed to submit: " + oError.message);
                        });
                    }
                }
            });
        },

        /**
         * Approve workflow (checker action)
         */
        onApprove: function () {
            var that = this;
            
            MessageBox.confirm("Approve this trial balance submission?", {
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        that._oApiService.approveWorkflow({
                            approved_at: new Date().toISOString()
                        }).then(function () {
                            MessageToast.show("Approved successfully");
                            that._loadWorkflowData();
                        }).catch(function (oError) {
                            MessageBox.error("Failed to approve: " + oError.message);
                        });
                    }
                }
            });
        },

        /**
         * Refresh data
         */
        onRefresh: function () {
            this._loadWorkflowData();
            MessageToast.show("Data refreshed");
        },

        /**
         * Navigate back
         */
        onNavBack: function () {
            this.getOwnerComponent().getRouter().navTo("home");
        }

    });
});