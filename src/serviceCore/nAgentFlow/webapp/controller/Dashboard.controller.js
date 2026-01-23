sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/routing/History",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/ui/model/Sorter",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment"
], function (Controller, History, JSONModel, Filter, FilterOperator, Sorter, MessageBox, MessageToast, Fragment) {
    "use strict";

    return Controller.extend("serviceCore.nWorkflow.controller.Dashboard", {

        onInit: function () {
            // Dashboard state model
            var oDashboardModel = new JSONModel({
                serverOnline: true,
                busy: false
            });
            this.getView().setModel(oDashboardModel, "dashboard");

            this._loadWorkflows();
            this._loadStatistics();
            this._initializeFilters();
            this._checkServerStatus();
        },

        // ==================== Navigation ====================

        onNavToLaunchpad: function () {
            window.location.href = "../nLaunchpad/webapp/index.html";
        },

        onRefreshPress: function () {
            MessageToast.show("Refreshing...");
            this._loadWorkflows();
            this._loadStatistics();
            this._checkServerStatus();
        },

        _checkServerStatus: function () {
            var that = this;
            var oDashboardModel = this.getView().getModel("dashboard");

            // Simulate server health check
            setTimeout(function () {
                // In production, make actual health check request
                oDashboardModel.setProperty("/serverOnline", true);
            }, 500);
        },

        // ==================== Data Loading ====================

        _loadWorkflows: function () {
            var oModel = this.getOwnerComponent().getModel();

            // Mock data - in production, this would call a backend API
            var aWorkflows = [
                {
                    id: "wf-001",
                    name: "Customer Onboarding",
                    description: "Automated customer onboarding workflow with verification steps",
                    status: "active",
                    lastModified: "2026-01-19 10:30",
                    executionCount: 145,
                    successRate: 98
                },
                {
                    id: "wf-002",
                    name: "Invoice Processing",
                    description: "Process incoming invoices with AI extraction and validation",
                    status: "active",
                    lastModified: "2026-01-18 15:45",
                    executionCount: 89,
                    successRate: 95
                },
                {
                    id: "wf-003",
                    name: "Document Review",
                    description: "Automated document classification and review workflow",
                    status: "draft",
                    lastModified: "2026-01-17 09:20",
                    executionCount: 0,
                    successRate: 0
                },
                {
                    id: "wf-004",
                    name: "Employee Onboarding",
                    description: "HR workflow for new employee setup and training",
                    status: "active",
                    lastModified: "2026-01-16 14:00",
                    executionCount: 34,
                    successRate: 100
                },
                {
                    id: "wf-005",
                    name: "Support Ticket Routing",
                    description: "AI-powered ticket classification and routing",
                    status: "inactive",
                    lastModified: "2026-01-10 11:15",
                    executionCount: 256,
                    successRate: 87
                }
            ];

            oModel.setProperty("/workflows", aWorkflows);
            oModel.setProperty("/filteredWorkflows", aWorkflows);
            oModel.setProperty("/filteredWorkflowCount", aWorkflows.length);
        },

        _loadStatistics: function () {
            var oModel = this.getOwnerComponent().getModel();

            // Mock statistics - in production, this would come from backend
            oModel.setProperty("/statistics", {
                totalWorkflows: 5,
                totalWorkflowsSubheader: "All configured workflows",
                activeWorkflows: 3,
                activeWorkflowsSubheader: "Currently running",
                executionsToday: 47,
                executionsTodaySubheader: "Since midnight",
                successRate: 94,
                successRateSubheader: "Last 30 days",
                successRateColor: "Good"
            });
        },

        _initializeFilters: function () {
            var oModel = this.getOwnerComponent().getModel();
            oModel.setProperty("/filters", {
                searchQuery: "",
                status: "all",
                sortBy: "name"
            });
        },

        // ==================== Navigation ====================

        onSettingsPress: function () {
            this.getOwnerComponent().getRouter().navTo("settings");
        },

        onCreateWorkflowPress: function () {
            this.getOwnerComponent().getRouter().navTo("workflowEditor", {
                workflowId: "new"
            });
        },

        onWorkflowPress: function (oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oContext = oItem.getBindingContext();
            var sWorkflowId = oContext.getProperty("id");

            this.getOwnerComponent().getRouter().navTo("workflowEditor", {
                workflowId: sWorkflowId
            });
        },

        onWorkflowSelectionChange: function (oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oContext = oItem.getBindingContext();
            this._selectedWorkflow = oContext.getObject();
        },

        // ==================== Workflow CRUD Operations ====================

        onEditWorkflow: function (oEvent) {
            var oButton = oEvent.getSource();
            var oContext = oButton.getBindingContext();
            var sWorkflowId = oContext.getProperty("id");

            this.getOwnerComponent().getRouter().navTo("workflowEditor", {
                workflowId: sWorkflowId
            });
        },

        onRunWorkflow: function (oEvent) {
            var oButton = oEvent.getSource();
            var oContext = oButton.getBindingContext();
            var oWorkflow = oContext.getObject();
            var oResourceBundle = this.getView().getModel("i18n").getResourceBundle();

            // Mock workflow execution
            MessageToast.show(oResourceBundle.getText("workflowRunning") + ": " + oWorkflow.name);

            // In production, this would call the backend API to start workflow execution
            this._executeWorkflow(oWorkflow.id);
        },

        _executeWorkflow: function (sWorkflowId) {
            // Mock execution - in production, this would call backend API
            console.log("Executing workflow:", sWorkflowId);

            // Update statistics after execution
            var oModel = this.getOwnerComponent().getModel();
            var nExecutionsToday = oModel.getProperty("/statistics/executionsToday");
            oModel.setProperty("/statistics/executionsToday", nExecutionsToday + 1);
        },

        onDeleteWorkflow: function (oEvent) {
            var oButton = oEvent.getSource();
            var oContext = oButton.getBindingContext();
            var oWorkflow = oContext.getObject();
            var oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
            var that = this;

            MessageBox.confirm(oResourceBundle.getText("deleteWorkflowMsg"), {
                title: oResourceBundle.getText("confirmDelete"),
                onClose: function (oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        that._deleteWorkflow(oWorkflow.id);
                    }
                }
            });
        },

        _deleteWorkflow: function (sWorkflowId) {
            var oModel = this.getOwnerComponent().getModel();
            var aWorkflows = oModel.getProperty("/workflows");
            var oResourceBundle = this.getView().getModel("i18n").getResourceBundle();

            // Filter out the deleted workflow
            var aFilteredWorkflows = aWorkflows.filter(function (oWorkflow) {
                return oWorkflow.id !== sWorkflowId;
            });

            oModel.setProperty("/workflows", aFilteredWorkflows);
            this._applyFilters();
            this._updateStatistics();

            MessageToast.show(oResourceBundle.getText("workflowDeleted"));
        },

        _updateStatistics: function () {
            var oModel = this.getOwnerComponent().getModel();
            var aWorkflows = oModel.getProperty("/workflows");

            var nTotal = aWorkflows.length;
            var nActive = aWorkflows.filter(function (w) { return w.status === "active"; }).length;

            oModel.setProperty("/statistics/totalWorkflows", nTotal);
            oModel.setProperty("/statistics/activeWorkflows", nActive);
        },

        // ==================== Search and Filter ====================

        onSearch: function (oEvent) {
            var sQuery = oEvent.getParameter("query") || oEvent.getParameter("newValue") || "";
            var oModel = this.getOwnerComponent().getModel();
            oModel.setProperty("/filters/searchQuery", sQuery);
            this._applyFilters();
        },

        onSearchLiveChange: function (oEvent) {
            var sQuery = oEvent.getParameter("newValue") || "";
            var oModel = this.getOwnerComponent().getModel();
            oModel.setProperty("/filters/searchQuery", sQuery);
            this._applyFilters();
        },

        onStatusFilterChange: function (oEvent) {
            var sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            var oModel = this.getOwnerComponent().getModel();
            oModel.setProperty("/filters/status", sSelectedKey);
            this._applyFilters();
        },

        onSortChange: function (oEvent) {
            var sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            var oModel = this.getOwnerComponent().getModel();
            oModel.setProperty("/filters/sortBy", sSelectedKey);
            this._applyFilters();
        },

        _applyFilters: function () {
            var oModel = this.getOwnerComponent().getModel();
            var aWorkflows = oModel.getProperty("/workflows");
            var oFilters = oModel.getProperty("/filters");

            // Apply search filter
            var aFiltered = aWorkflows.filter(function (oWorkflow) {
                var sQuery = oFilters.searchQuery.toLowerCase();
                if (!sQuery) return true;

                return oWorkflow.name.toLowerCase().indexOf(sQuery) !== -1 ||
                       oWorkflow.description.toLowerCase().indexOf(sQuery) !== -1 ||
                       oWorkflow.id.toLowerCase().indexOf(sQuery) !== -1;
            });

            // Apply status filter
            if (oFilters.status !== "all") {
                aFiltered = aFiltered.filter(function (oWorkflow) {
                    return oWorkflow.status === oFilters.status;
                });
            }

            // Apply sorting
            aFiltered.sort(function (a, b) {
                switch (oFilters.sortBy) {
                    case "name":
                        return a.name.localeCompare(b.name);
                    case "lastModified":
                        return new Date(b.lastModified) - new Date(a.lastModified);
                    case "status":
                        return a.status.localeCompare(b.status);
                    default:
                        return 0;
                }
            });

            oModel.setProperty("/filteredWorkflows", aFiltered);
            oModel.setProperty("/filteredWorkflowCount", aFiltered.length);
        },

        // ==================== Import/Export ====================

        onImportWorkflowPress: function () {
            // In production, this would open a file picker dialog
            MessageToast.show("Import workflow feature coming soon");
        },

        // ==================== Execution History ====================

        onViewExecutionsPress: function () {
            this._openExecutionHistoryDialog();
        },

        onExecutionsTilePress: function () {
            this._openExecutionHistoryDialog();
        },

        _openExecutionHistoryDialog: function () {
            var oView = this.getView();
            var that = this;

            if (!this._pExecutionHistoryDialog) {
                this._pExecutionHistoryDialog = Fragment.load({
                    id: oView.getId(),
                    name: "serviceCore.nWorkflow.fragment.ExecutionHistoryDialog",
                    controller: this
                }).then(function (oDialog) {
                    oView.addDependent(oDialog);
                    that._loadExecutionHistory();
                    return oDialog;
                });
            }

            this._pExecutionHistoryDialog.then(function (oDialog) {
                that._loadExecutionHistory();
                oDialog.open();
            });
        },

        _loadExecutionHistory: function () {
            var oModel = this.getOwnerComponent().getModel();

            // Mock execution history data
            var aExecutions = [
                {
                    id: "exec-001",
                    workflowName: "Customer Onboarding",
                    status: "completed",
                    startTime: "2026-01-19 09:15:00",
                    endTime: "2026-01-19 09:15:45",
                    duration: "45s"
                },
                {
                    id: "exec-002",
                    workflowName: "Invoice Processing",
                    status: "completed",
                    startTime: "2026-01-19 08:30:00",
                    endTime: "2026-01-19 08:32:15",
                    duration: "2m 15s"
                },
                {
                    id: "exec-003",
                    workflowName: "Customer Onboarding",
                    status: "running",
                    startTime: "2026-01-19 10:00:00",
                    endTime: "-",
                    duration: "Running..."
                },
                {
                    id: "exec-004",
                    workflowName: "Support Ticket Routing",
                    status: "failed",
                    startTime: "2026-01-18 16:45:00",
                    endTime: "2026-01-18 16:45:12",
                    duration: "12s"
                },
                {
                    id: "exec-005",
                    workflowName: "Employee Onboarding",
                    status: "completed",
                    startTime: "2026-01-18 14:20:00",
                    endTime: "2026-01-18 14:25:30",
                    duration: "5m 30s"
                }
            ];

            oModel.setProperty("/executions", aExecutions);
        },

        onCloseExecutionHistoryDialog: function () {
            this.byId("executionHistoryDialog").close();
        },

        // ==================== Statistics Tiles ====================

        onStatisticsTilePress: function (oEvent) {
            // Could navigate to detailed analytics page
            MessageToast.show("Analytics dashboard coming soon");
        },

        // ==================== User Menu ====================

        onProfilePress: function () {
            MessageToast.show("Profile settings coming soon");
        },

        onLogoutPress: function () {
            MessageToast.show("Logout functionality coming soon");
        }
    });
});

