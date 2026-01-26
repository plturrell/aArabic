sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/routing/History",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast"
], function (Controller, History, JSONModel, MessageToast) {
    "use strict";

    return Controller.extend("trial.balance.controller.TrialBalance", {
        onInit: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("trialBalance").attachPatternMatched(this._onObjectMatched, this);

            // Initialize model with mock data
            this._initializeModel();
        },

        _onObjectMatched: function () {
            // Load data when route is matched
            this._loadTrialBalanceData();
        },

        _initializeModel: function () {
            var oModel = new JSONModel({
                accounts: [],
                summary: {
                    totalDebit: 0,
                    totalCredit: 0,
                    difference: 0,
                    differenceState: "Success"
                }
            });
            this.getView().setModel(oModel, "trialBalance");
        },

        _loadTrialBalanceData: function () {
            // TODO: Replace with actual service call
            var oModel = this.getView().getModel("trialBalance");
            
            // Mock data
            var aAccounts = [
                {
                    accountNumber: "1000",
                    accountName: "Cash",
                    debit: 50000,
                    credit: 0,
                    balance: 50000,
                    currency: "USD",
                    status: "Balanced",
                    statusState: "Success",
                    balanceState: "None"
                },
                {
                    accountNumber: "2000",
                    accountName: "Accounts Payable",
                    debit: 0,
                    credit: 50000,
                    balance: -50000,
                    currency: "USD",
                    status: "Balanced",
                    statusState: "Success",
                    balanceState: "None"
                }
            ];

            oModel.setProperty("/accounts", aAccounts);
            this._updateSummary();
        },

        _updateSummary: function () {
            var oModel = this.getView().getModel("trialBalance");
            var aAccounts = oModel.getProperty("/accounts");

            var totalDebit = 0;
            var totalCredit = 0;

            aAccounts.forEach(function (oAccount) {
                totalDebit += oAccount.debit;
                totalCredit += oAccount.credit;
            });

            var difference = totalDebit - totalCredit;
            var differenceState = difference === 0 ? "Success" : "Error";

            oModel.setProperty("/summary", {
                totalDebit: totalDebit,
                totalCredit: totalCredit,
                difference: Math.abs(difference),
                differenceState: differenceState
            });
        },

        onNavBack: function () {
            var oHistory = History.getInstance();
            var sPreviousHash = oHistory.getPreviousHash();

            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                var oRouter = this.getOwnerComponent().getRouter();
                oRouter.navTo("home", {}, true);
            }
        },

        onRefresh: function () {
            this._loadTrialBalanceData();
            MessageToast.show("Data refreshed");
        },

        onExport: function () {
            MessageToast.show("Export functionality to be implemented");
            // TODO: Implement export to Excel
        },

        onFilterChange: function () {
            MessageToast.show("Filter functionality to be implemented");
            // TODO: Implement filtering
        },

        onSearch: function () {
            MessageToast.show("Search functionality to be implemented");
            // TODO: Implement search
        },

        onRowSelect: function (oEvent) {
            var oSelectedRow = oEvent.getParameter("rowContext");
            if (oSelectedRow) {
                var oAccount = oSelectedRow.getObject();
                var oRouter = this.getOwnerComponent().getRouter();
                oRouter.navTo("accountDetail", {
                    accountId: oAccount.accountNumber
                });
            }
        }
    });
});