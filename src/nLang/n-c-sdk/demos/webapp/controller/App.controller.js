sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment"
], function (Controller, MessageToast, Fragment) {
    "use strict";

    return Controller.extend("galaxy.sim.controller.App", {

        onInit: function () {
            // Get the router and navigate to home by default
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.initialize();
        },

        // ========================================
        // Header Event Handlers
        // ========================================

        /**
         * Menu button pressed - toggle sidebar
         */
        onMenuButtonPress: function () {
            var oToolPage = this.byId("toolPage");
            oToolPage.setSideExpanded(!oToolPage.getSideExpanded());
        },

        /**
         * Home icon pressed - navigate to home
         */
        onHomeIconPress: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("home");
        },

        /**
         * Search button pressed
         */
        onSearchPress: function () {
            MessageToast.show("Search functionality coming soon!");
        },

        /**
         * Search field handler
         */
        onSearch: function (oEvent) {
            var sQuery = oEvent.getParameter("query");
            if (sQuery) {
                MessageToast.show("Searching for: " + sQuery);
            }
        },

        /**
         * Documentation button pressed
         */
        onDocsPress: function () {
            // Navigate to documentation or open external docs
            window.open("https://github.com/nLang/n-c-sdk", "_blank");
        },

        /**
         * Settings button pressed
         */
        onSettingsPress: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("settings");
        },

        /**
         * AI Assistant button pressed - opens/closes Copilot chat
         */
        onAIAssistantPress: function () {
            if (!this._copilotDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "galaxy.sim.view.fragments.CopilotChat",
                    controller: this
                }).then(function (oDialog) {
                    this._copilotDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                if (this._copilotDialog.isOpen()) {
                    this._copilotDialog.close();
                } else {
                    this._copilotDialog.open();
                }
            }
        },

        /**
         * Close Copilot dialog
         */
        onCloseCopilot: function () {
            if (this._copilotDialog) {
                this._copilotDialog.close();
            }
        },

        /**
         * Product Switch button pressed - opens product switch popover
         */
        onProductSwitchPress: function (oEvent) {
            var oButton = oEvent.getSource();

            if (!this._productSwitch) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "galaxy.sim.view.fragments.ProductSwitch",
                    controller: this
                }).then(function (oPopover) {
                    this._productSwitch = oPopover;
                    this.getView().addDependent(oPopover);
                    oPopover.openBy(oButton);
                }.bind(this));
            } else {
                this._productSwitch.openBy(oButton);
            }
        },

        // ========================================
        // Product Switch Handlers
        // ========================================

        onProductNCSDK: function () {
            // Current app - just close popover
            if (this._productSwitch) {
                this._productSwitch.close();
            }
            MessageToast.show("You are here: n-c-sdk");
        },

        onProductPythonSDK: function () {
            // Navigate to Intelligence SDK (runs on port 8084)
            window.open('http://localhost:8084/index.html', "_blank");
            if (this._productSwitch) {
                this._productSwitch.close();
            }
        },

        onProductLocalModels: function () {
            // Navigate to nLocalModels webapp (runs on port 8081)
            window.open('http://localhost:8081/index.html', "_blank");
            if (this._productSwitch) {
                this._productSwitch.close();
            }
        },

        onProductAgentFlow: function () {
            // Navigate to nAgentFlow webapp (runs on port 8082)
            window.open('http://localhost:8082/index.html', "_blank");
            if (this._productSwitch) {
                this._productSwitch.close();
            }
        },

        onProductAgentMeta: function () {
            // nAgentMeta is a backend service with no webapp UI
            MessageToast.show("nAgentMeta: Backend-only service (no UI)");
            if (this._productSwitch) {
                this._productSwitch.close();
            }
        },

        onProductGrounding: function () {
            // Navigate to nGrounding webapp (runs on port 8083)
            window.open('http://localhost:8083/index.html', "_blank");
            if (this._productSwitch) {
                this._productSwitch.close();
            }
        },

        onProductEmbeddedAI: function () {
            var sEmbeddedAIPath = window.location.pathname.replace('/src/nLang/n-c-sdk/demos/webapp/index.html', '/src/serviceCore/nEmbeddedAI/index.html');
            window.open(window.location.origin + sEmbeddedAIPath, "_blank");
            if (this._productSwitch) {
                this._productSwitch.close();
            }
        },

        // ========================================
        // Side Navigation Handlers
        // ========================================

        /**
         * Toggle sidebar expanded/collapsed state
         */
        onToggleSidebar: function () {
            var oToolPage = this.byId("toolPage");
            oToolPage.setSideExpanded(!oToolPage.getSideExpanded());
        },

        /**
         * Handle navigation item selection
         */
        onNavigationSelect: function (oEvent) {
            var oItem = oEvent.getParameter("item");
            var sKey = oItem.getKey();

            // Only navigate if item has a key (not a group header)
            if (sKey) {
                var oRouter = this.getOwnerComponent().getRouter();
                oRouter.navTo(sKey);
            }
        },

        /**
         * Refresh button handler
         */
        onRefresh: function () {
            // Reload the current view/data
            window.location.reload();
        },

        /**
         * Settings button handler
         */
        onSettings: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("settings");
        },

        // ========================================
        // User Menu Handlers
        // ========================================

        /**
         * Avatar pressed - open user menu
         */
        onAvatarPress: function (oEvent) {
            var oAvatar = oEvent.getSource();

            if (!this._userMenu) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "galaxy.sim.view.fragments.UserMenu",
                    controller: this
                }).then(function (oMenu) {
                    this._userMenu = oMenu;
                    this.getView().addDependent(oMenu);
                    oMenu.openBy(oAvatar);
                }.bind(this));
            } else {
                this._userMenu.openBy(oAvatar);
            }
        },

        /**
         * User menu - Profile
         */
        onUserMenuProfile: function () {
            MessageToast.show("Opening user profile...");
        },

        /**
         * User menu - Settings
         */
        onUserMenuSettings: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("settings");
        }

    });
});