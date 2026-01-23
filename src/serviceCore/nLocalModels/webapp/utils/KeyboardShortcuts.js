sap.ui.define([
    "llm/server/dashboard/utils/ThemeService"
], function (ThemeService) {
    "use strict";

    /**
     * Keyboard Shortcuts Service
     * Handles global keyboard shortcuts for the dashboard
     */
    var KeyboardShortcuts = {

        // Shortcut definitions
        SHORTCUTS: {
            // General
            SEARCH: { key: "k", description: "Open global search" },
            NEW_AGENT: { key: "n", description: "Add new agent" },
            NEW_WORKFLOW: { key: "w", description: "Create new workflow" },
            REFRESH: { key: "r", description: "Refresh current view" },
            TOGGLE_DARK_MODE: { key: "d", description: "Toggle dark mode" },
            HELP: { key: "?", description: "Show shortcuts help" },
            ESCAPE: { key: "Escape", description: "Close dialogs/modals" },

            // Page Navigation
            NAV_MAIN: { key: "1", description: "Go to Dashboard" },
            NAV_ORCHESTRATION: { key: "2", description: "Go to Agent Orchestration" },
            NAV_MODEL_ROUTER: { key: "3", description: "Go to Model Router" },
            NAV_AB_TESTING: { key: "4", description: "Go to A/B Testing" },
            NAV_SETTINGS: { key: "5", description: "Go to Settings" },

            // Zoom Controls
            ZOOM_IN: { key: "+", description: "Zoom in" },
            ZOOM_OUT: { key: "-", description: "Zoom out" },
            ZOOM_RESET: { key: "0", description: "Reset zoom to 100%" },

            // Selection
            SELECT_ALL: { key: "a", description: "Select all nodes" },
            DELETE: { key: "Delete", description: "Delete selected" },
            COPY: { key: "c", description: "Copy selected" },
            PASTE: { key: "v", description: "Paste" },
            CUT: { key: "x", description: "Cut selected" },
            UNDO: { key: "z", description: "Undo" },
            REDO: { key: "Z", description: "Redo (Shift+Z)" },

            // Workflow Operations
            SAVE: { key: "s", description: "Save workflow" },
            OPEN: { key: "o", description: "Open workflow" },
            EXPORT: { key: "e", description: "Export workflow" },
            EXECUTE: { key: "Enter", description: "Execute workflow" },

            // Canvas Navigation
            PAN_UP: { key: "ArrowUp", description: "Pan canvas up" },
            PAN_DOWN: { key: "ArrowDown", description: "Pan canvas down" },
            PAN_LEFT: { key: "ArrowLeft", description: "Pan canvas left" },
            PAN_RIGHT: { key: "ArrowRight", description: "Pan canvas right" },
            FOCUS_NEXT: { key: "Tab", description: "Focus next node" },
            FOCUS_PREV: { key: "Tab", shift: true, description: "Focus previous node" }
        },

        // State for pan mode
        _isPanMode: false,

        // Page navigation map (key number -> navigation key)
        PAGE_MAP: {
            "1": "main",
            "2": "orchestration", 
            "3": "modelRouter",
            "4": "abTesting",
            "5": "settings"
        },

        // State
        _controller: null,
        _fnKeyDownHandler: null,
        _helpDialog: null,
        _enabled: true,

        /**
         * Initialize keyboard shortcuts
         * @param {sap.ui.core.mvc.Controller} oController - The App controller
         */
        init: function (oController) {
            this._controller = oController;
            this._attachKeyboardListener();
            return this;
        },

        /**
         * Destroy and cleanup keyboard shortcuts
         */
        destroy: function () {
            this._detachKeyboardListener();
            if (this._helpDialog) {
                this._helpDialog.destroy();
                this._helpDialog = null;
            }
            this._controller = null;
        },

        /**
         * Enable/disable keyboard shortcuts
         */
        setEnabled: function (bEnabled) {
            this._enabled = bEnabled;
        },

        /**
         * Attach keyboard event listener
         * @private
         */
        _attachKeyboardListener: function () {
            var that = this;
            this._fnKeyDownHandler = function (oEvent) {
                that._onKeyDown(oEvent);
            };
            this._fnKeyUpHandler = function (oEvent) {
                that._onKeyUp(oEvent);
            };
            document.addEventListener("keydown", this._fnKeyDownHandler);
            document.addEventListener("keyup", this._fnKeyUpHandler);
        },

        /**
         * Detach keyboard event listener
         * @private
         */
        _detachKeyboardListener: function () {
            if (this._fnKeyDownHandler) {
                document.removeEventListener("keydown", this._fnKeyDownHandler);
                this._fnKeyDownHandler = null;
            }
            if (this._fnKeyUpHandler) {
                document.removeEventListener("keyup", this._fnKeyUpHandler);
                this._fnKeyUpHandler = null;
            }
        },

        /**
         * Handle keydown events
         * @private
         */
        _onKeyDown: function (oEvent) {
            if (!this._enabled || !this._controller) {
                return;
            }

            // Skip if user is typing in an input field
            var sTagName = oEvent.target.tagName.toLowerCase();
            var bIsInputField = sTagName === "input" || sTagName === "textarea" ||
                                oEvent.target.isContentEditable;

            // Handle Escape key (works even in input fields)
            if (oEvent.key === "Escape") {
                this._closeDialogs();
                return;
            }

            // Handle ? for help (Shift + /)
            if (oEvent.key === "?" && !bIsInputField) {
                oEvent.preventDefault();
                this._showHelpDialog();
                return;
            }

            // Skip other shortcuts if in input field
            if (bIsInputField) {
                return;
            }

            // Check for Ctrl/Cmd modifier
            var bCtrlOrCmd = oEvent.ctrlKey || oEvent.metaKey;

            if (bCtrlOrCmd) {
                this._handleCtrlShortcut(oEvent);
            } else {
                this._handleNonCtrlShortcut(oEvent);
            }
        },

        /**
         * Handle non-Ctrl shortcuts (arrow keys, Tab, Delete, Space)
         * @private
         */
        _handleNonCtrlShortcut: function (oEvent) {
            var sKey = oEvent.key;

            switch (sKey) {
                // Delete/Backspace - Delete selected
                case "Delete":
                case "Backspace":
                    oEvent.preventDefault();
                    this._deleteSelected();
                    break;

                // Arrow keys - Pan canvas
                case "ArrowUp":
                    oEvent.preventDefault();
                    this._panCanvas("up");
                    break;
                case "ArrowDown":
                    oEvent.preventDefault();
                    this._panCanvas("down");
                    break;
                case "ArrowLeft":
                    oEvent.preventDefault();
                    this._panCanvas("left");
                    break;
                case "ArrowRight":
                    oEvent.preventDefault();
                    this._panCanvas("right");
                    break;

                // Tab - Focus next/previous node
                case "Tab":
                    oEvent.preventDefault();
                    if (oEvent.shiftKey) {
                        this._focusPreviousNode();
                    } else {
                        this._focusNextNode();
                    }
                    break;

                // Space - Toggle pan mode
                case " ":
                    oEvent.preventDefault();
                    this._togglePanMode(true);
                    break;
            }
        },

        /**
         * Handle keyup events for Space (pan mode)
         * @private
         */
        _onKeyUp: function (oEvent) {
            if (oEvent.key === " ") {
                this._togglePanMode(false);
            }
        },

        /**
         * Handle Ctrl/Cmd shortcuts
         * @private
         */
        _handleCtrlShortcut: function (oEvent) {
            var sKey = oEvent.key;
            var bShift = oEvent.shiftKey;

            // Handle Ctrl+Shift+Z for Redo
            if (bShift && sKey.toLowerCase() === "z") {
                oEvent.preventDefault();
                this._redo();
                return;
            }

            switch (sKey.toLowerCase()) {
                case "k":
                    oEvent.preventDefault();
                    this._openGlobalSearch();
                    break;
                case "n":
                    oEvent.preventDefault();
                    this._addNewAgent();
                    break;
                case "w":
                    oEvent.preventDefault();
                    this._createNewWorkflow();
                    break;
                case "r":
                    oEvent.preventDefault();
                    this._refreshCurrentView();
                    break;
                case "d":
                    oEvent.preventDefault();
                    this._toggleDarkMode();
                    break;
                case "1":
                case "2":
                case "3":
                case "4":
                case "5":
                    oEvent.preventDefault();
                    this._navigateToPage(sKey);
                    break;

                // Zoom controls
                case "+":
                case "=":
                    oEvent.preventDefault();
                    this._zoomIn();
                    break;
                case "-":
                    oEvent.preventDefault();
                    this._zoomOut();
                    break;
                case "0":
                    oEvent.preventDefault();
                    this._zoomReset();
                    break;

                // Selection
                case "a":
                    oEvent.preventDefault();
                    this._selectAll();
                    break;
                case "c":
                    oEvent.preventDefault();
                    this._copySelected();
                    break;
                case "v":
                    oEvent.preventDefault();
                    this._paste();
                    break;
                case "x":
                    oEvent.preventDefault();
                    this._cutSelected();
                    break;
                case "z":
                    oEvent.preventDefault();
                    this._undo();
                    break;

                // Workflow operations
                case "s":
                    oEvent.preventDefault();
                    this._saveWorkflow();
                    break;
                case "o":
                    oEvent.preventDefault();
                    this._openWorkflow();
                    break;
                case "e":
                    oEvent.preventDefault();
                    this._exportWorkflow();
                    break;
                case "enter":
                    oEvent.preventDefault();
                    this._executeWorkflow();
                    break;
            }
        },

        /**
         * Open global search
         * @private
         */
        _openGlobalSearch: function () {
            sap.m.MessageToast.show("Global Search (Ctrl+K)");
            // TODO: Implement search dialog
        },

        /**
         * Add new agent
         * @private
         */
        _addNewAgent: function () {
            sap.m.MessageToast.show("Add New Agent (Ctrl+N)");
            // Navigate to orchestration page if not already there
            this._navigateToPage("2");
        },

        /**
         * Create new workflow
         * @private
         */
        _createNewWorkflow: function () {
            sap.m.MessageToast.show("Create New Workflow (Ctrl+W)");
            // Navigate to orchestration page if not already there
            this._navigateToPage("2");
        },

        /**
         * Refresh current view
         * @private
         */
        _refreshCurrentView: function () {
            if (this._controller && this._controller.onRefresh) {
                this._controller.onRefresh();
            }
        },

        /**
         * Toggle dark mode
         * @private
         */
        _toggleDarkMode: function () {
            ThemeService.toggleTheme();
            var bIsDark = ThemeService.isDarkMode();
            sap.m.MessageToast.show(bIsDark ? "Dark Mode enabled" : "Light Mode enabled");

            // Update the theme toggle button if it exists
            if (this._controller) {
                var oThemeToggleBtn = this._controller.byId("themeToggleBtn");
                if (oThemeToggleBtn) {
                    oThemeToggleBtn.setPressed(bIsDark);
                    this._controller._updateThemeToggleIcon(bIsDark);
                }
            }
        },

        /**
         * Navigate to a specific page
         * @private
         */
        _navigateToPage: function (sKeyNumber) {
            var sNavKey = this.PAGE_MAP[sKeyNumber];
            if (!sNavKey || !this._controller) {
                return;
            }

            var oRouter = this._controller.getOwnerComponent().getRouter();
            var oSideNav = this._controller.byId("sideNavigation");

            // Map navigation keys to routes
            var mPageMap = {
                "main": "main",
                "orchestration": "orchestration",
                "modelRouter": "modelRouter",
                "abTesting": "abTesting",
                "settings": "settings"
            };

            var sRoute = mPageMap[sNavKey];
            if (sRoute && oRouter) {
                oRouter.navTo(sRoute);

                // Update side navigation selection
                if (oSideNav) {
                    var oNavList = oSideNav.getItem();
                    if (oNavList) {
                        var aItems = oNavList.getItems();
                        for (var i = 0; i < aItems.length; i++) {
                            if (aItems[i].getKey() === sNavKey) {
                                oNavList.setSelectedItem(aItems[i]);
                                break;
                            }
                        }
                    }
                }
            } else if (sNavKey === "settings") {
                sap.m.MessageToast.show("Settings page coming soon");
            }
        },

        /**
         * Close any open dialogs
         * @private
         */
        _closeDialogs: function () {
            // Close help dialog if open
            if (this._helpDialog && this._helpDialog.isOpen()) {
                this._helpDialog.close();
                return;
            }

            // Try to close any open popups/dialogs
            var oPopup = sap.ui.core.Popup.getLastOpenPopup();
            if (oPopup) {
                oPopup.close();
            }
        },

        /**
         * Show keyboard shortcuts help dialog
         * @private
         */
        _showHelpDialog: function () {
            var that = this;

            if (!this._helpDialog) {
                sap.ui.core.Fragment.load({
                    name: "llm.server.dashboard.view.fragments.ShortcutsHelp",
                    controller: this
                }).then(function (oDialog) {
                    that._helpDialog = oDialog;
                    that._controller.getView().addDependent(oDialog);
                    oDialog.open();
                });
            } else {
                this._helpDialog.open();
            }
        },

        /**
         * Close help dialog (called from fragment)
         */
        onCloseHelpDialog: function () {
            if (this._helpDialog) {
                this._helpDialog.close();
            }
        },

        // ============================================
        // Zoom Control Methods
        // ============================================

        /**
         * Zoom in on the canvas
         * @private
         */
        _zoomIn: function () {
            if (this._controller && this._controller.onZoomIn) {
                this._controller.onZoomIn();
            } else {
                sap.m.MessageToast.show("Zoom In");
            }
        },

        /**
         * Zoom out on the canvas
         * @private
         */
        _zoomOut: function () {
            if (this._controller && this._controller.onZoomOut) {
                this._controller.onZoomOut();
            } else {
                sap.m.MessageToast.show("Zoom Out");
            }
        },

        /**
         * Reset zoom to 100%
         * @private
         */
        _zoomReset: function () {
            if (this._controller && this._controller.onZoomReset) {
                this._controller.onZoomReset();
            } else {
                sap.m.MessageToast.show("Zoom Reset to 100%");
            }
        },

        // ============================================
        // Selection Methods
        // ============================================

        /**
         * Select all nodes
         * @private
         */
        _selectAll: function () {
            if (this._controller && this._controller.onSelectAll) {
                this._controller.onSelectAll();
            } else {
                sap.m.MessageToast.show("Select All");
            }
        },

        /**
         * Delete selected nodes
         * @private
         */
        _deleteSelected: function () {
            if (this._controller && this._controller.onDeleteSelected) {
                this._controller.onDeleteSelected();
            } else {
                sap.m.MessageToast.show("Delete Selected");
            }
        },

        /**
         * Copy selected nodes
         * @private
         */
        _copySelected: function () {
            if (this._controller && this._controller.onCopy) {
                this._controller.onCopy();
            } else {
                sap.m.MessageToast.show("Copy");
            }
        },

        /**
         * Paste nodes from clipboard
         * @private
         */
        _paste: function () {
            if (this._controller && this._controller.onPaste) {
                this._controller.onPaste();
            } else {
                sap.m.MessageToast.show("Paste");
            }
        },

        /**
         * Cut selected nodes
         * @private
         */
        _cutSelected: function () {
            if (this._controller && this._controller.onCut) {
                this._controller.onCut();
            } else {
                sap.m.MessageToast.show("Cut");
            }
        },

        /**
         * Undo last action
         * @private
         */
        _undo: function () {
            if (this._controller && this._controller.onUndo) {
                this._controller.onUndo();
            } else {
                sap.m.MessageToast.show("Undo");
            }
        },

        /**
         * Redo last undone action
         * @private
         */
        _redo: function () {
            if (this._controller && this._controller.onRedo) {
                this._controller.onRedo();
            } else {
                sap.m.MessageToast.show("Redo");
            }
        },

        // ============================================
        // Workflow Operation Methods
        // ============================================

        /**
         * Save current workflow
         * @private
         */
        _saveWorkflow: function () {
            if (this._controller && this._controller.onSaveWorkflow) {
                this._controller.onSaveWorkflow();
            } else {
                sap.m.MessageToast.show("Save Workflow");
            }
        },

        /**
         * Open workflow dialog
         * @private
         */
        _openWorkflow: function () {
            if (this._controller && this._controller.onOpenWorkflow) {
                this._controller.onOpenWorkflow();
            } else {
                sap.m.MessageToast.show("Open Workflow");
            }
        },

        /**
         * Export workflow
         * @private
         */
        _exportWorkflow: function () {
            if (this._controller && this._controller.onExportWorkflow) {
                this._controller.onExportWorkflow();
            } else {
                sap.m.MessageToast.show("Export Workflow");
            }
        },

        /**
         * Execute current workflow
         * @private
         */
        _executeWorkflow: function () {
            if (this._controller && this._controller.onExecuteWorkflow) {
                this._controller.onExecuteWorkflow();
            } else {
                sap.m.MessageToast.show("Execute Workflow");
            }
        },

        // ============================================
        // Canvas Navigation Methods
        // ============================================

        /**
         * Pan the canvas in a direction
         * @param {string} sDirection - Direction to pan: "up", "down", "left", "right"
         * @private
         */
        _panCanvas: function (sDirection) {
            if (this._controller && this._controller.onPanCanvas) {
                this._controller.onPanCanvas(sDirection);
            }
        },

        /**
         * Toggle pan mode (Space + drag)
         * @param {boolean} bActive - Whether pan mode is active
         * @private
         */
        _togglePanMode: function (bActive) {
            this._isPanMode = bActive;
            if (this._controller && this._controller.onTogglePanMode) {
                this._controller.onTogglePanMode(bActive);
            }
        },

        /**
         * Focus next node in workflow
         * @private
         */
        _focusNextNode: function () {
            if (this._controller && this._controller.onFocusNextNode) {
                this._controller.onFocusNextNode();
            } else {
                sap.m.MessageToast.show("Focus Next Node");
            }
        },

        /**
         * Focus previous node in workflow
         * @private
         */
        _focusPreviousNode: function () {
            if (this._controller && this._controller.onFocusPreviousNode) {
                this._controller.onFocusPreviousNode();
            } else {
                sap.m.MessageToast.show("Focus Previous Node");
            }
        },

        /**
         * Get all shortcuts for display organized by category
         * @returns {Object} Object with shortcuts organized by category
         */
        getShortcutsByCategory: function () {
            var bIsMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;
            var sModifier = bIsMac ? "⌘" : "Ctrl";

            return {
                zoom: [
                    { shortcut: sModifier + " + +", description: "Zoom in" },
                    { shortcut: sModifier + " + -", description: "Zoom out" },
                    { shortcut: sModifier + " + 0", description: "Reset zoom to 100%" }
                ],
                selection: [
                    { shortcut: sModifier + " + A", description: "Select all nodes" },
                    { shortcut: "Delete / Backspace", description: "Delete selected" },
                    { shortcut: sModifier + " + C", description: "Copy selected" },
                    { shortcut: sModifier + " + V", description: "Paste" },
                    { shortcut: sModifier + " + X", description: "Cut selected" },
                    { shortcut: sModifier + " + Z", description: "Undo" },
                    { shortcut: sModifier + " + Shift + Z", description: "Redo" }
                ],
                workflow: [
                    { shortcut: sModifier + " + S", description: "Save workflow" },
                    { shortcut: sModifier + " + O", description: "Open workflow" },
                    { shortcut: sModifier + " + E", description: "Export workflow" },
                    { shortcut: sModifier + " + Enter", description: "Execute workflow" }
                ],
                navigation: [
                    { shortcut: "Arrow Keys", description: "Pan canvas" },
                    { shortcut: "Space + Drag", description: "Pan mode" },
                    { shortcut: "Tab", description: "Focus next node" },
                    { shortcut: "Shift + Tab", description: "Focus previous node" },
                    { shortcut: sModifier + " + 1-5", description: "Navigate to pages" }
                ],
                general: [
                    { shortcut: sModifier + " + K", description: "Open global search" },
                    { shortcut: sModifier + " + N", description: "Add new agent" },
                    { shortcut: sModifier + " + W", description: "Create new workflow" },
                    { shortcut: sModifier + " + R", description: "Refresh current view" },
                    { shortcut: sModifier + " + D", description: "Toggle dark mode" },
                    { shortcut: "?", description: "Show this help dialog" },
                    { shortcut: "Esc", description: "Close dialogs/modals" }
                ]
            };
        },

        /**
         * Get all shortcuts for display (flat list)
         * @returns {Array} Array of shortcut objects
         */
        getShortcutsList: function () {
            var oCategories = this.getShortcutsByCategory();
            var aAllShortcuts = [];

            Object.keys(oCategories).forEach(function(sCategory) {
                aAllShortcuts = aAllShortcuts.concat(oCategories[sCategory]);
            });

            return aAllShortcuts;
        }
    };

    return KeyboardShortcuts;
});
