sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/model/json/JSONModel"
], function (UIComponent, JSONModel) {
    "use strict";

    return UIComponent.extend("hypershimmy.Component", {
        metadata: {
            manifest: "json"
        },

        /**
         * The component is initialized by UI5 automatically during the startup of the app and calls the init method once.
         * @public
         * @override
         */
        init: function () {
            // Call the base component's init function
            UIComponent.prototype.init.apply(this, arguments);

            // Create the device model
            this.setModel(this._createDeviceModel(), "device");

            // Create the app state model
            this.setModel(this._createAppStateModel(), "appState");

            // Initialize mock data (temporary for Day 5)
            this._initializeMockData();

            // Enable routing
            this.getRouter().initialize();
        },

        /**
         * Initialize mock data for testing (temporary for Day 5)
         * @private
         */
        _initializeMockData: function () {
            var oModel = this.getModel();
            
            // Create mock sources
            var aMockSources = [
                {
                    Id: "source_001",
                    Title: "Introduction to SAPUI5",
                    SourceType: "URL",
                    Url: "https://sapui5.hana.ondemand.com",
                    Status: "Ready",
                    Content: "SAPUI5 is an HTML5 framework for building enterprise-ready web applications. It provides a rich set of UI controls and follows the Model-View-Controller (MVC) pattern. The framework includes responsive design capabilities, ensuring applications work seamlessly across desktop, tablet, and mobile devices.",
                    CreatedAt: "2026-01-15T10:00:00Z",
                    UpdatedAt: "2026-01-15T10:00:00Z"
                },
                {
                    Id: "source_002",
                    Title: "SAP Fiori Design Guidelines",
                    SourceType: "PDF",
                    Url: "https://experience.sap.com/fiori-design",
                    Status: "Ready",
                    Content: "SAP Fiori represents a design system that creates a consistent user experience across SAP products. It emphasizes simplicity, coherence, and instant value. The design principles include role-based, adaptive, and simple design patterns that enhance user productivity.",
                    CreatedAt: "2026-01-15T11:30:00Z",
                    UpdatedAt: "2026-01-15T11:30:00Z"
                },
                {
                    Id: "source_003",
                    Title: "OData V4 Protocol Specification",
                    SourceType: "URL",
                    Url: "https://www.odata.org/documentation",
                    Status: "Processing",
                    Content: "OData (Open Data Protocol) is an ISO/IEC approved, OASIS standard that defines best practices for building and consuming RESTful APIs. OData V4 introduces significant improvements including simplified URLs, enhanced type system, and better support for JSON.",
                    CreatedAt: "2026-01-16T09:00:00Z",
                    UpdatedAt: "2026-01-16T09:15:00Z"
                },
                {
                    Id: "source_004",
                    Title: "Zig Programming Language Guide",
                    SourceType: "Text",
                    Url: "https://ziglang.org/documentation",
                    Status: "Ready",
                    Content: "Zig is a general-purpose programming language designed for robustness, optimality, and maintainability. It provides manual memory management with safety features, compile-time code execution, and seamless C interoperability. Zig aims to be a better C without hidden control flow or memory allocations.",
                    CreatedAt: "2026-01-16T12:00:00Z",
                    UpdatedAt: "2026-01-16T12:00:00Z"
                }
            ];
            
            // Set mock data on the default model
            oModel.setProperty("/Sources", aMockSources);
        },

        /**
         * Creates the device model
         * @private
         * @returns {sap.ui.model.json.JSONModel} the device model
         */
        _createDeviceModel: function () {
            var oModel = new JSONModel({
                isTouch: sap.ui.Device.support.touch,
                isNoTouch: !sap.ui.Device.support.touch,
                isPhone: sap.ui.Device.system.phone,
                isNoPhone: !sap.ui.Device.system.phone,
                listMode: sap.ui.Device.system.phone ? "None" : "SingleSelectMaster",
                listItemType: sap.ui.Device.system.phone ? "Active" : "Inactive"
            });
            oModel.setDefaultBindingMode("OneWay");
            return oModel;
        },

        /**
         * Creates the app state model for managing UI state
         * @private
         * @returns {sap.ui.model.json.JSONModel} the app state model
         */
        _createAppStateModel: function () {
            var oModel = new JSONModel({
                sessionId: this._generateGuid(),
                currentLayout: "TwoColumnsMidExpanded",
                previousLayout: "",
                actionButtonsInfo: {
                    midColumn: {
                        fullScreen: false
                    }
                },
                busy: false,
                selectedSourceId: null,
                chatHistory: []
            });
            return oModel;
        },

        /**
         * Generates a GUID for session identification
         * @private
         * @returns {string} a GUID
         */
        _generateGuid: function () {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                var r = Math.random() * 16 | 0,
                    v = c === 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }
    });
});
