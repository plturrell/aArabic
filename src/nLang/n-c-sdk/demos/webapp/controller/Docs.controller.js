sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast"
], function (Controller, JSONModel, MessageToast) {
    "use strict";

    return Controller.extend("galaxy.sim.controller.Docs", {

        onInit: function () {
            // Set up documentation model
            var oDocsModel = new JSONModel({
                gettingStarted: [
                    {
                        title: "Quick Start Guide",
                        description: "Get up and running with n-c-sdkHPC in minutes",
                        url: "https://github.com/nLang/n-c-sdk#quick-start"
                    },
                    {
                        title: "Installation",
                        description: "System requirements and installation instructions",
                        url: "https://github.com/nLang/n-c-sdk#installation"
                    },
                    {
                        title: "Project Setup",
                        description: "Creating your first HPC project",
                        url: "https://github.com/nLang/n-c-sdk/wiki/Project-Setup"
                    }
                ],
                apiReference: [
                    {
                        title: "Core API",
                        description: "Nucleus HPC kernel API reference",
                        url: "https://github.com/nLang/n-c-sdk/wiki/Core-API"
                    },
                    {
                        title: "SIMD Operations",
                        description: "Vectorization and SIMD instruction API",
                        url: "https://github.com/nLang/n-c-sdk/wiki/SIMD-API"
                    },
                    {
                        title: "Barnes-Hut Algorithm",
                        description: "N-body simulation algorithms",
                        url: "https://github.com/nLang/n-c-sdk/wiki/Barnes-Hut"
                    },
                    {
                        title: "WebAssembly Bridge",
                        description: "WASM compilation and browser integration",
                        url: "https://github.com/nLang/n-c-sdk/wiki/WASM-Bridge"
                    }
                ],
                guides: [
                    {
                        title: "Performance Optimization",
                        description: "Best practices for maximum performance",
                        url: "https://github.com/nLang/n-c-sdk/wiki/Performance"
                    },
                    {
                        title: "Multi-threading Guide",
                        description: "Parallel processing and thread management",
                        url: "https://github.com/nLang/n-c-sdk/wiki/Multithreading"
                    },
                    {
                        title: "Memory Management",
                        description: "Efficient memory usage patterns",
                        url: "https://github.com/nLang/n-c-sdk/wiki/Memory"
                    },
                    {
                        title: "Debugging & Profiling",
                        description: "Tools and techniques for debugging HPC code",
                        url: "https://github.com/nLang/n-c-sdk/wiki/Debugging"
                    }
                ]
            });
            this.getView().setModel(oDocsModel, "docs");
        },

        onNavBack: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("home");
        },

        onDocPress: function (oEvent) {
            var oItem = oEvent.getSource();
            var oContext = oItem.getBindingContext("docs");
            var sTitle = oContext.getProperty("title");
            var sUrl = oContext.getProperty("url");
            
            MessageToast.show("Opening " + sTitle + "...");
            window.open(sUrl, "_blank");
        }

    });
});