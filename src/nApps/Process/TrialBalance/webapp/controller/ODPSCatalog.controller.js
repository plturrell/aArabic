sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function (Controller, JSONModel, MessageToast, MessageBox) {
    "use strict";

    return Controller.extend("trialbalance.controller.ODPSCatalog", {

        onInit: function () {
            // Initialize models
            const oCatalogModel = new JSONModel({
                name: "Trial Balance Data Products",
                version: "1.0.0",
                specification: "ODPS v4.1",
                productCount: 8,
                categories: ["primary", "metadata", "operational"]
            });
            this.getView().setModel(oCatalogModel, "catalog");

            const oViewModel = new JSONModel({
                category: "all"
            });
            this.getView().setModel(oViewModel, "view");

            // Load data products
            this._loadDataProducts();
            this._loadQualityReport();
        },

        /**
         * Load all ODPS data products from backend
         */
        _loadDataProducts: function () {
            const sUrl = "/api/v1/data-products";
            
            fetch(sUrl)
                .then(response => response.json())
                .then(data => {
                    const oModel = new JSONModel(data);
                    this.getView().setModel(oModel, "products");
                })
                .catch(error => {
                    MessageToast.show("Failed to load data products: " + error.message);
                    // Load mock data for demonstration
                    this._loadMockDataProducts();
                });
        },

        /**
         * Load quality report
         */
        _loadQualityReport: function () {
            const sUrl = "/api/v1/data-products/quality-report";
            
            fetch(sUrl)
                .then(response => response.json())
                .then(data => {
                    const oModel = new JSONModel(data);
                    this.getView().setModel(oModel, "quality");
                })
                .catch(error => {
                    // Load mock quality data
                    const oModel = new JSONModel({
                        generatedAt: Date.now(),
                        averageQuality: 94.6,
                        products: [
                            { name: "ACDOCA Journal Entries", qualityScore: 95.0 },
                            { name: "Exchange Rates", qualityScore: 98.0 },
                            { name: "Trial Balance Aggregated", qualityScore: 92.0 },
                            { name: "Period Variances", qualityScore: 90.0 },
                            { name: "Account Master", qualityScore: 98.0 }
                        ]
                    });
                    this.getView().setModel(oModel, "quality");
                });
        },

        /**
         * Load mock data products (fallback)
         */
        _loadMockDataProducts: function () {
            const aMockProducts = [
                {
                    productID: "urn:uuid:acdoca-journal-entries-v1",
                    name: "ACDOCA Universal Journal Entries",
                    version: "1.0.0",
                    category: "primary",
                    qualityScore: 95.0,
                    description: "Complete SAP S/4HANA ACDOCA universal journal entries",
                    owner: "Trial Balance Team",
                    updateFrequency: "real-time",
                    dataFormat: "CSN",
                    endpoints: [
                        { name: "REST API", endpoint: "/api/v1/journal-entries" },
                        { name: "Lineage API", endpoint: "/api/v1/lineage/acdoca" }
                    ]
                },
                {
                    productID: "urn:uuid:exchange-rates-v1",
                    name: "Foreign Exchange Rates",
                    version: "1.0.0",
                    category: "primary",
                    qualityScore: 98.0,
                    description: "Multi-currency exchange rates for financial consolidation",
                    owner: "Trial Balance Team",
                    updateFrequency: "daily",
                    dataFormat: "CSN",
                    endpoints: [
                        { name: "REST API", endpoint: "/api/v1/exchange-rates" },
                        { name: "Conversion API", endpoint: "/api/v1/exchange-rates/convert" }
                    ]
                },
                {
                    productID: "urn:uuid:trial-balance-aggregated-v1",
                    name: "Trial Balance (Aggregated)",
                    version: "1.0.0",
                    category: "primary",
                    qualityScore: 92.0,
                    description: "Aggregated trial balance entries by account",
                    owner: "Trial Balance Team",
                    updateFrequency: "daily",
                    dataFormat: "CSN",
                    endpoints: [
                        { name: "REST API", endpoint: "/api/v1/trial-balance" },
                        { name: "Overview API", endpoint: "/api/v1/trial-balance/overview" }
                    ]
                },
                {
                    productID: "urn:uuid:variances-v1",
                    name: "Period-over-Period Variances",
                    version: "1.0.0",
                    category: "primary",
                    qualityScore: 90.0,
                    description: "Period-over-period variance analysis with AI commentary",
                    owner: "Trial Balance Team",
                    updateFrequency: "monthly",
                    dataFormat: "CSN",
                    endpoints: [
                        { name: "REST API", endpoint: "/api/v1/trial-balance/variance" },
                        { name: "Commentary API", endpoint: "/api/v1/ai/commentary/generate" }
                    ]
                },
                {
                    productID: "urn:uuid:account-master-v1",
                    name: "G/L Account Master Data",
                    version: "1.0.0",
                    category: "primary",
                    qualityScore: 98.0,
                    description: "Chart of accounts with hierarchical relationships",
                    owner: "Trial Balance Team",
                    updateFrequency: "on-change",
                    dataFormat: "CSN",
                    endpoints: [
                        { name: "REST API", endpoint: "/api/v1/accounts" },
                        { name: "Hierarchy API", endpoint: "/api/v1/accounts/hierarchy" }
                    ]
                },
                {
                    productID: "urn:uuid:data-lineage-v1",
                    name: "Data Lineage Tracking",
                    version: "1.0.0",
                    category: "metadata",
                    qualityScore: 100.0,
                    description: "Complete data lineage tracking with SHA-256 verification",
                    owner: "Trial Balance Team",
                    updateFrequency: "real-time",
                    dataFormat: "CSN",
                    endpoints: [
                        { name: "Lineage API", endpoint: "/api/v1/lineage" },
                        { name: "Graph API", endpoint: "/api/v1/lineage/graph" }
                    ]
                },
                {
                    productID: "urn:uuid:dataset-metadata-v1",
                    name: "Dataset Quality Metrics",
                    version: "1.0.0",
                    category: "metadata",
                    qualityScore: 100.0,
                    description: "Quality metrics and dataset discovery metadata",
                    owner: "Trial Balance Team",
                    updateFrequency: "real-time",
                    dataFormat: "CSN",
                    endpoints: [
                        { name: "Metadata API", endpoint: "/api/v1/metadata" },
                        { name: "Quality API", endpoint: "/api/v1/data-products/:id/quality" }
                    ]
                },
                {
                    productID: "urn:uuid:checklist-items-v1",
                    name: "Trial Balance Workflow Checklist",
                    version: "1.0.0",
                    category: "operational",
                    qualityScore: 85.0,
                    description: "13-stage IFRS workflow tracking",
                    owner: "Trial Balance Team",
                    updateFrequency: "real-time",
                    dataFormat: "CSN",
                    endpoints: [
                        { name: "Checklist API", endpoint: "/api/v1/trial-balance/checklist" },
                        { name: "Workflow State API", endpoint: "/api/v1/workflow/state" }
                    ]
                }
            ];

            const oModel = new JSONModel({ products: aMockProducts });
            this.getView().setModel(oModel, "products");
        },

        /**
         * Refresh data products
         */
        onRefresh: function () {
            MessageToast.show("Refreshing data products...");
            this._loadDataProducts();
            this._loadQualityReport();
        },

        /**
         * Export catalog
         */
        onExport: function () {
            MessageToast.show("Exporting catalog metadata...");
            // TODO: Export to JSON/YAML
        },

        /**
         * Search products
         */
        onSearch: function (oEvent) {
            const sQuery = oEvent.getParameter("query");
            const oTable = this.byId("productsTable");
            const oBinding = oTable.getBinding("items");
            
            if (sQuery) {
                const aFilters = [
                    new sap.ui.model.Filter("name", sap.ui.model.FilterOperator.Contains, sQuery)
                ];
                oBinding.filter(aFilters);
            } else {
                oBinding.filter([]);
            }
        },

        /**
         * Filter by category
         */
        onCategoryChange: function (oEvent) {
            const sKey = oEvent.getParameter("item").getKey();
            const oTable = this.byId("productsTable");
            const oBinding = oTable.getBinding("items");
            
            if (sKey === "all") {
                oBinding.filter([]);
            } else {
                const aFilters = [
                    new sap.ui.model.Filter("category", sap.ui.model.FilterOperator.EQ, sKey)
                ];
                oBinding.filter(aFilters);
            }
        },

        /**
         * Product selected
         */
        onProductSelect: function (oEvent) {
            const oItem = oEvent.getParameter("listItem");
            const oContext = oItem.getBindingContext("products");
            const oProduct = oContext.getObject();
            
            // Update selected product model
            const oModel = new JSONModel(oProduct);
            this.getView().setModel(oModel, "selectedProduct");
            
            // Expand details panel
            this.byId("detailsPanel").setExpanded(true);
        },

        /**
         * View product details
         */
        onViewDetails: function (oEvent) {
            const oItem = oEvent.getSource().getParent().getParent();
            const oContext = oItem.getBindingContext("products");
            const oProduct = oContext.getObject();
            
            MessageBox.information(
                `Product: ${oProduct.name}\n` +
                `ID: ${oProduct.productID}\n` +
                `Quality: ${oProduct.qualityScore}%\n` +
                `Category: ${oProduct.category}`,
                {
                    title: "ODPS Product Details"
                }
            );
        },

        /**
         * View lineage graph
         */
        onViewLineage: function (oEvent) {
            const oItem = oEvent.getSource().getParent().getParent();
            const oContext = oItem.getBindingContext("products");
            const oProduct = oContext.getObject();
            
            this.getOwnerComponent().getRouter().navTo("lineageGraph", {
                productID: encodeURIComponent(oProduct.productID)
            });
        },

        /**
         * View in ORD format
         */
        onViewORD: function (oEvent) {
            const oItem = oEvent.getSource().getParent().getParent();
            const oContext = oItem.getBindingContext("products");
            const oProduct = oContext.getObject();
            
            // Navigate to metadata view filtered to this product
            this.getOwnerComponent().getRouter().navTo("metadata");
        },

        /**
         * Navigate to quality dashboard
         */
        onViewQualityDashboard: function () {
            this.getOwnerComponent().getRouter().navTo("qualityDashboard");
        },

        /**
         * Navigate to data lineage
         */
        onViewDataLineage: function () {
            this.getOwnerComponent().getRouter().navTo("lineageGraph");
        },

        /**
         * Product pressed
         */
        onProductPress: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("products");
            const oProduct = oContext.getObject();
            
            // Update selected product
            const oModel = new JSONModel(oProduct);
            this.getView().setModel(oModel, "selectedProduct");
            
            // Expand details
            this.byId("detailsPanel").setExpanded(true);
        },

        /**
         * Endpoint pressed
         */
        onEndpointPress: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("selectedProduct");
            const oEndpoint = oContext.getObject();
            
            MessageToast.show(`Opening: ${oEndpoint.endpoint}`);
            // TODO: Open API endpoint in new tab or test interface
        },

        /**
         * Navigate back
         */
        onNavBack: function () {
            this.getOwnerComponent().getRouter().navTo("home");
        }
    });
});