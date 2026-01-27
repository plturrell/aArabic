/**
 * ============================================================================
 * ODPS Catalog Controller
 * Browse and explore ODPS data products and their metadata
 * ============================================================================
 *
 * [CODE:file=ODPSCatalog.controller.js]
 * [CODE:module=controller]
 * [CODE:language=javascript]
 *
 * [ODPS:product=trial-balance-aggregated]
 * [ODPS:product=variances]
 * [ODPS:product=exchange-rates]
 * [ODPS:product=acdoca-journal-entries]
 * [ODPS:product=account-master]
 *
 * [VIEW:binding=ODPSCatalog.view.xml]
 *
 * [API:consumes=/api/v1/odps/products]
 * [API:consumes=/api/v1/odps/products/{id}/rules]
 * [API:consumes=/api/v1/odps/products/{id}/lineage]
 *
 * [RELATION:uses=CODE:ApiService.js]
 * [RELATION:calls=CODE:odps_api.zig]
 * [RELATION:calls=CODE:odps_mapper.zig]
 *
 * This controller provides a catalog view of all ODPS data products,
 * their validation rules, field mappings, and lineage information.
 */
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "trialbalance/service/ApiService"
], function (Controller, JSONModel, MessageToast, ApiService) {
    "use strict";

    return Controller.extend("trialbalance.controller.ODPSCatalog", {

        /**
         * Controller initialization
         */
        onInit: function () {
            // Initialize API service
            this._oApiService = new ApiService();
            
            // Create view model
            var oViewModel = new JSONModel({
                busy: true,
                products: [],
                selectedProduct: null,
                productDetails: {
                    fields: [],
                    rules: [],
                    lineage: null
                },
                categories: [
                    { key: "primary", text: "Primary Data Products" },
                    { key: "operational", text: "Operational Data" },
                    { key: "metadata", text: "Metadata Products" }
                ],
                lastUpdated: null
            });
            this.getView().setModel(oViewModel, "view");
            
            // Load data when route is matched
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("odpsCatalog").attachPatternMatched(this._onRouteMatched, this);
        },

        /**
         * Route matched handler
         * @private
         */
        _onRouteMatched: function () {
            this._loadCatalog();
        },

        /**
         * Load ODPS catalog from API
         * @private
         */
        _loadCatalog: function () {
            var that = this;
            var oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/busy", true);
            
            this._oApiService.getODPSProducts()
                .then(function (aProducts) {
                    // Group products by category
                    var aTransformed = (aProducts || []).map(function (p) {
                        return {
                            id: p.id || p.name,
                            name: p.name,
                            description: p.description || "",
                            category: p.category || "primary",
                            version: p.version || "1.0",
                            fieldCount: p.field_count || 0,
                            ruleCount: p.rule_count || 0,
                            status: p.status || "active",
                            lastModified: p.last_modified ? new Date(p.last_modified) : null
                        };
                    });
                    
                    oViewModel.setProperty("/products", aTransformed);
                    oViewModel.setProperty("/lastUpdated", new Date());
                    oViewModel.setProperty("/busy", false);
                })
                .catch(function (oError) {
                    // Use fallback static data
                    oViewModel.setProperty("/products", that._getStaticProducts());
                    oViewModel.setProperty("/busy", false);
                });
        },

        /**
         * Get static products for offline mode
         * @private
         */
        _getStaticProducts: function () {
            return [
                {
                    id: "trial-balance-aggregated",
                    name: "Trial Balance Aggregated",
                    description: "Aggregated trial balance data with validation rules TB001-TB006",
                    category: "primary",
                    version: "1.0",
                    fieldCount: 15,
                    ruleCount: 6,
                    status: "active"
                },
                {
                    id: "variances",
                    name: "Variance Analysis",
                    description: "Period-over-period variance analysis with VAR001-VAR008 rules",
                    category: "primary",
                    version: "1.0",
                    fieldCount: 12,
                    ruleCount: 8,
                    status: "active"
                },
                {
                    id: "exchange-rates",
                    name: "Exchange Rates",
                    description: "Multi-currency exchange rates with FX001-FX007 validation",
                    category: "primary",
                    version: "1.0",
                    fieldCount: 8,
                    ruleCount: 7,
                    status: "active"
                },
                {
                    id: "acdoca-journal-entries",
                    name: "ACDOCA Journal Entries",
                    description: "Universal journal entries from SAP S/4HANA",
                    category: "primary",
                    version: "1.0",
                    fieldCount: 25,
                    ruleCount: 0,
                    status: "active"
                },
                {
                    id: "account-master",
                    name: "Account Master",
                    description: "G/L account master data with GCOA mapping",
                    category: "primary",
                    version: "1.0",
                    fieldCount: 10,
                    ruleCount: 2,
                    status: "active"
                },
                {
                    id: "checklist-items",
                    name: "Checklist Items",
                    description: "Maker/checker workflow checklist",
                    category: "operational",
                    version: "1.0",
                    fieldCount: 8,
                    ruleCount: 0,
                    status: "active"
                },
                {
                    id: "data-lineage",
                    name: "Data Lineage",
                    description: "SCIP-based code-to-data lineage tracking",
                    category: "metadata",
                    version: "1.0",
                    fieldCount: 6,
                    ruleCount: 0,
                    status: "active"
                },
                {
                    id: "dataset-metadata",
                    name: "Dataset Metadata",
                    description: "Metadata catalog for all data products",
                    category: "metadata",
                    version: "1.0",
                    fieldCount: 12,
                    ruleCount: 0,
                    status: "active"
                }
            ];
        },

        /**
         * Handle product selection
         */
        onProductSelect: function (oEvent) {
            var that = this;
            var oItem = oEvent.getParameter("listItem");
            var oContext = oItem.getBindingContext("view");
            var oProduct = oContext.getObject();
            
            var oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/selectedProduct", oProduct);
            oViewModel.setProperty("/productDetails", {
                fields: [],
                rules: [],
                lineage: null
            });
            
            // Load product details
            this._oApiService.getODPSRules(oProduct.id)
                .then(function (aRules) {
                    oViewModel.setProperty("/productDetails/rules", aRules || []);
                })
                .catch(function () {
                    // Use static rules
                    oViewModel.setProperty("/productDetails/rules", that._getStaticRules(oProduct.id));
                });
        },

        /**
         * Get static rules for a product
         * @private
         */
        _getStaticRules: function (sProductId) {
            var oRules = {
                "trial-balance-aggregated": [
                    { id: "TB001", name: "Balance Equation", severity: "error" },
                    { id: "TB002", name: "Debit Credit Balance", severity: "error" },
                    { id: "TB003", name: "IFRS Classification", severity: "warning" },
                    { id: "TB004", name: "Period Data Accuracy", severity: "error" },
                    { id: "TB005", name: "GCOA Mapping Completeness", severity: "error" },
                    { id: "TB006", name: "Global Mapping Currency", severity: "error" }
                ],
                "variances": [
                    { id: "VAR001", name: "Variance Calculation", severity: "error" },
                    { id: "VAR002", name: "Variance Percent", severity: "error" },
                    { id: "VAR003", name: "Materiality Threshold BS", severity: "warning" },
                    { id: "VAR004", name: "Materiality Threshold PL", severity: "warning" },
                    { id: "VAR005", name: "Commentary Required", severity: "warning" },
                    { id: "VAR006", name: "Commentary Coverage 90%", severity: "error" },
                    { id: "VAR007", name: "Exception Flagging", severity: "warning" },
                    { id: "VAR008", name: "Major Driver Identification", severity: "info" }
                ],
                "exchange-rates": [
                    { id: "FX001", name: "From Currency Mandatory", severity: "error" },
                    { id: "FX002", name: "To Currency Mandatory", severity: "error" },
                    { id: "FX003", name: "Rate Positive", severity: "error" },
                    { id: "FX004", name: "Ratio Positive", severity: "error" },
                    { id: "FX005", name: "Exchange Rate Verification", severity: "warning" },
                    { id: "FX006", name: "Period-Specific Rate", severity: "error" },
                    { id: "FX007", name: "Group Rate Source", severity: "error" }
                ]
            };
            return oRules[sProductId] || [];
        },

        /**
         * View lineage for selected product
         */
        onViewLineage: function () {
            var oProduct = this.getView().getModel("view").getProperty("/selectedProduct");
            if (oProduct) {
                // Navigate to lineage view with product context
                this.getOwnerComponent().getRouter().navTo("lineageGraph", {
                    productId: oProduct.id
                });
            }
        },

        /**
         * Get category text
         */
        getCategoryText: function (sCategory) {
            var oCategories = {
                "primary": "Primary Data Products",
                "operational": "Operational Data",
                "metadata": "Metadata Products"
            };
            return oCategories[sCategory] || sCategory;
        },

        /**
         * Get status state
         */
        getStatusState: function (sStatus) {
            return sStatus === "active" ? "Success" : "Warning";
        },

        /**
         * Refresh catalog
         */
        onRefresh: function () {
            this._loadCatalog();
            MessageToast.show("Catalog refreshed");
        },

        /**
         * Navigate back
         */
        onNavBack: function () {
            this.getOwnerComponent().getRouter().navTo("home");
        }

    });
});