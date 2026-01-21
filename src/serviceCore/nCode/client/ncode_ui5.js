/**
 * nCode Client Library for SAPUI5
 * Provides a UI5-compatible API client for the nCode SCIP-based code intelligence platform
 * 
 * Usage in UI5 Controller:
 *   const client = new NCodeClient("http://localhost:18003");
 *   client.health().then(health => {
 *       console.log("Status:", health.status);
 *   });
 */

sap.ui.define([
    "sap/ui/base/Object",
    "sap/ui/model/json/JSONModel"
], function(BaseObject, JSONModel) {
    "use strict";

    /**
     * Main nCode client for SAPUI5 applications
     * @class
     * @extends sap.ui.base.Object
     */
    return BaseObject.extend("ncode.client.NCodeClient", {

        /**
         * Constructor
         * @param {string} baseUrl - Base URL of the nCode server (default: http://localhost:18003)
         * @param {number} timeoutMs - Request timeout in milliseconds (default: 30000)
         */
        constructor: function(baseUrl, timeoutMs) {
            this._baseUrl = baseUrl || "http://localhost:18003";
            this._timeoutMs = timeoutMs || 30000;
            this._model = new JSONModel();
        },

        /**
         * Get the JSON model for data binding
         * @returns {sap.ui.model.json.JSONModel}
         */
        getModel: function() {
            return this._model;
        },

        /**
         * Make an HTTP request
         * @private
         * @param {string} method - HTTP method (GET, POST, etc.)
         * @param {string} endpoint - API endpoint
         * @param {object} data - Request payload
         * @returns {Promise}
         */
        _request: function(method, endpoint, data) {
            const url = this._baseUrl + endpoint;
            const settings = {
                url: url,
                type: method,
                contentType: "application/json",
                dataType: "json",
                timeout: this._timeoutMs
            };

            if (data) {
                settings.data = JSON.stringify(data);
            }

            return new Promise((resolve, reject) => {
                jQuery.ajax(settings)
                    .done(resolve)
                    .fail(function(jqXHR, textStatus, errorThrown) {
                        reject(new Error(`Request failed: ${textStatus} - ${errorThrown}`));
                    });
            });
        },

        /**
         * Check the health of the nCode server
         * @returns {Promise<object>} Health status
         */
        health: function() {
            return this._request("GET", "/health");
        },

        /**
         * Load a SCIP index file
         * @param {string} scipPath - Path to the SCIP index file
         * @returns {Promise<object>} Load result
         */
        loadIndex: function(scipPath) {
            return this._request("POST", "/v1/index/load", { path: scipPath });
        },

        /**
         * Find the definition of a symbol
         * @param {string} file - File path
         * @param {number} line - Line number (0-indexed)
         * @param {number} character - Character offset (0-indexed)
         * @returns {Promise<object>} Definition location
         */
        findDefinition: function(file, line, character) {
            return this._request("POST", "/v1/definition", {
                file: file,
                line: line,
                character: character
            });
        },

        /**
         * Find all references to a symbol
         * @param {string} file - File path
         * @param {number} line - Line number (0-indexed)
         * @param {number} character - Character offset (0-indexed)
         * @param {boolean} includeDeclaration - Include declaration in results
         * @returns {Promise<object>} References
         */
        findReferences: function(file, line, character, includeDeclaration) {
            return this._request("POST", "/v1/references", {
                file: file,
                line: line,
                character: character,
                include_declaration: includeDeclaration !== false
            });
        },

        /**
         * Get hover information for a symbol
         * @param {string} file - File path
         * @param {number} line - Line number (0-indexed)
         * @param {number} character - Character offset (0-indexed)
         * @returns {Promise<object>} Hover information
         */
        getHover: function(file, line, character) {
            return this._request("POST", "/v1/hover", {
                file: file,
                line: line,
                character: character
            });
        },

        /**
         * Get all symbols in a file
         * @param {string} filePath - File path
         * @returns {Promise<object>} Symbol list
         */
        getSymbols: function(filePath) {
            return this._request("POST", "/v1/symbols", { file: filePath });
        },

        /**
         * Get document outline (hierarchical symbol tree)
         * @param {string} filePath - File path
         * @returns {Promise<object>} Document symbols
         */
        getDocumentSymbols: function(filePath) {
            return this._request("POST", "/v1/document-symbols", { file: filePath });
        },

        /**
         * Load symbols into the model for UI binding
         * @param {string} filePath - File path
         * @returns {Promise<sap.ui.model.json.JSONModel>}
         */
        loadSymbolsModel: function(filePath) {
            return this.getSymbols(filePath).then((data) => {
                this._model.setData({
                    symbols: data.symbols || [],
                    file: data.file
                });
                return this._model;
            });
        }
    });
});

/**
 * Qdrant Client Helper for SAPUI5
 * @class
 */
sap.ui.define([
    "sap/ui/base/Object"
], function(BaseObject) {
    "use strict";

    return BaseObject.extend("ncode.client.QdrantClient", {

        /**
         * Constructor
         * @param {string} baseUrl - Qdrant server URL
         * @param {string} collectionName - Collection name
         */
        constructor: function(baseUrl, collectionName) {
            this._baseUrl = baseUrl || "http://localhost:6333";
            this._collectionName = collectionName || "ncode";
        },

        /**
         * Perform semantic search
         * @param {string} query - Search query
         * @param {number} limit - Maximum results
         * @returns {Promise<object>}
         */
        semanticSearch: function(query, limit) {
            const url = `${this._baseUrl}/collections/${this._collectionName}/points/search`;
            // Note: In production, you would embed the query first
            const payload = {
                vector: [],
                limit: limit || 10,
                with_payload: true
            };

            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: url,
                    type: "POST",
                    contentType: "application/json",
                    dataType: "json",
                    data: JSON.stringify(payload)
                }).done(resolve).fail((jqXHR, textStatus, errorThrown) => {
                    reject(new Error(`Search failed: ${textStatus} - ${errorThrown}`));
                });
            });
        },

        /**
         * Filter symbols by language
         * @param {string} language - Programming language
         * @param {number} limit - Maximum results
         * @returns {Promise<object>}
         */
        filterByLanguage: function(language, limit) {
            const url = `${this._baseUrl}/collections/${this._collectionName}/points/scroll`;
            const payload = {
                filter: {
                    must: [{
                        key: "language",
                        match: { value: language }
                    }]
                },
                limit: limit || 10,
                with_payload: true
            };

            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: url,
                    type: "POST",
                    contentType: "application/json",
                    dataType: "json",
                    data: JSON.stringify(payload)
                }).done(resolve).fail((jqXHR, textStatus, errorThrown) => {
                    reject(new Error(`Filter failed: ${textStatus} - ${errorThrown}`));
                });
            });
        }
    });
});

/**
 * Example UI5 View (XML)
 * Save as: view/CodeSearch.view.xml
 * 
 * <mvc:View
 *     xmlns:mvc="sap.ui.core.mvc"
 *     xmlns="sap.m"
 *     controllerName="ncode.controller.CodeSearch">
 *     <Page title="nCode - Code Search">
 *         <content>
 *             <VBox class="sapUiSmallMargin">
 *                 <Label text="File Path:" />
 *                 <Input id="filePathInput" placeholder="src/main.js" />
 *                 
 *                 <Button text="Load Symbols" press="onLoadSymbols" />
 *                 
 *                 <Table id="symbolsTable" items="{/symbols}">
 *                     <columns>
 *                         <Column><Text text="Name" /></Column>
 *                         <Column><Text text="Kind" /></Column>
 *                         <Column><Text text="Line" /></Column>
 *                     </columns>
 *                     <items>
 *                         <ColumnListItem>
 *                             <cells>
 *                                 <Text text="{name}" />
 *                                 <Text text="{kind}" />
 *                                 <Text text="{range/start/line}" />
 *                             </cells>
 *                         </ColumnListItem>
 *                     </items>
 *                 </Table>
 *             </VBox>
 *         </content>
 *     </Page>
 * </mvc:View>
 */

/**
 * Example UI5 Controller
 * Save as: controller/CodeSearch.controller.js
 * 
 * sap.ui.define([
 *     "sap/ui/core/mvc/Controller",
 *     "ncode/client/ncode_ui5"
 * ], function(Controller, NCodeClient) {
 *     "use strict";
 * 
 *     return Controller.extend("ncode.controller.CodeSearch", {
 * 
 *         onInit: function() {
 *             this._client = new NCodeClient("http://localhost:18003");
 *             
 *             // Set the model to the view
 *             this.getView().setModel(this._client.getModel());
 *             
 *             // Check health on init
 *             this._client.health().then((health) => {
 *                 console.log("nCode Status:", health.status);
 *             }).catch((error) => {
 *                 console.error("Health check failed:", error);
 *             });
 *         },
 * 
 *         onLoadSymbols: function() {
 *             const filePath = this.byId("filePathInput").getValue();
 *             
 *             if (!filePath) {
 *                 sap.m.MessageToast.show("Please enter a file path");
 *                 return;
 *             }
 *             
 *             this.getView().setBusy(true);
 *             
 *             this._client.loadSymbolsModel(filePath)
 *                 .then(() => {
 *                     this.getView().setBusy(false);
 *                     sap.m.MessageToast.show("Symbols loaded successfully");
 *                 })
 *                 .catch((error) => {
 *                     this.getView().setBusy(false);
 *                     sap.m.MessageBox.error("Failed to load symbols: " + error.message);
 *                 });
 *         },
 * 
 *         onSymbolPress: function(oEvent) {
 *             const item = oEvent.getSource();
 *             const context = item.getBindingContext();
 *             const symbol = context.getObject();
 *             
 *             // Show symbol details or navigate to definition
 *             const line = symbol.range.start.line;
 *             const char = symbol.range.start.character;
 *             
 *             this._client.findDefinition(symbol.file, line, char)
 *                 .then((definition) => {
 *                     console.log("Definition:", definition);
 *                     // Navigate or show details
 *                 });
 *         }
 *     });
 * });
 */

/**
 * Minimal standalone HTML example (without full UI5 framework)
 * Save as: index.html
 */
const minimalExample = `
<!DOCTYPE html>
<html>
<head>
    <title>nCode Client Example</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
    <h1>nCode Code Search</h1>
    
    <div>
        <label>File Path:</label>
        <input type="text" id="filePath" value="src/main.js" />
        <button onclick="loadSymbols()">Load Symbols</button>
    </div>
    
    <div id="results"></div>
    
    <script>
        // Simplified client (without UI5 dependencies)
        class SimpleNCodeClient {
            constructor(baseUrl) {
                this.baseUrl = baseUrl || 'http://localhost:18003';
            }
            
            async getSymbols(filePath) {
                const response = await fetch(this.baseUrl + '/v1/symbols', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file: filePath })
                });
                return await response.json();
            }
        }
        
        const client = new SimpleNCodeClient();
        
        async function loadSymbols() {
            const filePath = document.getElementById('filePath').value;
            const results = document.getElementById('results');
            
            try {
                const data = await client.getSymbols(filePath);
                results.innerHTML = '<h2>Symbols:</h2><pre>' + 
                    JSON.stringify(data, null, 2) + '</pre>';
            } catch (error) {
                results.innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
            }
        }
    </script>
</body>
</html>
`;
