sap.ui.define([
    "sap/ui/base/Object",
    "sap/ui/model/odata/v4/ODataModel",
    "sap/ui/model/json/JSONModel",
    "sap/base/Log"
], function (BaseObject, ODataModel, JSONModel, Log) {
    "use strict";

    var STORAGE_KEY_PREFIX = "nLeanProof.";
    var STORAGE_KEYS = {
        LEAN_FILES: STORAGE_KEY_PREFIX + "leanFiles",
        COMPILE_RESULTS: STORAGE_KEY_PREFIX + "compileResults",
        CHAT_MESSAGES: STORAGE_KEY_PREFIX + "chatMessages"
    };

    return BaseObject.extend("nLeanProof.webapp.model.dataService", {
        _oODataModel: null,
        _bOfflineMode: false,

        constructor: function (oComponent) {
            BaseObject.call(this);
            this._oComponent = oComponent;
            this._initializeService();
        },

        _initializeService: function () {
            try {
                this._oODataModel = this._oComponent.getModel("odata");
                if (this._oODataModel) {
                    this._oODataModel.attachRequestFailed(this._onRequestFailed.bind(this));
                    Log.info("DataService: OData model initialized");
                } else {
                    this._enableOfflineMode();
                }
            } catch (oError) {
                Log.warning("DataService: OData not available, using offline mode", oError);
                this._enableOfflineMode();
            }
        },

        _onRequestFailed: function (oEvent) {
            Log.warning("DataService: OData request failed, switching to offline mode");
            this._enableOfflineMode();
        },

        _enableOfflineMode: function () {
            this._bOfflineMode = true;
            Log.info("DataService: Offline mode enabled");
        },

        isOfflineMode: function () {
            return this._bOfflineMode;
        },

        // ============================================
        // LeanFile CRUD Operations
        // ============================================

        getLeanFiles: function () {
            if (this._bOfflineMode) {
                return Promise.resolve(this._getFromStorage(STORAGE_KEYS.LEAN_FILES) || []);
            }
            return this._oODataModel.bindList("/LeanFiles").requestContexts().then(function (aContexts) {
                return aContexts.map(function (oContext) {
                    return oContext.getObject();
                });
            }).catch(function () {
                return this._getFromStorage(STORAGE_KEYS.LEAN_FILES) || [];
            }.bind(this));
        },

        getLeanFile: function (sId) {
            if (this._bOfflineMode) {
                var aFiles = this._getFromStorage(STORAGE_KEYS.LEAN_FILES) || [];
                return Promise.resolve(aFiles.find(function (f) { return f.id === sId; }));
            }
            return this._oODataModel.bindContext("/LeanFiles('" + sId + "')").requestObject()
                .catch(function () {
                    var aFiles = this._getFromStorage(STORAGE_KEYS.LEAN_FILES) || [];
                    return aFiles.find(function (f) { return f.id === sId; });
                }.bind(this));
        },

        createLeanFile: function (oData) {
            oData.id = oData.id || this._generateId();
            oData.lastModified = new Date().toISOString();

            if (this._bOfflineMode) {
                return this._createInStorage(STORAGE_KEYS.LEAN_FILES, oData);
            }
            var oListBinding = this._oODataModel.bindList("/LeanFiles");
            return oListBinding.create(oData).created().catch(function () {
                return this._createInStorage(STORAGE_KEYS.LEAN_FILES, oData);
            }.bind(this));
        },

        updateLeanFile: function (sId, oData) {
            oData.lastModified = new Date().toISOString();

            if (this._bOfflineMode) {
                return this._updateInStorage(STORAGE_KEYS.LEAN_FILES, sId, oData);
            }
            return this._oODataModel.bindContext("/LeanFiles('" + sId + "')")
                .requestObject().then(function (oContext) {
                    Object.keys(oData).forEach(function (sKey) {
                        oContext.setProperty(sKey, oData[sKey]);
                    });
                    return this._oODataModel.submitBatch("updateGroup");
                }.bind(this)).catch(function () {
                    return this._updateInStorage(STORAGE_KEYS.LEAN_FILES, sId, oData);
                }.bind(this));
        },

        deleteLeanFile: function (sId) {
            if (this._bOfflineMode) {
                return this._deleteFromStorage(STORAGE_KEYS.LEAN_FILES, sId);
            }
            return this._oODataModel.bindContext("/LeanFiles('" + sId + "')")
                .requestObject().then(function (oContext) {
                    oContext.delete();
                    return this._oODataModel.submitBatch("updateGroup");
                }.bind(this)).catch(function () {
                    return this._deleteFromStorage(STORAGE_KEYS.LEAN_FILES, sId);
                }.bind(this));
        },

        // ============================================
        // LocalStorage Helpers
        // ============================================

        _getFromStorage: function (sKey) {
            try {
                var sData = localStorage.getItem(sKey);
                return sData ? JSON.parse(sData) : null;
            } catch (e) {
                Log.error("DataService: Failed to read from localStorage", e);
                return null;
            }
        },

        _saveToStorage: function (sKey, aData) {
            try {
                localStorage.setItem(sKey, JSON.stringify(aData));
                return true;
            } catch (e) {
                Log.error("DataService: Failed to write to localStorage", e);
                return false;
            }
        },

        _createInStorage: function (sKey, oData) {
            var aItems = this._getFromStorage(sKey) || [];
            aItems.push(oData);
            this._saveToStorage(sKey, aItems);
            return Promise.resolve(oData);
        },

        _updateInStorage: function (sKey, sId, oData) {
            var aItems = this._getFromStorage(sKey) || [];
            var iIndex = aItems.findIndex(function (item) { return item.id === sId; });
            if (iIndex >= 0) {
                aItems[iIndex] = Object.assign({}, aItems[iIndex], oData);
                this._saveToStorage(sKey, aItems);
                return Promise.resolve(aItems[iIndex]);
            }
            return Promise.reject(new Error("Item not found: " + sId));
        },

        _deleteFromStorage: function (sKey, sId) {
            var aItems = this._getFromStorage(sKey) || [];
            var aFiltered = aItems.filter(function (item) { return item.id !== sId; });
            this._saveToStorage(sKey, aFiltered);
            return Promise.resolve(true);
        },

        _generateId: function () {
            return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
                var r = Math.random() * 16 | 0;
                var v = c === "x" ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }
    });
});

