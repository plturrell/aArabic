sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "llm/server/dashboard/utils/ThemeService",
    "llm/server/dashboard/utils/KeyboardShortcuts"
], function (Controller, JSONModel, MessageBox, MessageToast, ThemeService, KeyboardShortcuts) {
    "use strict";

    return Controller.extend("llm.server.dashboard.controller.Settings", {

        // LocalStorage key for settings
        STORAGE_KEY: "nucleus_dashboard_settings",

        // Default settings configuration
        _defaultSettings: {
            general: {
                appName: "Nucleus OpenAI Server",
                language: "en",
                autoSave: true,
                autoSaveInterval: 30
            },
            server: {
                apiUrl: "http://localhost:11434",
                wsUrl: "ws://localhost:8080/ws",
                timeout: 30,
                maxRetries: 3,
                connectionStatus: "Not tested",
                connectionState: "None"
            },
            models: {
                defaultModel: "lfm2.5-1.2b-q4_0",
                fallbackModel: "none",
                temperature: 0.7,
                topP: 0.9,
                maxTokens: 4096
            },
            theme: {
                darkMode: false,
                accentColor: "default",
                fontSize: "medium",
                compactMode: true
            },
            advanced: {
                debugMode: false,
                loggingLevel: "info",
                enableCache: true,
                cacheSize: 100,
                cacheTTL: 60,
                cacheStatus: "",
                enableAnimations: true,
                lazyLoading: true
            },
            shortcuts: []
        },

        onInit: function () {
            // Load settings from localStorage
            var oSettings = this._loadSettings();
            
            // Initialize settings model
            this._oSettingsModel = new JSONModel(oSettings);
            this.getView().setModel(this._oSettingsModel, "settings");

            // Initialize dark mode from ThemeService
            var bIsDark = ThemeService.isDarkMode();
            this._oSettingsModel.setProperty("/theme/darkMode", bIsDark);

            // Load keyboard shortcuts
            this._loadShortcuts();

            // Auto-save on any property change
            this._oSettingsModel.attachPropertyChange(this._onSettingsChange.bind(this));
        },

        /**
         * Load settings from localStorage
         * @private
         * @returns {Object} Settings object
         */
        _loadSettings: function () {
            try {
                var sStoredSettings = window.localStorage.getItem(this.STORAGE_KEY);
                if (sStoredSettings) {
                    var oStoredSettings = JSON.parse(sStoredSettings);
                    // Merge with defaults to ensure all properties exist
                    return this._mergeSettings(this._defaultSettings, oStoredSettings);
                }
            } catch (e) {
                console.warn("Settings: Could not load from localStorage:", e);
            }
            return JSON.parse(JSON.stringify(this._defaultSettings));
        },

        /**
         * Merge stored settings with defaults
         * @private
         */
        _mergeSettings: function (oDefaults, oStored) {
            var oResult = JSON.parse(JSON.stringify(oDefaults));
            for (var sKey in oStored) {
                if (oStored.hasOwnProperty(sKey) && oResult.hasOwnProperty(sKey)) {
                    if (typeof oStored[sKey] === "object" && !Array.isArray(oStored[sKey])) {
                        oResult[sKey] = this._mergeSettings(oResult[sKey], oStored[sKey]);
                    } else {
                        oResult[sKey] = oStored[sKey];
                    }
                }
            }
            return oResult;
        },

        /**
         * Save settings to localStorage
         * @private
         */
        _saveSettings: function () {
            try {
                var oSettings = this._oSettingsModel.getData();
                // Don't save transient properties
                var oToSave = JSON.parse(JSON.stringify(oSettings));
                delete oToSave.server.connectionStatus;
                delete oToSave.server.connectionState;
                delete oToSave.advanced.cacheStatus;
                delete oToSave.shortcuts;
                
                window.localStorage.setItem(this.STORAGE_KEY, JSON.stringify(oToSave));
            } catch (e) {
                console.warn("Settings: Could not save to localStorage:", e);
            }
        },

        /**
         * Handle settings property change for auto-save
         * @private
         */
        _onSettingsChange: function () {
            if (this._oSettingsModel.getProperty("/general/autoSave")) {
                this._saveSettings();
            }
        },

        /**
         * Load keyboard shortcuts from KeyboardShortcuts service
         * @private
         */
        _loadShortcuts: function () {
            var oCategories = KeyboardShortcuts.getShortcutsByCategory();
            var aShortcuts = [];
            
            var mCategoryNames = {
                general: "General",
                navigation: "Navigation",
                zoom: "Zoom",
                selection: "Selection",
                workflow: "Workflow"
            };

            for (var sCategory in oCategories) {
                if (oCategories.hasOwnProperty(sCategory)) {
                    var sCategoryName = mCategoryNames[sCategory] || sCategory;
                    oCategories[sCategory].forEach(function (oShortcut) {
                        aShortcuts.push({
                            shortcut: oShortcut.shortcut,
                            description: oShortcut.description,
                            category: sCategoryName
                        });
                    });
                }
            }

            this._oSettingsModel.setProperty("/shortcuts", aShortcuts);
        },

        // ========================================================================
        // EVENT HANDLERS
        // ========================================================================

        onBreadcrumbHome: function () {
            var oNavContainer = this.getView().getParent().getParent();
            if (oNavContainer && oNavContainer.to) {
                oNavContainer.to("mainPageContent");
            }
        },

        onResetDefaults: function () {
            var that = this;
            MessageBox.confirm("Are you sure you want to reset all settings to defaults?", {
                title: "Reset Settings",
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        // Reset to defaults
                        var oDefaults = JSON.parse(JSON.stringify(that._defaultSettings));
                        oDefaults.theme.darkMode = ThemeService.isDarkMode();
                        that._oSettingsModel.setData(oDefaults);
                        that._loadShortcuts();
                        that._saveSettings();
                        MessageToast.show("Settings reset to defaults");
                    }
                }
            });
        },

        onExportSettings: function () {
            try {
                var oSettings = this._oSettingsModel.getData();
                var oToExport = JSON.parse(JSON.stringify(oSettings));
                delete oToExport.server.connectionStatus;
                delete oToExport.server.connectionState;
                delete oToExport.advanced.cacheStatus;
                delete oToExport.shortcuts;

                var sJson = JSON.stringify(oToExport, null, 2);
                var oBlob = new Blob([sJson], { type: "application/json" });
                var sUrl = URL.createObjectURL(oBlob);

                var oLink = document.createElement("a");
                oLink.href = sUrl;
                oLink.download = "nucleus_settings.json";
                document.body.appendChild(oLink);
                oLink.click();
                document.body.removeChild(oLink);
                URL.revokeObjectURL(sUrl);

                MessageToast.show("Settings exported successfully");
            } catch (e) {
                MessageBox.error("Failed to export settings: " + e.message);
            }
        },

        onImportSettings: function () {
            var that = this;
            var oFileInput = document.createElement("input");
            oFileInput.type = "file";
            oFileInput.accept = ".json";

            oFileInput.onchange = function (oEvent) {
                var oFile = oEvent.target.files[0];
                if (oFile) {
                    var oReader = new FileReader();
                    oReader.onload = function (oLoadEvent) {
                        try {
                            var oImported = JSON.parse(oLoadEvent.target.result);
                            var oMerged = that._mergeSettings(that._defaultSettings, oImported);
                            oMerged.theme.darkMode = ThemeService.isDarkMode();
                            that._oSettingsModel.setData(oMerged);
                            that._loadShortcuts();
                            that._saveSettings();
                            MessageToast.show("Settings imported successfully");
                        } catch (e) {
                            MessageBox.error("Failed to import settings: Invalid JSON file");
                        }
                    };
                    oReader.readAsText(oFile);
                }
            };

            oFileInput.click();
        },

        onTestConnection: function () {
            var that = this;
            var sApiUrl = this._oSettingsModel.getProperty("/server/apiUrl");

            this._oSettingsModel.setProperty("/server/connectionStatus", "Testing...");
            this._oSettingsModel.setProperty("/server/connectionState", "Warning");

            // Test the API connection
            fetch(sApiUrl + "/api/tags", {
                method: "GET",
                signal: AbortSignal.timeout(5000)
            })
            .then(function (oResponse) {
                if (oResponse.ok) {
                    that._oSettingsModel.setProperty("/server/connectionStatus", "Connected");
                    that._oSettingsModel.setProperty("/server/connectionState", "Success");
                    MessageToast.show("Connection successful!");
                } else {
                    throw new Error("HTTP " + oResponse.status);
                }
            })
            .catch(function (oError) {
                that._oSettingsModel.setProperty("/server/connectionStatus", "Failed: " + oError.message);
                that._oSettingsModel.setProperty("/server/connectionState", "Error");
            });
        },

        onDarkModeChange: function (oEvent) {
            var bDarkMode = oEvent.getParameter("state");
            var sNewTheme = bDarkMode ? ThemeService.THEMES.DARK : ThemeService.THEMES.LIGHT;
            ThemeService.setTheme(sNewTheme);
        },

        onClearCache: function () {
            var that = this;
            MessageBox.confirm("Are you sure you want to clear all cached data?", {
                title: "Clear Cache",
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        // Clear various caches
                        try {
                            // Clear application-specific cache keys
                            var aKeysToRemove = [];
                            for (var i = 0; i < window.localStorage.length; i++) {
                                var sKey = window.localStorage.key(i);
                                if (sKey && sKey.startsWith("nucleus_cache_")) {
                                    aKeysToRemove.push(sKey);
                                }
                            }
                            aKeysToRemove.forEach(function (sKey) {
                                window.localStorage.removeItem(sKey);
                            });

                            that._oSettingsModel.setProperty("/advanced/cacheStatus",
                                "Cache cleared at " + new Date().toLocaleTimeString());
                            MessageToast.show("Cache cleared successfully");
                        } catch (e) {
                            MessageBox.error("Failed to clear cache: " + e.message);
                        }
                    }
                }
            });
        },

        /**
         * Called when the controller is destroyed
         */
        onExit: function () {
            // Ensure settings are saved on exit
            this._saveSettings();
        }
    });
});

