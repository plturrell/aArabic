sap.ui.define([
    "sap/ui/base/Object",
    "sap/base/Log"
], function(BaseObject, Log) {
    "use strict";

    /**
     * Extension Manager
     * Central registry for managing UI5 extensions in the Trial Balance application
     * 
     * Features:
     * - Extension registration and lifecycle management
     * - Hook execution with priority support
     * - Extension dependency resolution
     * - Validation and error handling
     * 
     * @class
     * @extends sap.ui.base.Object
     */
    return BaseObject.extend("trialbalance.extensions.ExtensionManager", {
        
        constructor: function() {
            BaseObject.call(this);
            
            // Extension storage
            this._extensions = new Map();
            this._hooks = new Map();
            this._dependencies = new Map();
            
            // Lifecycle state
            this._initialized = false;
            
            Log.info("ExtensionManager initialized", "trialbalance.extensions.ExtensionManager");
        },

        /**
         * Initialize the extension manager and all registered extensions
         * @returns {Promise} Promise that resolves when all extensions are initialized
         */
        initialize: function() {
            if (this._initialized) {
                return Promise.resolve();
            }

            Log.info("Initializing extensions...", "trialbalance.extensions.ExtensionManager");

            const aInitPromises = [];
            
            this._extensions.forEach(function(oExtension) {
                if (oExtension.lifecycle && oExtension.lifecycle.init) {
                    try {
                        const result = oExtension.lifecycle.init();
                        if (result instanceof Promise) {
                            aInitPromises.push(result);
                        }
                    } catch (e) {
                        Log.error("Error initializing extension: " + oExtension.id, e.message, "trialbalance.extensions.ExtensionManager");
                    }
                }
            });

            this._initialized = true;

            return Promise.all(aInitPromises).then(function() {
                Log.info("All extensions initialized successfully", "trialbalance.extensions.ExtensionManager");
            });
        },

        /**
         * Register an extension
         * @param {Object} oExtension - Extension configuration object
         * @param {string} oExtension.id - Unique extension identifier
         * @param {string} oExtension.name - Human-readable extension name
         * @param {string} oExtension.version - Extension version (semver)
         * @param {string} [oExtension.type] - Extension type (component|controller|service)
         * @param {Object} [oExtension.lifecycle] - Lifecycle hooks
         * @param {Object} [oExtension.hooks] - Extension hooks
         * @param {Array<string>} [oExtension.dependencies] - Extension dependencies
         * @param {number} [oExtension.priority] - Execution priority (higher = earlier)
         * @returns {trialbalance.extensions.ExtensionManager} This instance for chaining
         */
        registerExtension: function(oExtension) {
            // Validation
            if (!oExtension || typeof oExtension !== 'object') {
                throw new Error("Extension must be an object");
            }

            if (!oExtension.id) {
                throw new Error("Extension must have an id");
            }

            if (this._extensions.has(oExtension.id)) {
                Log.warning("Extension already registered: " + oExtension.id, "trialbalance.extensions.ExtensionManager");
                return this;
            }

            // Set defaults
            oExtension.version = oExtension.version || "1.0.0";
            oExtension.type = oExtension.type || "component";
            oExtension.priority = oExtension.priority || 0;
            oExtension.dependencies = oExtension.dependencies || [];

            // Validate dependencies
            if (!this._validateDependencies(oExtension)) {
                throw new Error("Extension dependencies not met: " + oExtension.id);
            }

            // Register extension
            this._extensions.set(oExtension.id, oExtension);
            this._dependencies.set(oExtension.id, oExtension.dependencies);

            // Register hooks
            this._registerHooks(oExtension);

            Log.info("Extension registered: " + oExtension.id + " v" + oExtension.version, "trialbalance.extensions.ExtensionManager");

            return this;
        },

        /**
         * Unregister an extension
         * @param {string} sExtensionId - Extension ID to unregister
         * @returns {boolean} True if extension was unregistered
         */
        unregisterExtension: function(sExtensionId) {
            const oExtension = this._extensions.get(sExtensionId);
            
            if (!oExtension) {
                return false;
            }

            // Call destroy lifecycle hook
            if (oExtension.lifecycle && oExtension.lifecycle.destroy) {
                try {
                    oExtension.lifecycle.destroy();
                } catch (e) {
                    Log.error("Error destroying extension: " + sExtensionId, e.message, "trialbalance.extensions.ExtensionManager");
                }
            }

            // Unregister hooks
            this._unregisterHooks(oExtension);

            // Remove from registries
            this._extensions.delete(sExtensionId);
            this._dependencies.delete(sExtensionId);

            Log.info("Extension unregistered: " + sExtensionId, "trialbalance.extensions.ExtensionManager");

            return true;
        },

        /**
         * Get a registered extension by ID
         * @param {string} sExtensionId - Extension ID
         * @returns {Object|undefined} Extension object or undefined
         */
        getExtension: function(sExtensionId) {
            return this._extensions.get(sExtensionId);
        },

        /**
         * Get all registered extensions
         * @returns {Array<Object>} Array of extension objects
         */
        getAllExtensions: function() {
            return Array.from(this._extensions.values());
        },

        /**
         * Check if an extension is registered
         * @param {string} sExtensionId - Extension ID
         * @returns {boolean} True if extension is registered
         */
        hasExtension: function(sExtensionId) {
            return this._extensions.has(sExtensionId);
        },

        /**
         * Execute a hook with all registered handlers
         * @param {string} sHookName - Hook name
         * @param {Object} oContext - Hook context
         * @param {*} oContext.data - Data to be processed
         * @param {Object} [oContext.metadata] - Additional metadata
         * @returns {*} Processed data after all hooks
         */
        executeHook: function(sHookName, oContext) {
            const aHooks = this._hooks.get(sHookName);
            
            if (!aHooks || aHooks.length === 0) {
                return oContext.data;
            }

            Log.debug("Executing hook: " + sHookName + " (" + aHooks.length + " handlers)", "trialbalance.extensions.ExtensionManager");

            // Sort hooks by priority (higher priority first)
            const aSortedHooks = aHooks.slice().sort(function(a, b) {
                return b.priority - a.priority;
            });

            // Execute hooks in sequence
            let result = oContext.data;
            
            aSortedHooks.forEach(function(oHook) {
                try {
                    const newResult = oHook.handler.call(null, result, oContext);
                    if (newResult !== undefined) {
                        result = newResult;
                    }
                } catch (e) {
                    Log.error("Error executing hook: " + sHookName + " from extension: " + oHook.extensionId, 
                             e.message, "trialbalance.extensions.ExtensionManager");
                }
            });

            return result;
        },

        /**
         * Execute a hook asynchronously
         * @param {string} sHookName - Hook name
         * @param {Object} oContext - Hook context
         * @returns {Promise<*>} Promise resolving to processed data
         */
        executeHookAsync: function(sHookName, oContext) {
            const aHooks = this._hooks.get(sHookName);
            
            if (!aHooks || aHooks.length === 0) {
                return Promise.resolve(oContext.data);
            }

            Log.debug("Executing async hook: " + sHookName + " (" + aHooks.length + " handlers)", "trialbalance.extensions.ExtensionManager");

            // Sort hooks by priority
            const aSortedHooks = aHooks.slice().sort(function(a, b) {
                return b.priority - a.priority;
            });

            // Execute hooks in sequence (each waits for previous)
            return aSortedHooks.reduce(function(promise, oHook) {
                return promise.then(function(result) {
                    try {
                        const handlerResult = oHook.handler.call(null, result, oContext);
                        return Promise.resolve(handlerResult).then(function(newResult) {
                            return newResult !== undefined ? newResult : result;
                        });
                    } catch (e) {
                        Log.error("Error executing async hook: " + sHookName + " from extension: " + oHook.extensionId,
                                 e.message, "trialbalance.extensions.ExtensionManager");
                        return result;
                    }
                });
            }, Promise.resolve(oContext.data));
        },

        /**
         * Register hooks from an extension
         * @private
         */
        _registerHooks: function(oExtension) {
            if (!oExtension.hooks) {
                return;
            }

            Object.keys(oExtension.hooks).forEach(function(sHookName) {
                if (!this._hooks.has(sHookName)) {
                    this._hooks.set(sHookName, []);
                }

                this._hooks.get(sHookName).push({
                    extensionId: oExtension.id,
                    priority: oExtension.priority || 0,
                    handler: oExtension.hooks[sHookName]
                });

                Log.debug("Registered hook: " + sHookName + " from extension: " + oExtension.id, "trialbalance.extensions.ExtensionManager");
            }.bind(this));
        },

        /**
         * Unregister hooks from an extension
         * @private
         */
        _unregisterHooks: function(oExtension) {
            if (!oExtension.hooks) {
                return;
            }

            Object.keys(oExtension.hooks).forEach(function(sHookName) {
                const aHooks = this._hooks.get(sHookName);
                if (aHooks) {
                    const filtered = aHooks.filter(function(oHook) {
                        return oHook.extensionId !== oExtension.id;
                    });
                    
                    if (filtered.length === 0) {
                        this._hooks.delete(sHookName);
                    } else {
                        this._hooks.set(sHookName, filtered);
                    }
                }
            }.bind(this));
        },

        /**
         * Validate extension dependencies
         * @private
         */
        _validateDependencies: function(oExtension) {
            if (!oExtension.dependencies || oExtension.dependencies.length === 0) {
                return true;
            }

            for (let i = 0; i < oExtension.dependencies.length; i++) {
                const sDep = oExtension.dependencies[i];
                if (!this._extensions.has(sDep)) {
                    Log.error("Missing dependency: " + sDep + " for extension: " + oExtension.id, "trialbalance.extensions.ExtensionManager");
                    return false;
                }
            }

            return true;
        },

        /**
         * Destroy the extension manager and all extensions
         */
        destroy: function() {
            Log.info("Destroying ExtensionManager", "trialbalance.extensions.ExtensionManager");

            // Destroy all extensions
            this._extensions.forEach(function(oExtension) {
                if (oExtension.lifecycle && oExtension.lifecycle.destroy) {
                    try {
                        oExtension.lifecycle.destroy();
                    } catch (e) {
                        Log.error("Error destroying extension: " + oExtension.id, e.message, "trialbalance.extensions.ExtensionManager");
                    }
                }
            });

            // Clear all registries
            this._extensions.clear();
            this._hooks.clear();
            this._dependencies.clear();
            this._initialized = false;

            BaseObject.prototype.destroy.call(this);
        }
    });
});