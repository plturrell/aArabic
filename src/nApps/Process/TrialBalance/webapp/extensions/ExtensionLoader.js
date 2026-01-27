sap.ui.define([
    "sap/ui/base/Object",
    "sap/base/Log",
    "trialbalance/extensions/ExtensionManager"
], function(BaseObject, Log, ExtensionManager) {
    "use strict";

    /**
     * Extension Loader
     * Discovers, loads, and manages extensions dynamically.
     * Supports hot-reload and extension marketplace discovery.
     * 
     * Features:
     * - Automatic extension discovery
     * - Lazy loading of extensions
     * - Hot-reload support
     * - Extension manifest parsing
     * - Dependency resolution
     * - Backend extension registration
     * 
     * @class
     * @extends sap.ui.base.Object
     */
    var ExtensionLoader = BaseObject.extend("trialbalance.extensions.ExtensionLoader", {
        
        constructor: function(mSettings) {
            BaseObject.call(this);
            
            this._extensionManager = mSettings && mSettings.extensionManager 
                ? mSettings.extensionManager 
                : new ExtensionManager();
            
            this._loadedExtensions = new Map();
            this._pendingLoads = new Map();
            this._manifestCache = new Map();
            this._watchInterval = null;
            this._backendUrl = mSettings && mSettings.backendUrl 
                ? mSettings.backendUrl 
                : "/api/v1/extensions";
            
            Log.info("ExtensionLoader initialized", "trialbalance.extensions.ExtensionLoader");
        },

        /**
         * Get the extension manager
         * @returns {Object} Extension manager instance
         */
        getExtensionManager: function() {
            return this._extensionManager;
        },

        /**
         * Discover and load all registered extensions
         * @returns {Promise} Promise resolving when all extensions are loaded
         */
        discoverAndLoad: function() {
            Log.info("Discovering extensions...", "trialbalance.extensions.ExtensionLoader");
            
            return Promise.all([
                this._discoverFrontendExtensions(),
                this._discoverBackendExtensions()
            ]).then(function(results) {
                const frontendManifests = results[0] || [];
                const backendManifests = results[1] || [];
                
                // Merge and deduplicate
                const allManifests = this._mergeManifests(frontendManifests, backendManifests);
                
                Log.info("Found " + allManifests.length + " extensions", 
                        "trialbalance.extensions.ExtensionLoader");
                
                // Load extensions in dependency order
                return this._loadExtensionsInOrder(allManifests);
            }.bind(this));
        },

        /**
         * Load a specific extension by ID
         * @param {string} sExtensionId - Extension ID
         * @returns {Promise} Promise resolving to loaded extension
         */
        loadExtension: function(sExtensionId) {
            // Check if already loaded
            if (this._loadedExtensions.has(sExtensionId)) {
                return Promise.resolve(this._loadedExtensions.get(sExtensionId));
            }
            
            // Check if load is pending
            if (this._pendingLoads.has(sExtensionId)) {
                return this._pendingLoads.get(sExtensionId);
            }
            
            // Start loading
            const loadPromise = this._loadExtensionById(sExtensionId);
            this._pendingLoads.set(sExtensionId, loadPromise);
            
            return loadPromise.finally(function() {
                this._pendingLoads.delete(sExtensionId);
            }.bind(this));
        },

        /**
         * Unload an extension
         * @param {string} sExtensionId - Extension ID
         */
        unloadExtension: function(sExtensionId) {
            const extension = this._loadedExtensions.get(sExtensionId);
            if (!extension) return;
            
            // Unregister from extension manager
            this._extensionManager.unregisterExtension(sExtensionId);
            
            // Remove from loaded extensions
            this._loadedExtensions.delete(sExtensionId);
            
            Log.info("Extension unloaded: " + sExtensionId, "trialbalance.extensions.ExtensionLoader");
        },

        /**
         * Reload an extension (hot-reload)
         * @param {string} sExtensionId - Extension ID
         * @returns {Promise} Promise resolving when extension is reloaded
         */
        reloadExtension: function(sExtensionId) {
            Log.info("Hot-reloading extension: " + sExtensionId, "trialbalance.extensions.ExtensionLoader");
            
            // Unload current
            this.unloadExtension(sExtensionId);
            
            // Clear manifest cache
            this._manifestCache.delete(sExtensionId);
            
            // Reload
            return this.loadExtension(sExtensionId);
        },

        /**
         * Enable hot-reload watching
         * @param {number} nInterval - Check interval in milliseconds (default: 5000)
         */
        enableHotReload: function(nInterval) {
            if (this._watchInterval) {
                this.disableHotReload();
            }
            
            const interval = nInterval || 5000;
            
            this._watchInterval = setInterval(function() {
                this._checkForUpdates();
            }.bind(this), interval);
            
            Log.info("Hot-reload enabled with " + interval + "ms interval", 
                    "trialbalance.extensions.ExtensionLoader");
        },

        /**
         * Disable hot-reload watching
         */
        disableHotReload: function() {
            if (this._watchInterval) {
                clearInterval(this._watchInterval);
                this._watchInterval = null;
                Log.info("Hot-reload disabled", "trialbalance.extensions.ExtensionLoader");
            }
        },

        /**
         * Register built-in extensions
         * @returns {Promise} Promise resolving when core extensions are loaded
         */
        registerCoreExtensions: function() {
            Log.info("Registering core extensions", "trialbalance.extensions.ExtensionLoader");
            
            // Define core extension modules to load
            const aCoreExtensions = [
                "trialbalance/extensions/components/NetworkGraphExtension",
                "trialbalance/extensions/components/ProcessFlowExtension",
                "trialbalance/extensions/components/WorkflowBuilderExtension",
                "trialbalance/extensions/components/ChartsExtension"
            ];
            
            // Load each core extension
            const aLoadPromises = aCoreExtensions.map(function(sModulePath) {
                return new Promise(function(resolve, reject) {
                    sap.ui.require([sModulePath], function(Extension) {
                        try {
                            const oExtension = new Extension();
                            
                            // Set extension manager reference
                            if (oExtension.setExtensionManager) {
                                oExtension.setExtensionManager(this._extensionManager);
                            }
                            
                            // Get extension config and register
                            const config = oExtension.getExtensionConfig();
                            this._extensionManager.registerExtension(config);
                            
                            // Store reference
                            this._loadedExtensions.set(config.id, oExtension);
                            
                            Log.info("Core extension registered: " + config.id, 
                                    "trialbalance.extensions.ExtensionLoader");
                            
                            resolve(oExtension);
                        } catch (e) {
                            Log.error("Failed to register core extension: " + sModulePath, 
                                     e.message, "trialbalance.extensions.ExtensionLoader");
                            reject(e);
                        }
                    }.bind(this), reject);
                }.bind(this));
            }.bind(this));
            
            return Promise.all(aLoadPromises);
        },

        /**
         * Get list of loaded extensions
         * @returns {Array} Array of extension info objects
         */
        getLoadedExtensions: function() {
            const extensions = [];
            
            this._loadedExtensions.forEach(function(ext, id) {
                extensions.push({
                    id: id,
                    name: ext.getName ? ext.getName() : id,
                    version: ext.getVersion ? ext.getVersion() : "unknown",
                    enabled: ext.getEnabled ? ext.getEnabled() : true
                });
            });
            
            return extensions;
        },

        /**
         * Get extension by ID
         * @param {string} sExtensionId - Extension ID
         * @returns {Object|undefined} Extension instance
         */
        getExtension: function(sExtensionId) {
            return this._loadedExtensions.get(sExtensionId);
        },

        // ========== Private Methods ==========

        /**
         * Discover frontend extensions from manifest files
         * @private
         */
        _discoverFrontendExtensions: function() {
            // In production, this would scan for extension manifests
            // For now, return known extension locations
            return Promise.resolve([
                {
                    id: "network-graph-core",
                    module: "trialbalance/extensions/components/NetworkGraphExtension",
                    type: "frontend"
                },
                {
                    id: "process-flow-core",
                    module: "trialbalance/extensions/components/ProcessFlowExtension",
                    type: "frontend"
                },
                {
                    id: "workflow-builder-core",
                    module: "trialbalance/extensions/components/WorkflowBuilderExtension",
                    type: "frontend"
                },
                {
                    id: "charts-core",
                    module: "trialbalance/extensions/components/ChartsExtension",
                    type: "frontend"
                }
            ]);
        },

        /**
         * Discover backend extensions from API
         * @private
         */
        _discoverBackendExtensions: function() {
            return fetch(this._backendUrl + "/discover")
                .then(function(response) { 
                    if (!response.ok) throw new Error("Discovery failed");
                    return response.json(); 
                })
                .then(function(data) {
                    return data.extensions || [];
                })
                .catch(function(error) {
                    Log.warning("Backend extension discovery failed: " + error.message, 
                              "trialbalance.extensions.ExtensionLoader");
                    return [];
                });
        },

        /**
         * Merge frontend and backend manifests
         * @private
         */
        _mergeManifests: function(frontendManifests, backendManifests) {
            const merged = new Map();
            
            // Add frontend manifests
            frontendManifests.forEach(function(m) {
                merged.set(m.id, m);
            });
            
            // Merge backend manifests
            backendManifests.forEach(function(m) {
                if (merged.has(m.id)) {
                    // Merge properties
                    const existing = merged.get(m.id);
                    merged.set(m.id, Object.assign({}, existing, m, {
                        type: ["frontend", "backend"]
                    }));
                } else {
                    merged.set(m.id, m);
                }
            });
            
            return Array.from(merged.values());
        },

        /**
         * Load extensions in dependency order
         * @private
         */
        _loadExtensionsInOrder: function(aManifests) {
            // Sort by dependencies (topological sort)
            const sorted = this._topologicalSort(aManifests);
            
            // Load in sequence
            return sorted.reduce(function(promise, manifest) {
                return promise.then(function() {
                    return this._loadExtensionFromManifest(manifest);
                }.bind(this));
            }.bind(this), Promise.resolve());
        },

        /**
         * Topological sort for dependency ordering
         * @private
         */
        _topologicalSort: function(manifests) {
            const sorted = [];
            const visited = new Set();
            const visiting = new Set();
            
            const visit = function(manifest) {
                if (visited.has(manifest.id)) return;
                if (visiting.has(manifest.id)) {
                    Log.warning("Circular dependency detected: " + manifest.id, 
                              "trialbalance.extensions.ExtensionLoader");
                    return;
                }
                
                visiting.add(manifest.id);
                
                // Visit dependencies first
                if (manifest.dependencies) {
                    manifest.dependencies.forEach(function(depId) {
                        const dep = manifests.find(function(m) { return m.id === depId; });
                        if (dep) visit(dep);
                    });
                }
                
                visiting.delete(manifest.id);
                visited.add(manifest.id);
                sorted.push(manifest);
            }.bind(this);
            
            manifests.forEach(visit);
            
            return sorted;
        },

        /**
         * Load extension from manifest
         * @private
         */
        _loadExtensionFromManifest: function(manifest) {
            if (this._loadedExtensions.has(manifest.id)) {
                return Promise.resolve(this._loadedExtensions.get(manifest.id));
            }
            
            Log.debug("Loading extension: " + manifest.id, "trialbalance.extensions.ExtensionLoader");
            
            // Cache manifest
            this._manifestCache.set(manifest.id, manifest);
            
            // Load based on type
            if (manifest.module) {
                return new Promise(function(resolve, reject) {
                    sap.ui.require([manifest.module], function(Extension) {
                        try {
                            const oExtension = new Extension();
                            
                            // Set extension manager reference
                            if (oExtension.setExtensionManager) {
                                oExtension.setExtensionManager(this._extensionManager);
                            }
                            
                            // Get config and register
                            const config = oExtension.getExtensionConfig();
                            this._extensionManager.registerExtension(config);
                            
                            // Store reference
                            this._loadedExtensions.set(manifest.id, oExtension);
                            
                            Log.info("Extension loaded: " + manifest.id, 
                                    "trialbalance.extensions.ExtensionLoader");
                            
                            resolve(oExtension);
                        } catch (e) {
                            Log.error("Failed to load extension: " + manifest.id, 
                                     e.message, "trialbalance.extensions.ExtensionLoader");
                            reject(e);
                        }
                    }.bind(this), reject);
                }.bind(this));
            }
            
            // Backend-only extension
            return this._registerBackendExtension(manifest);
        },

        /**
         * Load extension by ID
         * @private
         */
        _loadExtensionById: function(sExtensionId) {
            // Check manifest cache first
            if (this._manifestCache.has(sExtensionId)) {
                return this._loadExtensionFromManifest(this._manifestCache.get(sExtensionId));
            }
            
            // Fetch manifest from backend
            return fetch(this._backendUrl + "/" + sExtensionId + "/manifest")
                .then(function(response) { 
                    if (!response.ok) throw new Error("Extension not found");
                    return response.json(); 
                })
                .then(function(manifest) {
                    return this._loadExtensionFromManifest(manifest);
                }.bind(this));
        },

        /**
         * Register a backend-only extension
         * @private
         */
        _registerBackendExtension: function(manifest) {
            return fetch(this._backendUrl + "/" + manifest.id + "/register", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(manifest)
            }).then(function(response) {
                if (!response.ok) throw new Error("Registration failed");
                
                Log.info("Backend extension registered: " + manifest.id, 
                        "trialbalance.extensions.ExtensionLoader");
                
                // Create placeholder for backend extension
                this._loadedExtensions.set(manifest.id, { 
                    id: manifest.id, 
                    type: "backend",
                    manifest: manifest 
                });
                
                return null;
            }.bind(this)).catch(function(error) {
                Log.warning("Backend extension registration failed: " + manifest.id, 
                          "trialbalance.extensions.ExtensionLoader");
                return null;
            });
        },

        /**
         * Check for extension updates (hot-reload)
         * @private
         */
        _checkForUpdates: function() {
            // Check each loaded extension for updates
            this._loadedExtensions.forEach(function(ext, id) {
                this._checkExtensionUpdate(id);
            }.bind(this));
        },

        /**
         * Check if a specific extension has updates
         * @private
         */
        _checkExtensionUpdate: function(sExtensionId) {
            fetch(this._backendUrl + "/" + sExtensionId + "/version")
                .then(function(response) { 
                    if (!response.ok) return null;
                    return response.json(); 
                })
                .then(function(data) {
                    if (!data) return;
                    
                    const ext = this._loadedExtensions.get(sExtensionId);
                    const currentVersion = ext && ext.getVersion ? ext.getVersion() : null;
                    
                    if (data.version && currentVersion && data.version !== currentVersion) {
                        Log.info("Extension update available: " + sExtensionId + 
                                " (" + currentVersion + " -> " + data.version + ")", 
                                "trialbalance.extensions.ExtensionLoader");
                        
                        // Auto-reload if enabled
                        if (data.autoReload) {
                            this.reloadExtension(sExtensionId);
                        }
                    }
                }.bind(this))
                .catch(function() {
                    // Silently ignore version check failures
                });
        },

        /**
         * Destroy the loader and all extensions
         */
        destroy: function() {
            Log.info("Destroying ExtensionLoader", "trialbalance.extensions.ExtensionLoader");
            
            // Disable hot-reload
            this.disableHotReload();
            
            // Unload all extensions
            this._loadedExtensions.forEach(function(ext, id) {
                this.unloadExtension(id);
            }.bind(this));
            
            // Clear caches
            this._manifestCache.clear();
            this._pendingLoads.clear();
            
            // Destroy extension manager
            if (this._extensionManager) {
                this._extensionManager.destroy();
                this._extensionManager = null;
            }
            
            BaseObject.prototype.destroy.call(this);
        }
    });

    return ExtensionLoader;
});