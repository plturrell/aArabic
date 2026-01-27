sap.ui.define([
    "sap/ui/base/Object",
    "sap/base/Log"
], function(BaseObject, Log) {
    "use strict";

    /**
     * Component Extension Base Class
     * Base class for creating extensions that enhance or modify UI5 components
     * 
     * Features:
     * - Standardized extension interface
     * - Lifecycle management
     * - Hook registration
     * - Component wrapping and enhancement
     * 
     * @class
     * @extends sap.ui.base.Object
     * @abstract
     */
    return BaseObject.extend("trialbalance.extensions.ComponentExtension", {
        
        metadata: {
            abstract: true,
            properties: {
                /**
                 * Unique extension identifier
                 */
                id: { type: "string", defaultValue: "" },
                
                /**
                 * Extension name
                 */
                name: { type: "string", defaultValue: "" },
                
                /**
                 * Extension version (semver)
                 */
                version: { type: "string", defaultValue: "1.0.0" },
                
                /**
                 * Component class names that this extension applies to
                 */
                targetComponents: { type: "string[]", defaultValue: [] },
                
                /**
                 * Extension priority (higher = earlier execution)
                 */
                priority: { type: "int", defaultValue: 0 },
                
                /**
                 * Whether the extension is enabled
                 */
                enabled: { type: "boolean", defaultValue: true }
            }
        },

        constructor: function(mSettings) {
            BaseObject.call(this);
            
            // Apply settings
            if (mSettings) {
                this.applySettings(mSettings);
            }
            
            this._hooks = {};
            this._wrappedComponents = new WeakMap();
            
            Log.debug("ComponentExtension created: " + this.getId(), "trialbalance.extensions.ComponentExtension");
        },

        /**
         * Initialize the extension
         * Override this method to perform initialization tasks
         * @returns {Promise|undefined} Optional promise for async initialization
         */
        init: function() {
            // To be overridden by subclasses
            Log.debug("Initializing extension: " + this.getId(), "trialbalance.extensions.ComponentExtension");
        },

        /**
         * Extend a component instance with this extension's functionality
         * @param {sap.ui.core.Control} oComponent - Component to extend
         * @returns {sap.ui.core.Control} Extended component
         */
        extend: function(oComponent) {
            if (!this.getEnabled()) {
                return oComponent;
            }

            if (!this._isTargetComponent(oComponent)) {
                return oComponent;
            }

            if (this._wrappedComponents.has(oComponent)) {
                Log.warning("Component already extended: " + oComponent.getId(), "trialbalance.extensions.ComponentExtension");
                return oComponent;
            }

            Log.debug("Extending component: " + oComponent.getId() + " with extension: " + this.getId(), 
                     "trialbalance.extensions.ComponentExtension");

            // Call extension hook
            this.onBeforeExtend(oComponent);

            // Wrap component methods
            this._wrapComponentMethods(oComponent);

            // Mark as wrapped
            this._wrappedComponents.set(oComponent, true);

            // Call extension hook
            this.onAfterExtend(oComponent);

            return oComponent;
        },

        /**
         * Hook called before extending a component
         * Override this to perform pre-extension tasks
         * @param {sap.ui.core.Control} oComponent - Component being extended
         */
        onBeforeExtend: function(oComponent) {
            // To be overridden by subclasses
        },

        /**
         * Hook called after extending a component
         * Override this to perform post-extension tasks
         * @param {sap.ui.core.Control} oComponent - Component that was extended
         */
        onAfterExtend: function(oComponent) {
            // To be overridden by subclasses
        },

        /**
         * Transform data received by the component
         * Override this to modify data before it's processed
         * @param {*} vData - Original data
         * @param {Object} mContext - Context information
         * @returns {*} Transformed data
         */
        onDataReceived: function(vData, mContext) {
            // To be overridden by subclasses
            return vData;
        },

        /**
         * Transform data before rendering
         * Override this to modify data before visualization
         * @param {*} vData - Original data
         * @param {Object} mContext - Context information
         * @returns {*} Transformed data
         */
        onBeforeRender: function(vData, mContext) {
            // To be overridden by subclasses
            return vData;
        },

        /**
         * Hook called after component rendering
         * Override this to perform post-render tasks
         * @param {sap.ui.core.Control} oComponent - Rendered component
         */
        onAfterRender: function(oComponent) {
            // To be overridden by subclasses
        },

        /**
         * Handle user action on the component
         * Override this to intercept or enhance user interactions
         * @param {string} sAction - Action name
         * @param {Object} mParams - Action parameters
         * @returns {boolean} False to prevent default action
         */
        onUserAction: function(sAction, mParams) {
            // To be overridden by subclasses
            return true;
        },

        /**
         * Handle errors in the component
         * Override this to provide custom error handling
         * @param {Error} oError - Error object
         * @param {Object} mContext - Error context
         */
        onError: function(oError, mContext) {
            // To be overridden by subclasses
            Log.error("Error in component: " + (mContext.componentId || "unknown"), 
                     oError.message, "trialbalance.extensions.ComponentExtension");
        },

        /**
         * Get extension configuration as object for ExtensionManager
         * @returns {Object} Extension configuration
         */
        getExtensionConfig: function() {
            return {
                id: this.getId(),
                name: this.getName(),
                version: this.getVersion(),
                type: "component",
                priority: this.getPriority(),
                targetComponents: this.getTargetComponents(),
                
                lifecycle: {
                    init: this.init.bind(this),
                    destroy: this.destroy.bind(this)
                },
                
                hooks: {
                    "component.beforeExtend": this.onBeforeExtend.bind(this),
                    "component.afterExtend": this.onAfterExtend.bind(this),
                    "data.received": this.onDataReceived.bind(this),
                    "component.beforeRender": this.onBeforeRender.bind(this),
                    "component.afterRender": this.onAfterRender.bind(this),
                    "user.action": this.onUserAction.bind(this),
                    "component.error": this.onError.bind(this)
                }
            };
        },

        /**
         * Check if a component is a target for this extension
         * @private
         */
        _isTargetComponent: function(oComponent) {
            const aTargets = this.getTargetComponents();
            
            if (!aTargets || aTargets.length === 0) {
                return true; // No specific targets = applies to all
            }

            const sComponentType = oComponent.getMetadata().getName();
            
            return aTargets.some(function(sTarget) {
                return sComponentType === sTarget || sComponentType.indexOf(sTarget) !== -1;
            });
        },

        /**
         * Wrap component methods to inject extension behavior
         * @private
         */
        _wrapComponentMethods: function(oComponent) {
            const that = this;
            
            // Wrap onBeforeRendering
            if (oComponent.onBeforeRendering) {
                const fnOriginalBefore = oComponent.onBeforeRendering;
                oComponent.onBeforeRendering = function() {
                    try {
                        // Call extension hook
                        that.onBeforeRender(null, { component: oComponent });
                    } catch (e) {
                        Log.error("Error in onBeforeRender hook", e.message, "trialbalance.extensions.ComponentExtension");
                    }
                    // Call original
                    return fnOriginalBefore.apply(this, arguments);
                };
            }

            // Wrap onAfterRendering
            if (oComponent.onAfterRendering) {
                const fnOriginalAfter = oComponent.onAfterRendering;
                oComponent.onAfterRendering = function() {
                    const result = fnOriginalAfter.apply(this, arguments);
                    try {
                        // Call extension hook
                        that.onAfterRender(oComponent);
                    } catch (e) {
                        Log.error("Error in onAfterRender hook", e.message, "trialbalance.extensions.ComponentExtension");
                    }
                    return result;
                };
            }

            // Wrap setProperty methods to intercept data changes
            const fnOriginalSetProperty = oComponent.setProperty;
            if (fnOriginalSetProperty) {
                oComponent.setProperty = function(sPropertyName, vValue) {
                    try {
                        // Transform data through extension
                        vValue = that.onDataReceived(vValue, { 
                            property: sPropertyName,
                            component: oComponent 
                        });
                    } catch (e) {
                        Log.error("Error in onDataReceived hook", e.message, "trialbalance.extensions.ComponentExtension");
                    }
                    return fnOriginalSetProperty.apply(this, [sPropertyName, vValue]);
                };
            }
        },

        /**
         * Destroy the extension and clean up resources
         */
        destroy: function() {
            Log.debug("Destroying extension: " + this.getId(), "trialbalance.extensions.ComponentExtension");
            
            this._wrappedComponents = null;
            this._hooks = null;
            
            BaseObject.prototype.destroy.call(this);
        }
    });
});