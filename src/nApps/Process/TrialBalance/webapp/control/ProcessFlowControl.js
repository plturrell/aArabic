sap.ui.define([
    "sap/ui/core/Control"
], function(Control) {
    "use strict";

    return Control.extend("trialbalance.control.ProcessFlowControl", {
        metadata: {
            properties: {
                width: { type: "sap.ui.core.CSSSize", defaultValue: "100%" },
                height: { type: "sap.ui.core.CSSSize", defaultValue: "400px" },
                steps: { type: "object", defaultValue: null }
            }
        },

        renderer: function(oRm, oControl) {
            oRm.openStart("div", oControl);
            oRm.class("processflow-container");
            oRm.style("width", oControl.getWidth());
            oRm.style("height", oControl.getHeight());
            oRm.style("position", "relative");
            oRm.style("border", "1px solid #ccc");
            oRm.style("border-radius", "4px");
            oRm.style("background", "#fafafa");
            oRm.openEnd();
            
            // Placeholder for ProcessFlow
            oRm.openStart("div");
            oRm.class("processflow-canvas");
            oRm.style("width", "100%");
            oRm.style("height", "100%");
            oRm.openEnd();
            oRm.close("div");
            
            oRm.close("div");
        },

        onAfterRendering: function() {
            // Load ProcessFlow script if not already loaded
            if (!window.ProcessFlow) {
                this._loadProcessFlow();
            } else {
                this._initializeFlow();
            }
        },

        _loadProcessFlow: function() {
            var sScriptPath = sap.ui.require.toUrl("trialbalance/components/dist/ProcessFlow/ProcessFlow.min.js");
            var sCssPath = sap.ui.require.toUrl("trialbalance/components/ProcessFlow/processflow.css");
            
            // Load CSS
            var link = document.createElement("link");
            link.rel = "stylesheet";
            link.href = sCssPath;
            document.head.appendChild(link);
            
            // Load JS as ES6 module
            var script = document.createElement("script");
            script.type = "module";
            script.textContent = `
                import { ProcessFlow } from '${sScriptPath}';
                window.ProcessFlow = ProcessFlow;
                window.dispatchEvent(new CustomEvent('processflow-loaded'));
            `;
            document.head.appendChild(script);
            
            // Listen for module loaded event
            window.addEventListener('processflow-loaded', function() {
                this._initializeFlow();
            }.bind(this), { once: true });
        },

        _initializeFlow: function() {
            var oContainer = this.getDomRef().querySelector(".processflow-canvas");
            var oSteps = this.getSteps();
            
            if (oContainer && window.ProcessFlow) {
                try {
                    this._flow = new window.ProcessFlow(oContainer, {
                        nodes: oSteps || this._getSampleSteps()
                    });
                } catch (e) {
                    console.error("Error initializing ProcessFlow:", e);
                    oContainer.innerHTML = '<div style="padding: 20px; text-align: center;">ProcessFlow component loaded. Add steps to visualize.</div>';
                }
            }
        },

        _getSampleSteps: function() {
            return [
                { id: "1", title: "Data Collection", state: "completed", lane: "Finance" },
                { id: "2", title: "Data Validation", state: "completed", lane: "Finance" },
                { id: "3", title: "Calculation", state: "current", lane: "Accounting" },
                { id: "4", title: "Review", state: "pending", lane: "Accounting" },
                { id: "5", title: "Approval", state: "pending", lane: "Management" }
            ];
        },

        onBeforeRendering: function() {
            if (this._flow && this._flow.destroy) {
                this._flow.destroy();
                this._flow = null;
            }
        },

        exit: function() {
            if (this._flow && this._flow.destroy) {
                this._flow.destroy();
                this._flow = null;
            }
        }
    });
});