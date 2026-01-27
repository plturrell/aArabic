sap.ui.define([
    "sap/ui/core/Control"
], function(Control) {
    "use strict";

    return Control.extend("trialbalance.control.NetworkGraphControl", {
        metadata: {
            properties: {
                width: { type: "sap.ui.core.CSSSize", defaultValue: "100%" },
                height: { type: "sap.ui.core.CSSSize", defaultValue: "600px" },
                nodes: { type: "object", defaultValue: null },
                edges: { type: "object", defaultValue: null }
            }
        },

        renderer: function(oRm, oControl) {
            oRm.openStart("div", oControl);
            oRm.class("networkgraph-container");
            oRm.style("width", oControl.getWidth());
            oRm.style("height", oControl.getHeight());
            oRm.style("position", "relative");
            oRm.style("border", "1px solid #ccc");
            oRm.style("border-radius", "4px");
            oRm.openEnd();
            
            // Placeholder for NetworkGraph
            oRm.openStart("div");
            oRm.class("networkgraph-canvas");
            oRm.style("width", "100%");
            oRm.style("height", "100%");
            oRm.openEnd();
            oRm.close("div");
            
            oRm.close("div");
        },

        onAfterRendering: function() {
            // Load NetworkGraph script if not already loaded
            if (!window.NetworkGraph) {
                this._loadNetworkGraph();
            } else {
                this._initializeGraph();
            }
        },

        _loadNetworkGraph: function() {
            var sScriptPath = sap.ui.require.toUrl("trialbalance/components/dist/NetworkGraph/NetworkGraph.min.js");
            var sCssPath = sap.ui.require.toUrl("trialbalance/components/NetworkGraph/styles.css");
            
            // Load CSS
            var link = document.createElement("link");
            link.rel = "stylesheet";
            link.href = sCssPath;
            document.head.appendChild(link);
            
            // Load JS as ES6 module
            var script = document.createElement("script");
            script.type = "module";
            script.textContent = `
                import { NetworkGraph } from '${sScriptPath}';
                window.NetworkGraph = NetworkGraph;
                window.dispatchEvent(new CustomEvent('networkgraph-loaded'));
            `;
            document.head.appendChild(script);
            
            // Listen for module loaded event
            window.addEventListener('networkgraph-loaded', function() {
                this._initializeGraph();
            }.bind(this), { once: true });
        },

        _initializeGraph: function() {
            var oContainer = this.getDomRef().querySelector(".networkgraph-canvas");
            var oNodes = this.getNodes();
            var oEdges = this.getEdges();
            
            if (oContainer && window.NetworkGraph) {
                try {
                    this._graph = new window.NetworkGraph(oContainer, {
                        nodes: oNodes || this._getSampleNodes(),
                        edges: oEdges || this._getSampleEdges()
                    });
                } catch (e) {
                    console.error("Error initializing NetworkGraph:", e);
                    oContainer.innerHTML = '<div style="padding: 20px; text-align: center;">NetworkGraph component loaded. Add data to visualize.</div>';
                }
            }
        },

        _getSampleNodes: function() {
            return [
                { id: "1", label: "Assets", group: "BS", x: 100, y: 100 },
                { id: "2", label: "Liabilities", group: "BS", x: 300, y: 100 },
                { id: "3", label: "Equity", group: "BS", x: 200, y: 200 },
                { id: "4", label: "Revenue", group: "PL", x: 100, y: 300 },
                { id: "5", label: "Expenses", group: "PL", x: 300, y: 300 }
            ];
        },

        _getSampleEdges: function() {
            return [
                { source: "1", target: "2", value: 1 },
                { source: "2", target: "3", value: 1 },
                { source: "4", target: "5", value: 1 }
            ];
        },

        onBeforeRendering: function() {
            if (this._graph && this._graph.destroy) {
                this._graph.destroy();
                this._graph = null;
            }
        },

        exit: function() {
            if (this._graph && this._graph.destroy) {
                this._graph.destroy();
                this._graph = null;
            }
        }
    });
});