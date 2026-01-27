sap.ui.define([
    "trialbalance/extensions/ComponentExtension",
    "sap/base/Log"
], function(ComponentExtension, Log) {
    "use strict";

    /**
     * Charts Extension
     * Wraps the TypeScript Charts suite (RadialChart, GaugeChart, LineChart, BarChart, SankeyDiagram).
     * Provides unified chart creation and management for trial balance visualizations.
     * 
     * Features:
     * - RadialChart: Circular progress for percentages
     * - GaugeChart: Semi-circular gauge with zones
     * - LineChart: Multi-series line chart for trends
     * - BarChart: Grouped/stacked bar chart for comparisons
     * - SankeyDiagram: Flow visualization for data pipelines
     * 
     * @class
     * @extends trialbalance.extensions.ComponentExtension
     */
    return ComponentExtension.extend("trialbalance.extensions.components.ChartsExtension", {
        
        constructor: function(mSettings) {
            ComponentExtension.call(this, mSettings);
            
            this.setId("charts-core");
            this.setName("Charts Core");
            this.setVersion("1.0.0");
            this.setTargetComponents([
                "trialbalance.control.ChartControl",
                "trialbalance.control.RadialChartControl",
                "trialbalance.control.GaugeChartControl",
                "trialbalance.control.LineChartControl",
                "trialbalance.control.BarChartControl",
                "trialbalance.control.SankeyControl"
            ]);
            this.setPriority(100);
            
            this._charts = new Map(); // Store chart instances by ID
            this._chartFactory = null;
            this._config = null;
            this._dataCache = new Map();
            this._eventHandlers = new Map();
            this._extensionManager = null;
        },

        init: function() {
            Log.info("Initializing Charts Extension", "trialbalance.extensions.components.ChartsExtension");
            
            return Promise.all([
                this._loadChartsScript(),
                this._loadChartStyles(),
                this._loadConfiguration()
            ]).then(function() {
                Log.info("Charts Extension initialized", "trialbalance.extensions.components.ChartsExtension");
            }.bind(this));
        },

        setExtensionManager: function(oManager) {
            this._extensionManager = oManager;
        },

        onBeforeExtend: function(oComponent) {
            this._ui5Control = oComponent;
            oComponent.addStyleClass("charts-extended");
        },

        onAfterExtend: function(oComponent) {
            setTimeout(function() {
                this._initializeChart(oComponent);
            }.bind(this), 100);
        },

        onDataReceived: function(vData, mContext) {
            if (!vData) return vData;
            
            if (this._extensionManager) {
                vData = this._extensionManager.executeHook("charts.data.transform", {
                    data: vData,
                    metadata: mContext
                });
            }
            
            if (mContext && mContext.chartId) {
                this._dataCache.set(mContext.chartId, vData);
            }
            
            return vData;
        },

        onUserAction: function(sAction, mParams) {
            Log.debug("Chart action: " + sAction, "trialbalance.extensions.components.ChartsExtension");
            
            if (this._extensionManager) {
                const bContinue = this._extensionManager.executeHook("charts.action.before", {
                    action: sAction,
                    params: mParams
                });
                if (bContinue === false) return false;
            }
            
            if (this._extensionManager) {
                this._extensionManager.executeHook("charts.action.after", { action: sAction, params: mParams });
            }
            
            return true;
        },

        // ========== Public API ==========

        /**
         * Create a chart of specified type
         * @param {string} sType - Chart type (radial, gauge, line, bar, sankey)
         * @param {HTMLElement|string} vContainer - Container element or selector
         * @param {Object} oConfig - Chart configuration
         * @returns {Object} Chart instance
         */
        createChart: function(sType, vContainer, oConfig) {
            if (!this._chartFactory) {
                Log.error("Charts not loaded", "trialbalance.extensions.components.ChartsExtension");
                return null;
            }
            
            if (this._extensionManager) {
                oConfig = this._extensionManager.executeHook("charts.create.before", {
                    type: sType,
                    config: oConfig,
                    metadata: {}
                }) || oConfig;
            }
            
            try {
                const chart = this._chartFactory(sType, vContainer, oConfig);
                const chartId = oConfig.id || "chart-" + Date.now();
                this._charts.set(chartId, { type: sType, instance: chart, config: oConfig });
                
                if (this._extensionManager) {
                    this._extensionManager.executeHook("charts.created", { 
                        chartId: chartId, type: sType, chart: chart 
                    });
                }
                
                return chart;
            } catch (e) {
                Log.error("Failed to create chart: " + sType, e.message, "trialbalance.extensions.components.ChartsExtension");
                return null;
            }
        },

        /**
         * Create a Radial Chart
         * @param {HTMLElement|string} vContainer - Container
         * @param {Object} oConfig - Configuration (value, max, label, color)
         * @returns {Object} RadialChart instance
         */
        createRadialChart: function(vContainer, oConfig) {
            return this.createChart("radial", vContainer, oConfig);
        },

        /**
         * Create a Gauge Chart
         * @param {HTMLElement|string} vContainer - Container
         * @param {Object} oConfig - Configuration (value, min, max, zones)
         * @returns {Object} GaugeChart instance
         */
        createGaugeChart: function(vContainer, oConfig) {
            return this.createChart("gauge", vContainer, oConfig);
        },

        /**
         * Create a Line Chart
         * @param {HTMLElement|string} vContainer - Container
         * @param {Object} oConfig - Configuration (series, xAxis, yAxis)
         * @returns {Object} LineChart instance
         */
        createLineChart: function(vContainer, oConfig) {
            return this.createChart("line", vContainer, oConfig);
        },

        /**
         * Create a Bar Chart
         * @param {HTMLElement|string} vContainer - Container
         * @param {Object} oConfig - Configuration (data, categories, stacked)
         * @returns {Object} BarChart instance
         */
        createBarChart: function(vContainer, oConfig) {
            return this.createChart("bar", vContainer, oConfig);
        },

        /**
         * Create a Sankey Diagram
         * @param {HTMLElement|string} vContainer - Container
         * @param {Object} oConfig - Configuration (nodes, links)
         * @returns {Object} SankeyDiagram instance
         */
        createSankeyDiagram: function(vContainer, oConfig) {
            return this.createChart("sankey", vContainer, oConfig);
        },

        /**
         * Get a chart by ID
         * @param {string} sChartId - Chart ID
         * @returns {Object} Chart entry {type, instance, config}
         */
        getChart: function(sChartId) {
            return this._charts.get(sChartId);
        },

        /**
         * Update chart data
         * @param {string} sChartId - Chart ID
         * @param {*} vData - New data
         */
        updateChartData: function(sChartId, vData) {
            const chartEntry = this._charts.get(sChartId);
            if (!chartEntry || !chartEntry.instance) return;
            
            if (this._extensionManager) {
                vData = this._extensionManager.executeHook("charts.data.transform", {
                    data: vData,
                    metadata: { chartId: sChartId, type: chartEntry.type }
                }) || vData;
            }
            
            // Different update methods based on chart type
            const chart = chartEntry.instance;
            if (chart.update) {
                chart.update(vData);
            } else if (chart.setData) {
                chart.setData(vData);
            } else if (chart.render) {
                chart.render(vData);
            }
        },

        /**
         * Destroy a chart
         * @param {string} sChartId - Chart ID
         */
        destroyChart: function(sChartId) {
            const chartEntry = this._charts.get(sChartId);
            if (!chartEntry) return;
            
            if (chartEntry.instance && chartEntry.instance.destroy) {
                chartEntry.instance.destroy();
            }
            
            this._charts.delete(sChartId);
            this._dataCache.delete(sChartId);
        },

        /**
         * Get all chart instances
         * @returns {Map} Map of chart entries
         */
        getAllCharts: function() {
            return this._charts;
        },

        /**
         * Get available chart types
         * @returns {Array} Array of chart type names
         */
        getChartTypes: function() {
            return ["radial", "gauge", "line", "bar", "sankey"];
        },

        on: function(sEvent, fnHandler) {
            if (!this._eventHandlers.has(sEvent)) {
                this._eventHandlers.set(sEvent, []);
            }
            this._eventHandlers.get(sEvent).push(fnHandler);
        },

        off: function(sEvent, fnHandler) {
            if (!fnHandler) {
                this._eventHandlers.delete(sEvent);
            } else {
                const handlers = this._eventHandlers.get(sEvent);
                if (handlers) {
                    const idx = handlers.indexOf(fnHandler);
                    if (idx !== -1) handlers.splice(idx, 1);
                }
            }
        },

        // ========== Private Methods ==========

        _loadChartsScript: function() {
            return new Promise(function(resolve, reject) {
                if (window.createChart) {
                    this._chartFactory = window.createChart;
                    resolve();
                    return;
                }
                
                const sScriptPath = sap.ui.require.toUrl("trialbalance/components/dist/Charts/Charts.min.js");
                
                const script = document.createElement("script");
                script.type = "module";
                script.textContent = `
                    import { createChart, RadialChart, GaugeChart, LineChart, BarChart, SankeyDiagram } from '${sScriptPath}';
                    window.createChart = createChart;
                    window.RadialChart = RadialChart;
                    window.GaugeChart = GaugeChart;
                    window.LineChart = LineChart;
                    window.BarChart = BarChart;
                    window.SankeyDiagram = SankeyDiagram;
                    window.dispatchEvent(new CustomEvent('charts-loaded'));
                `;
                document.head.appendChild(script);
                
                window.addEventListener('charts-loaded', function() {
                    this._chartFactory = window.createChart;
                    Log.info("Charts script loaded", "trialbalance.extensions.components.ChartsExtension");
                    resolve();
                }.bind(this), { once: true });
                
                setTimeout(function() {
                    if (!window.createChart) reject(new Error("Charts script load timeout"));
                }, 10000);
            }.bind(this));
        },

        _loadChartStyles: function() {
            return new Promise(function(resolve) {
                const sCssPath = sap.ui.require.toUrl("trialbalance/components/Charts/charts.css");
                const link = document.createElement("link");
                link.rel = "stylesheet";
                link.href = sCssPath;
                link.onload = resolve;
                link.onerror = resolve; // Continue even if CSS fails
                document.head.appendChild(link);
            });
        },

        _loadConfiguration: function() {
            return fetch('/api/v1/extensions/charts-core/config')
                .then(function(response) { return response.json(); })
                .then(function(config) { this._config = config; }.bind(this))
                .catch(function() {
                    this._config = {
                        defaultColors: ['#0a84ff', '#30d158', '#ff9f0a', '#ff453a', '#bf5af2'],
                        animation: true,
                        responsive: true
                    };
                }.bind(this));
        },

        _initializeChart: function(oComponent) {
            const domRef = oComponent.getDomRef();
            if (!domRef) return;
            
            // Check for chart type from component metadata
            const chartType = oComponent.data("chartType") || "line";
            const chartConfig = oComponent.data("chartConfig") || {};
            const chartId = oComponent.getId();
            
            // Create chart instance
            const canvas = domRef.querySelector(".chart-canvas") || domRef;
            const chart = this.createChart(chartType, canvas, { id: chartId, ...chartConfig });
            
            if (chart) {
                // Sync cached data
                const cachedData = this._dataCache.get(chartId);
                if (cachedData) {
                    this.updateChartData(chartId, cachedData);
                }
            }
        },

        destroy: function() {
            Log.info("Destroying Charts Extension", "trialbalance.extensions.components.ChartsExtension");
            
            // Destroy all chart instances
            this._charts.forEach(function(chartEntry, chartId) {
                this.destroyChart(chartId);
            }.bind(this));
            
            this._charts.clear();
            this._dataCache.clear();
            this._eventHandlers.clear();
            this._chartFactory = null;
            this._config = null;
            this._extensionManager = null;
            
            ComponentExtension.prototype.destroy.call(this);
        }
    });
});