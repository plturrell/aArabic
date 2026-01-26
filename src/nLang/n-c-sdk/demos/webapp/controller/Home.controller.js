e sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/routing/History",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "sap/ui/core/Fragment"
], function (Controller, History, MessageToast, JSONModel, Fragment) {
    "use strict";

    return Controller.extend("galaxy.sim.controller.Home", {

        onInit: function () {
            console.log("🏠 Home controller initialized - Real-time HPC Dashboard");

            // Initialize HPC metrics model with live streaming data structure
            const hpcModel = new JSONModel({
                connected: false,
                // SIMD Performance
                simd: {
                    speedup: "0.0",
                    efficiency: "0",
                    scalarMs: "0.00",
                    simdMs: "0.00"
                },
                // N-Body Simulation
                simulation: {
                    fps: "0.0",
                    frameMs: "0.00",
                    treeBuildMs: "0.00",
                    forceCalcMs: "0.00",
                    bodies: "100,000"
                },
                // Memory Performance
                memory: {
                    cacheHitRate: "0.0",
                    bandwidthGbps: "0.0",
                    heapMb: "0"
                },
                // WASM
                wasm: {
                    sizeKb: "47.2",
                    loadMs: "0"
                },
                // Uptime
                uptime: {
                    seconds: 0,
                    display: "0s"
                },
                // History buffers for charts (60 samples)
                history: {
                    fps: [],
                    simd: [],
                    memory: [],
                    cache: []
                }
            });

            this.getView().setModel(hpcModel, "hpc");

            // Initialize HPC Benchmarks model with sample data
            const hpcBenchmarkModel = new JSONModel({
                timestamp: new Date().toISOString(),
                system: {
                    architecture: "x86_64",
                    vector_width: 8,
                    cache_line_bytes: 64
                },
                stream: {
                    copy_bw_gbs: 42.5,
                    scale_bw_gbs: 41.8,
                    add_bw_gbs: 38.2,
                    triad_bw_gbs: 37.9
                },
                linpack: {
                    achieved_gflops: 245.3,
                    theoretical_peak_gflops: 384.0,
                    efficiency_percent: 63.9
                },
                latency: {
                    l1_ns: 1.2,
                    l2_ns: 4.8,
                    l3_ns: 18.5,
                    dram_ns: 95.2
                },
                strong_scaling: [
                    { threads: 1, time_ms: 1000, speedup: 1.0, efficiency_percent: 100.0 },
                    { threads: 2, time_ms: 520, speedup: 1.92, efficiency_percent: 96.0 },
                    { threads: 4, time_ms: 270, speedup: 3.70, efficiency_percent: 92.5 },
                    { threads: 8, time_ms: 145, speedup: 6.90, efficiency_percent: 86.3 },
                    { threads: 16, time_ms: 85, speedup: 11.76, efficiency_percent: 73.5 }
                ],
                weak_scaling: [
                    { threads: 1, time_ms: 500, efficiency_percent: 100.0 },
                    { threads: 2, time_ms: 510, efficiency_percent: 98.0 },
                    { threads: 4, time_ms: 525, efficiency_percent: 95.2 },
                    { threads: 8, time_ms: 550, efficiency_percent: 90.9 },
                    { threads: 16, time_ms: 590, efficiency_percent: 84.7 }
                ],
                simd: {
                    scalar_gflops: 12.5,
                    vector_gflops: 65.2,
                    speedup: 5.22,
                    utilization_percent: 65.3
                },
                // History for streaming charts
                history: {
                    stream_copy: [],
                    stream_triad: [],
                    linpack: [],
                    latency_l1: [],
                    latency_dram: [],
                    simd_speedup: [],
                    strong_speedup: [],
                    weak_efficiency: []
                }
            });
            this.getView().setModel(hpcBenchmarkModel, "hpcModel");

            // Start simulated live benchmark streaming
            this._startBenchmarkStreaming();

            // Initialize drill-down model
            const drillDownModel = new JSONModel({
                title: "",
                currentValue: "",
                trend: "",
                trendClass: "",
                details: "",
                chartType: "line",
                min: "0",
                max: "0",
                avg: "0",
                stdDev: "0",
                data: []
            });
            this.getView().setModel(drillDownModel, "drillDown");

            // Connect to HPC WebSocket stream
            this._connectHPCStream();
        },

        _connectHPCStream: function () {
            // Check if socket.io client is available
            if (typeof io === 'undefined') {
                console.log('⚠️ WebSocket client not loaded - waiting...');
                setTimeout(() => this._connectHPCStream(), 500);
                return;
            }

            try {
                // Connect to Zig WebSocket server
                this._socket = io();

                this._socket.on('connect', () => {
                    console.log('✅ Connected to HPC real-time stream');
                    this.getView().getModel("hpc").setProperty("/connected", true);
                });

                this._socket.on('disconnect', () => {
                    console.log('❌ Disconnected from HPC stream');
                    this.getView().getModel("hpc").setProperty("/connected", false);
                });

                // Handle real-time HPC metrics from Zig server
                this._socket.on('hpc:metrics', (data) => {
                    this._updateHPCModel(data);
                });

            } catch (error) {
                console.warn('HPC WebSocket connection failed:', error);
            }
        },

        _updateHPCModel: function (data) {
            const model = this.getView().getModel("hpc");
            if (!data) return;

            // Update SIMD metrics
            if (data.simd) {
                if (data.simd.speedup !== undefined) model.setProperty("/simd/speedup", data.simd.speedup.toFixed(1));
                if (data.simd.efficiency !== undefined) model.setProperty("/simd/efficiency", data.simd.efficiency.toFixed(0));
                if (data.simd.scalarMs !== undefined) model.setProperty("/simd/scalarMs", data.simd.scalarMs.toFixed(3));
                if (data.simd.simdMs !== undefined) model.setProperty("/simd/simdMs", data.simd.simdMs.toFixed(3));
            }

            // Update Simulation metrics
            if (data.simulation) {
                if (data.simulation.fps !== undefined) model.setProperty("/simulation/fps", data.simulation.fps.toFixed(1));
                if (data.simulation.frameMs !== undefined) model.setProperty("/simulation/frameMs", data.simulation.frameMs.toFixed(2));
                if (data.simulation.treeBuildMs !== undefined) model.setProperty("/simulation/treeBuildMs", data.simulation.treeBuildMs.toFixed(2));
                if (data.simulation.forceCalcMs !== undefined) model.setProperty("/simulation/forceCalcMs", data.simulation.forceCalcMs.toFixed(2));
                if (data.simulation.bodies !== undefined) model.setProperty("/simulation/bodies", data.simulation.bodies.toLocaleString());
            }

            // Update Memory metrics
            if (data.memory) {
                if (data.memory.cacheHitRate !== undefined) model.setProperty("/memory/cacheHitRate", data.memory.cacheHitRate.toFixed(1));
                if (data.memory.bandwidthGbps !== undefined) model.setProperty("/memory/bandwidthGbps", data.memory.bandwidthGbps.toFixed(1));
                if (data.memory.heapMb !== undefined) model.setProperty("/memory/heapMb", data.memory.heapMb.toFixed(0));
            }

            // Update WASM metrics
            if (data.wasm) {
                if (data.wasm.sizeKb !== undefined) model.setProperty("/wasm/sizeKb", data.wasm.sizeKb.toFixed(1));
                if (data.wasm.loadMs !== undefined) model.setProperty("/wasm/loadMs", data.wasm.loadMs.toFixed(2));
            }

            // Update uptime
            if (data.uptime !== undefined) {
                model.setProperty("/uptime/seconds", data.uptime);
                model.setProperty("/uptime/display", this._formatUptime(data.uptime));
            }

            // Update history for charts
            this._updateHistory(data);

            // Update drill-down chart if open
            if (this._drillDownDialog && this._drillDownDialog.isOpen()) {
                this._updateDrillDownChart();
            }
        },

        _updateHistory: function(data) {
            const model = this.getView().getModel("hpc");
            const maxHistory = 60;

            // Get current history
            let fpsHistory = model.getProperty("/history/fps") || [];
            let cacheHistory = model.getProperty("/history/cache") || [];

            // Add new values
            if (data.simulation) {
                fpsHistory.push(data.simulation.fps);
                if (fpsHistory.length > maxHistory) fpsHistory.shift();
                model.setProperty("/history/fps", fpsHistory);
            }

            if (data.memory) {
                cacheHistory.push(data.memory.cacheHitRate);
                if (cacheHistory.length > maxHistory) cacheHistory.shift();
                model.setProperty("/history/cache", cacheHistory);
            }
        },

        _formatUptime: function (seconds) {
            if (seconds < 60) return Math.floor(seconds) + "s";
            if (seconds < 3600) return Math.floor(seconds / 60) + "m " + Math.floor(seconds % 60) + "s";
            return Math.floor(seconds / 3600) + "h " + Math.floor((seconds % 3600) / 60) + "m";
        },

        /**
         * Breadcrumb Handlers
         */
        onBreadcrumbHome: function () {
            this.getOwnerComponent().getRouter().navTo("home");
        },

        /**
         * Hero Banner Handlers
         */
        onGetStarted: function () {
            // Scroll to quick start section or navigate to getting started guide
            MessageToast.show("Welcome to nLang SDK! Explore our demos and documentation.");
            // Could navigate to a getting started page in the future
            // this.getOwnerComponent().getRouter().navTo("gettingStarted");
        },

        onSeeAllFeatures: function () {
            // Navigate to a full features page or scroll to SDK section
            MessageToast.show("Explore all nLang SDK features");
            // Could navigate to a features page in the future
            // this.getOwnerComponent().getRouter().navTo("features");
        },

        /**
         * Navigation Handlers - Performance & Simulation
         */
        onNavigateToGalaxy: function () {
            this.getOwnerComponent().getRouter().navTo("galaxy");
        },

        onNavigateToAnalytics: function () {
            this.getOwnerComponent().getRouter().navTo("analytics");
        },

        onNavigateToBenchmarks: function () {
            this.getOwnerComponent().getRouter().navTo("benchmarks");
        },

        // ============================================
        // LIVE BENCHMARK STREAMING
        // ============================================

        _startBenchmarkStreaming: function () {
            console.log("🔄 Starting live benchmark streaming at 40 FPS...");
            
            // Update at 40 FPS (25ms interval) for smooth real-time visualization
            this._benchmarkInterval = setInterval(() => {
                this._updateBenchmarkMetrics();
                this._renderBenchmarkCharts();
            }, 25);

            // Initial render
            setTimeout(() => {
                this._renderBenchmarkCharts();
            }, 100);
        },

        _updateBenchmarkMetrics: function () {
            const model = this.getView().getModel("hpcModel");
            const maxHistory = 60;

            // Add small random variations to metrics (±5%)
            const vary = (base) => base * (1 + (Math.random() - 0.5) * 0.1);

            // Update STREAM
            model.setProperty("/stream/copy_bw_gbs", vary(42.5).toFixed(1));
            model.setProperty("/stream/triad_bw_gbs", vary(37.9).toFixed(1));

            // Update LINPACK
            const linpack = vary(245.3);
            model.setProperty("/linpack/achieved_gflops", linpack.toFixed(1));
            model.setProperty("/linpack/efficiency_percent", (linpack / 384 * 100).toFixed(1));

            // Update Latency
            model.setProperty("/latency/l1_ns", vary(1.2).toFixed(2));
            model.setProperty("/latency/dram_ns", vary(95.2).toFixed(1));

            // Update SIMD
            model.setProperty("/simd/speedup", vary(5.22).toFixed(2));
            model.setProperty("/simd/utilization_percent", vary(65.3).toFixed(1));

            // Update history arrays
            const history = model.getProperty("/history");
            
            history.stream_copy.push(parseFloat(model.getProperty("/stream/copy_bw_gbs")));
            history.stream_triad.push(parseFloat(model.getProperty("/stream/triad_bw_gbs")));
            history.linpack.push(linpack);
            history.latency_l1.push(parseFloat(model.getProperty("/latency/l1_ns")));
            history.latency_dram.push(parseFloat(model.getProperty("/latency/dram_ns")));
            history.simd_speedup.push(parseFloat(model.getProperty("/simd/speedup")));

            // Keep max 60 samples
            Object.keys(history).forEach(key => {
                if (history[key].length > maxHistory) {
                    history[key].shift();
                }
            });

            model.setProperty("/history", history);
        },

        _renderBenchmarkCharts: function () {
            const model = this.getView().getModel("hpcModel");
            
            // Render all micro charts with live data
            this._renderLineChart("streamChart", model.getProperty("/history/stream_copy") || []);
            this._renderLineChart("latencyChart", model.getProperty("/history/latency_dram") || []);
            this._renderLineChart("scalingChart", model.getProperty("/history/simd_speedup") || []);
            
            this._renderBulletChart("linpackBullet", 
                model.getProperty("/linpack/efficiency_percent"), 
                100);
            this._renderBulletChart("simdBullet", 
                model.getProperty("/simd/utilization_percent"), 
                100);
            this._renderBulletChart("weakBullet", 
                model.getProperty("/weak_scaling/4/efficiency_percent"), 
                100);
        },

        _renderLineChart: function (canvasId, data) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;

            const ctx = canvas.getContext("2d");
            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;
            canvas.width = width;
            canvas.height = height;

            ctx.clearRect(0, 0, width, height);

            if (!data || data.length < 2) return;

            const padding = 4;
            const chartWidth = width - padding * 2;
            const chartHeight = height - padding * 2;

            const min = Math.min(...data) * 0.95;
            const max = Math.max(...data) * 1.05;
            const range = max - min || 1;

            // Draw gradient fill
            const gradient = ctx.createLinearGradient(0, padding, 0, height - padding);
            gradient.addColorStop(0, "rgba(0, 122, 255, 0.2)");
            gradient.addColorStop(1, "rgba(0, 122, 255, 0.02)");

            ctx.beginPath();
            ctx.moveTo(padding, height - padding);
            data.forEach((val, i) => {
                const x = padding + (i / (data.length - 1)) * chartWidth;
                const y = height - padding - ((val - min) / range) * chartHeight;
                ctx.lineTo(x, y);
            });
            ctx.lineTo(width - padding, height - padding);
            ctx.closePath();
            ctx.fillStyle = gradient;
            ctx.fill();

            // Draw line
            ctx.beginPath();
            ctx.strokeStyle = "#007aff";
            ctx.lineWidth = 2;
            ctx.lineJoin = "round";
            data.forEach((val, i) => {
                const x = padding + (i / (data.length - 1)) * chartWidth;
                const y = height - padding - ((val - min) / range) * chartHeight;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        },

        _renderBulletChart: function (elementId, value, max) {
            const element = document.getElementById(elementId);
            if (!element) return;

            const percentage = (value / max) * 100;
            
            // Color based on percentage
            let color = "#34c759"; // green
            if (percentage < 70) color = "#ff9500"; // orange
            if (percentage < 50) color = "#ff3b30"; // red

            element.style.width = percentage + "%";
            element.style.background = `linear-gradient(90deg, ${color}, ${color}dd)`;
            element.style.height = "100%";
            element.style.borderRadius = "4px";
            element.style.transition = "width 0.3s ease";
        },

        /**
         * Navigation Handlers - Data Architecture & Integration
         */
        onNavigateToCSN: function () {
            MessageToast.show("CSN Parser Demo - Coming Soon!");
            // Future implementation for CSN parser demo
            // this.getOwnerComponent().getRouter().navTo("csn");
        },

        onNavigateToORD: function () {
            MessageToast.show("ORD Discovery Demo - Coming Soon!");
            // Future implementation for ORD service discovery
            // this.getOwnerComponent().getRouter().navTo("ord");
        },

        onNavigateToLineage: function () {
            MessageToast.show("OpenLineage Tracking - Coming Soon!");
            // Future implementation for lineage tracking
            // this.getOwnerComponent().getRouter().navTo("lineage");
        },

        /**
         * Navigation Handlers - Optimization & Algorithms
         */
        onNavigateToHungarian: function () {
            MessageToast.show("Hungarian Algorithm Demo - Coming Soon!");
            // Future implementation for Hungarian algorithm visualization
            // this.getOwnerComponent().getRouter().navTo("hungarian");
        },

        onNavigateToSIMD: function () {
            MessageToast.show("SIMD Operations Demo - Coming Soon!");
            // Future implementation for SIMD visualization
            // this.getOwnerComponent().getRouter().navTo("simd");
        },

        onNavigateToMemory: function () {
            MessageToast.show("Memory Management Demo - Coming Soon!");
            // Future implementation for memory management tools
            // this.getOwnerComponent().getRouter().navTo("memory");
        },

        /**
         * Navigation Handlers - Development & Testing
         */
        onNavigateToFFI: function () {
            MessageToast.show("FFI Bindings Reference - Coming Soon!");
            // Future implementation for FFI documentation
            // this.getOwnerComponent().getRouter().navTo("ffi");
        },

        onNavigateToTests: function () {
            MessageToast.show("Unit Testing Suite - Coming Soon!");
            // Future implementation for test runner interface
            // this.getOwnerComponent().getRouter().navTo("tests");
        },

        onNavigateToDocs: function () {
            this.getOwnerComponent().getRouter().navTo("docs");
        },

        /**
         * Settings Handler
         */
        onSettingsPress: function () {
            this.getOwnerComponent().getRouter().navTo("settings");
        },

        // ============================================
        // DRILL-DOWN CARD HANDLERS - Live Chart Analytics
        // ============================================

        onSIMDCardPress: function () {
            this._openDrillDownDialog("simd", "SIMD Vectorization Performance", "speedup");
        },

        onSimulationCardPress: function () {
            this._openDrillDownDialog("simulation", "N-Body Simulation Performance", "fps");
        },

        onMemoryCardPress: function () {
            this._openDrillDownDialog("memory", "Memory Performance", "cacheHitRate");
        },

        onSystemCardPress: function () {
            this._openDrillDownDialog("system", "System Uptime & WASM", "uptime");
        },

        _openDrillDownDialog: function (metricType, title, valueKey) {
            const view = this.getView();
            const drillDownModel = view.getModel("drillDown");
            const hpcModel = view.getModel("hpc");

            // Set drill-down context
            this._currentMetricType = metricType;
            this._currentValueKey = valueKey;

            // Get current values based on metric type
            let currentValue, trend, details, historyData;

            switch (metricType) {
                case "simd":
                    currentValue = hpcModel.getProperty("/simd/speedup") + "x";
                    trend = "↑ " + hpcModel.getProperty("/simd/efficiency") + "% efficiency";
                    details = "Scalar: " + hpcModel.getProperty("/simd/scalarMs") + "ms → SIMD: " + hpcModel.getProperty("/simd/simdMs") + "ms";
                    historyData = hpcModel.getProperty("/history/fps") || [];
                    break;
                case "simulation":
                    currentValue = hpcModel.getProperty("/simulation/fps") + " FPS";
                    trend = parseFloat(hpcModel.getProperty("/simulation/fps")) >= 55 ? "↑ Smooth" : "↓ Needs optimization";
                    details = "Frame: " + hpcModel.getProperty("/simulation/frameMs") + "ms | Bodies: " + hpcModel.getProperty("/simulation/bodies");
                    historyData = hpcModel.getProperty("/history/fps") || [];
                    break;
                case "memory":
                    currentValue = hpcModel.getProperty("/memory/cacheHitRate") + "%";
                    trend = parseFloat(hpcModel.getProperty("/memory/cacheHitRate")) >= 90 ? "↑ Excellent" : "↓ Review layout";
                    details = "Bandwidth: " + hpcModel.getProperty("/memory/bandwidthGbps") + " GB/s | Heap: " + hpcModel.getProperty("/memory/heapMb") + " MB";
                    historyData = hpcModel.getProperty("/history/cache") || [];
                    break;
                case "system":
                    currentValue = hpcModel.getProperty("/uptime/display");
                    trend = "↑ Stable";
                    details = "WASM: " + hpcModel.getProperty("/wasm/sizeKb") + " KB | Load: " + hpcModel.getProperty("/wasm/loadMs") + " ms";
                    historyData = hpcModel.getProperty("/history/fps") || [];
                    break;
            }

            // Calculate stats
            const stats = this._calculateStats(historyData);

            drillDownModel.setProperty("/title", title);
            drillDownModel.setProperty("/currentValue", currentValue);
            drillDownModel.setProperty("/trend", trend);
            drillDownModel.setProperty("/trendClass", trend.startsWith("↑") ? "trendUp" : "trendDown");
            drillDownModel.setProperty("/details", details);
            drillDownModel.setProperty("/data", historyData);
            drillDownModel.setProperty("/min", stats.min.toFixed(1));
            drillDownModel.setProperty("/max", stats.max.toFixed(1));
            drillDownModel.setProperty("/avg", stats.avg.toFixed(1));
            drillDownModel.setProperty("/stdDev", stats.stdDev.toFixed(2));

            // Load and open fragment dialog
            if (!this._drillDownDialog) {
                Fragment.load({
                    id: view.getId(),
                    name: "galaxy.sim.view.fragments.DrillDownChart",
                    controller: this
                }).then(function (dialog) {
                    this._drillDownDialog = dialog;
                    view.addDependent(dialog);
                    dialog.open();
                    // Initial chart draw
                    setTimeout(() => this._updateDrillDownChart(), 100);
                }.bind(this));
            } else {
                this._drillDownDialog.open();
                setTimeout(() => this._updateDrillDownChart(), 100);
            }
        },

        _calculateStats: function (data) {
            if (!data || data.length === 0) {
                return { min: 0, max: 0, avg: 0, stdDev: 0 };
            }
            const min = Math.min(...data);
            const max = Math.max(...data);
            const avg = data.reduce((a, b) => a + b, 0) / data.length;
            const variance = data.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / data.length;
            const stdDev = Math.sqrt(variance);
            return { min, max, avg, stdDev };
        },

        _updateDrillDownChart: function () {
            const canvas = document.getElementById(this.getView().getId() + "--drillDownCanvas");
            if (!canvas) return;

            const ctx = canvas.getContext("2d");
            const drillDownModel = this.getView().getModel("drillDown");
            const data = drillDownModel.getProperty("/data") || [];
            const chartType = drillDownModel.getProperty("/chartType") || "line";

            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (data.length < 2) {
                ctx.fillStyle = "#888";
                ctx.font = "14px -apple-system, BlinkMacSystemFont, sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("Collecting data... (" + data.length + "/60 samples)", canvas.width / 2, canvas.height / 2);
                return;
            }

            const padding = { top: 20, right: 20, bottom: 30, left: 50 };
            const chartWidth = canvas.width - padding.left - padding.right;
            const chartHeight = canvas.height - padding.top - padding.bottom;

            const min = Math.min(...data) * 0.95;
            const max = Math.max(...data) * 1.05;
            const range = max - min || 1;

            // Draw grid
            ctx.strokeStyle = "rgba(0,0,0,0.08)";
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {
                const y = padding.top + (chartHeight * i / 5);
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(padding.left + chartWidth, y);
                ctx.stroke();

                // Y-axis labels
                const value = max - (range * i / 5);
                ctx.fillStyle = "#666";
                ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
                ctx.textAlign = "right";
                ctx.fillText(value.toFixed(1), padding.left - 8, y + 4);
            }

            if (chartType === "line") {
                // Draw line chart with gradient fill
                const gradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartHeight);
                gradient.addColorStop(0, "rgba(0, 122, 255, 0.3)");
                gradient.addColorStop(1, "rgba(0, 122, 255, 0.02)");

                // Fill area
                ctx.beginPath();
                ctx.moveTo(padding.left, padding.top + chartHeight);
                data.forEach((val, i) => {
                    const x = padding.left + (i / (data.length - 1)) * chartWidth;
                    const y = padding.top + chartHeight - ((val - min) / range) * chartHeight;
                    ctx.lineTo(x, y);
                });
                ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
                ctx.closePath();
                ctx.fillStyle = gradient;
                ctx.fill();

                // Draw line
                ctx.beginPath();
                ctx.strokeStyle = "#007aff";
                ctx.lineWidth = 2.5;
                ctx.lineJoin = "round";
                data.forEach((val, i) => {
                    const x = padding.left + (i / (data.length - 1)) * chartWidth;
                    const y = padding.top + chartHeight - ((val - min) / range) * chartHeight;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                });
                ctx.stroke();

                // Draw current value dot
                if (data.length > 0) {
                    const lastVal = data[data.length - 1];
                    const x = padding.left + chartWidth;
                    const y = padding.top + chartHeight - ((lastVal - min) / range) * chartHeight;

                    ctx.beginPath();
                    ctx.arc(x, y, 6, 0, Math.PI * 2);
                    ctx.fillStyle = "#007aff";
                    ctx.fill();
                    ctx.strokeStyle = "#fff";
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            } else {
                // Draw bar chart
                const barWidth = chartWidth / data.length * 0.8;
                const barGap = chartWidth / data.length * 0.2;

                data.forEach((val, i) => {
                    const x = padding.left + (i / data.length) * chartWidth + barGap / 2;
                    const barHeight = ((val - min) / range) * chartHeight;
                    const y = padding.top + chartHeight - barHeight;

                    const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
                    gradient.addColorStop(0, "#007aff");
                    gradient.addColorStop(1, "#5856d6");

                    ctx.fillStyle = gradient;
                    ctx.fillRect(x, y, barWidth, barHeight);
                });
            }

            // X-axis labels (time)
            ctx.fillStyle = "#666";
            ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("-60s", padding.left, canvas.height - 8);
            ctx.fillText("-30s", padding.left + chartWidth / 2, canvas.height - 8);
            ctx.fillText("now", padding.left + chartWidth, canvas.height - 8);
        },

        onCloseDrillDown: function () {
            if (this._drillDownDialog) {
                this._drillDownDialog.close();
            }
        },

        onExit: function () {
            if (this._socket) {
                this._socket.disconnect();
            }
            if (this._drillDownDialog) {
                this._drillDownDialog.destroy();
            }
        }

    });
});