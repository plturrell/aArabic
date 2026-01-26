sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel"
], function (Controller, JSONModel) {
    "use strict";

    return Controller.extend("galaxy.sim.controller.Benchmarks", {

        onInit: function () {
            // Create model for HPC benchmark data
            const oModel = new JSONModel();
            
            // Load sample data (will be replaced with real benchmark results)
            const sampleData = {
                "timestamp": "2026-01-26T08:42:00Z",
                "system": {
                    "architecture": "x86_64",
                    "vector_width": 8,
                    "cache_line_bytes": 64
                },
                "stream": {
                    "copy_bw_gbs": 42.5,
                    "scale_bw_gbs": 41.8,
                    "add_bw_gbs": 38.2,
                    "triad_bw_gbs": 37.9
                },
                "linpack": {
                    "achieved_gflops": 245.3,
                    "theoretical_peak_gflops": 384.0,
                    "efficiency_percent": 63.9
                },
                "latency": {
                    "l1_ns": 1.2,
                    "l2_ns": 4.8,
                    "l3_ns": 18.5,
                    "dram_ns": 95.2
                },
                "strong_scaling": [
                    { "threads": 1, "time_ms": 1000, "speedup": 1.0, "efficiency_percent": 100.0 },
                    { "threads": 2, "time_ms": 520, "speedup": 1.92, "efficiency_percent": 96.0 },
                    { "threads": 4, "time_ms": 270, "speedup": 3.70, "efficiency_percent": 92.5 },
                    { "threads": 8, "time_ms": 145, "speedup": 6.90, "efficiency_percent": 86.3 },
                    { "threads": 16, "time_ms": 85, "speedup": 11.76, "efficiency_percent": 73.5 }
                ],
                "weak_scaling": [
                    { "threads": 1, "time_ms": 500, "efficiency_percent": 100.0 },
                    { "threads": 2, "time_ms": 510, "efficiency_percent": 98.0 },
                    { "threads": 4, "time_ms": 525, "efficiency_percent": 95.2 },
                    { "threads": 8, "time_ms": 550, "efficiency_percent": 90.9 },
                    { "threads": 16, "time_ms": 590, "efficiency_percent": 84.7 }
                ],
                "simd": {
                    "scalar_gflops": 12.5,
                    "vector_gflops": 65.2,
                    "speedup": 5.22,
                    "utilization_percent": 65.3
                },
                "sudoku_demo": {
                    "description": "Practical HPC metrics via Sudoku solving",
                    "algorithms": [
                        { "name": "Backtracking", "time_ms": 125.30, "solved": true, "flops_m": 0.73 },
                        { "name": "Bitmask", "time_ms": 58.20, "solved": true, "flops_m": 0.73 },
                        { "name": "Constraint Propagation", "time_ms": 42.10, "solved": true, "flops_m": 0.73 }
                    ],
                    "speedups": {
                        "backtracking_baseline": 1.0,
                        "bitmask_vs_backtracking": 2.15,
                        "constraint_vs_backtracking": 2.98
                    },
                    "memory_patterns": {
                        "grid_size_bytes": 81,
                        "cache_friendly": true,
                        "access_pattern": "row-major"
                    }
                }
            };

            oModel.setData(sampleData);
            this.getView().setModel(oModel, "hpcModel");

            // Try to load real data if available
            this._loadRealBenchmarkData(oModel);
        },

        _loadRealBenchmarkData: function (oModel) {
            // Attempt to load actual benchmark results
            fetch("model/hpc_benchmarks.json")
                .then(response => {
                    if (response.ok) {
                        return response.json();
                    }
                    throw new Error("Benchmark data not available");
                })
                .then(data => {
                    oModel.setData(data);
                    sap.m.MessageToast.show("Loaded real benchmark results");
                })
                .catch(() => {
                    // Use sample data (already set)
                    console.log("Using sample benchmark data");
                });
        },

        onNavBack: function () {
            const oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("home");
        },

        onExportJSON: function () {
            const oModel = this.getView().getModel("hpcModel");
            const data = oModel.getData();
            const jsonString = JSON.stringify(data, null, 2);
            
            // Create download
            const blob = new Blob([jsonString], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `hpc_benchmarks_${new Date().toISOString()}.json`;
            link.click();
            URL.revokeObjectURL(url);
            
            sap.m.MessageToast.show("Benchmark results exported");
        }

    });
});