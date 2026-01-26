sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/routing/History",
    "sap/ui/model/json/JSONModel"
], function (Controller, History, JSONModel) {
    "use strict";

    return Controller.extend("galaxy.sim.controller.GalaxySimulation", {

        onNavBack: function () {
            var oHistory = History.getInstance();
            var sPreviousHash = oHistory.getPreviousHash();

            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                var oRouter = this.getOwnerComponent().getRouter();
                oRouter.navTo("home", {}, true);
            }
        },

        onInit: function () {
            // Initialize model with default values
            const model = new JSONModel({
                scenario: "disk",
                bodyCount: 100000,
                theta: 0.5,
                dt: 0.01,
                running: false,
                stats: {
                    fps: 0,
                    frameTime: 0,
                    treeBuild: 0,
                    forceCalc: 0,
                    integration: 0,
                    kineticEnergy: 0,
                    potentialEnergy: 0,
                    totalEnergy: 0,
                    energyDrift: 0,
                    cpuUsage: 0,
                    bodyCount: 0
                }
            });
            
            this.getView().setModel(model);
            this.simulation = null;
            this.initialEnergy = null;
        },

        onCanvasReady: function () {
            // Get canvas element
            const canvas = document.getElementById("galaxy-render");
            if (!canvas) {
                console.error("Canvas not found!");
                return;
            }
            
            // Set canvas size to fill parent
            const parent = canvas.parentElement;
            canvas.width = parent.clientWidth;
            canvas.height = parent.clientHeight;
            
            // Initialize simulation
            this.simulation = new GalaxySimulation(canvas);
            
            // Handle window resize
            window.addEventListener("resize", () => {
                canvas.width = parent.clientWidth;
                canvas.height = parent.clientHeight;
                if (this.simulation) {
                    this.simulation.resize(canvas.width, canvas.height);
                }
            });
            
            // Start render loop
            this.startRenderLoop();
            
            // Get initial energy
            setTimeout(() => {
                const stats = this.simulation.getStats();
                this.initialEnergy = stats.totalEnergy;
            }, 100);
        },

        onStart: function () {
            if (this.simulation) {
                this.simulation.start();
                this.getView().getModel().setProperty("/running", true);
            }
        },

        onPause: function () {
            if (this.simulation) {
                this.simulation.pause();
                this.getView().getModel().setProperty("/running", false);
            }
        },

        onReset: function () {
            if (this.simulation) {
                this.simulation.reset();
                this.updateStats();
                
                // Reset initial energy
                setTimeout(() => {
                    const stats = this.simulation.getStats();
                    this.initialEnergy = stats.totalEnergy;
                }, 100);
            }
        },

        onScenarioChange: function (event) {
            const scenario = event.getParameter("selectedItem").getKey();
            if (this.simulation) {
                this.simulation.setScenario(scenario);
                
                // Reset initial energy for new scenario
                setTimeout(() => {
                    const stats = this.simulation.getStats();
                    this.initialEnergy = stats.totalEnergy;
                }, 100);
            }
        },

        onBodyCountChange: function (event) {
            const count = event.getParameter("value");
            if (this.simulation && !this.getView().getModel().getProperty("/running")) {
                this.simulation.setBodies(count);
                
                // Reset initial energy
                setTimeout(() => {
                    const stats = this.simulation.getStats();
                    this.initialEnergy = stats.totalEnergy;
                }, 100);
            }
        },

        onResetCamera: function () {
            if (this.simulation) {
                this.simulation.resetCamera();
            }
        },

        startRenderLoop: function () {
            const model = this.getView().getModel();
            
            const render = () => {
                if (!this.simulation) {
                    requestAnimationFrame(render);
                    return;
                }
                
                // Step simulation if running
                if (model.getProperty("/running")) {
                    this.simulation.step();
                }
                
                // Render frame
                this.simulation.render();
                
                // Update statistics (throttled to every 10 frames)
                if (this._frameCount === undefined) this._frameCount = 0;
                this._frameCount++;
                
                if (this._frameCount % 10 === 0) {
                    this.updateStats();
                }
                
                requestAnimationFrame(render);
            };
            
            requestAnimationFrame(render);
        },

        updateStats: function () {
            if (!this.simulation) return;
            
            const stats = this.simulation.getStats();
            const model = this.getView().getModel();
            
            // Calculate energy drift
            let energyDrift = 0;
            if (this.initialEnergy && this.initialEnergy !== 0) {
                energyDrift = Math.abs((stats.totalEnergy - this.initialEnergy) / this.initialEnergy) * 100;
            }
            
            // Update model
            model.setProperty("/stats/fps", stats.fps.toFixed(1));
            model.setProperty("/stats/frameTime", stats.frameTime.toFixed(2));
            model.setProperty("/stats/treeBuild", stats.treeBuild.toFixed(2));
            model.setProperty("/stats/forceCalc", stats.forceCalc.toFixed(2));
            model.setProperty("/stats/integration", stats.integration.toFixed(2));
            model.setProperty("/stats/kineticEnergy", stats.kineticEnergy.toFixed(6));
            model.setProperty("/stats/potentialEnergy", stats.potentialEnergy.toFixed(6));
            model.setProperty("/stats/totalEnergy", stats.totalEnergy.toFixed(6));
            model.setProperty("/stats/energyDrift", energyDrift.toFixed(4));
            model.setProperty("/stats/cpuUsage", stats.cpuUsage.toFixed(1));
            model.setProperty("/stats/bodyCount", stats.bodyCount);
        },

        onExit: function () {
            if (this.simulation) {
                this.simulation.dispose();
            }
        }
    });
});