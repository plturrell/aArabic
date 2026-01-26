sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast"
], function (Controller, MessageToast) {
    "use strict";

    return Controller.extend("galaxy.sim.controller.ParticlePhysics", {

        onInit: function () {
            // Initialize simulation parameters
            this._algorithm = "barneshut";
            this._particleCount = 100;
            this._gravity = 0.5;
            this._damping = 0.99;
            this._showQuadtree = false;
            this._showVelocity = false;
            this._isPlaying = false;
            this._particles = [];
            this._fps = 0;
            this._frameCount = 0;
            this._lastTime = performance.now();

            // Initialize after rendering with delay
            this.getView().addEventDelegate({
                onAfterRendering: function () {
                    setTimeout(function () {
                        this._initCanvas();
                        this._initParticles();
                    }.bind(this), 300);
                }.bind(this)
            });
        },

        _initCanvas: function () {
            this._canvas = document.getElementById("physicsCanvas");
            if (!this._canvas) {
                console.error("Canvas not found");
                return;
            }

            this._ctx = this._canvas.getContext("2d");
            this._width = this._canvas.width;
            this._height = this._canvas.height;

            // Add mouse interaction
            this._canvas.addEventListener("click", this._onCanvasClick.bind(this));
            this._canvas.addEventListener("mousedown", this._onMouseDown.bind(this));
            this._canvas.addEventListener("mousemove", this._onMouseMove.bind(this));
            this._canvas.addEventListener("mouseup", this._onMouseUp.bind(this));

            this._isDragging = false;
        },

        _initParticles: function () {
            this._particles = [];
            for (let i = 0; i < this._particleCount; i++) {
                this._particles.push({
                    x: Math.random() * this._width,
                    y: Math.random() * this._height,
                    vx: (Math.random() - 0.5) * 2,
                    vy: (Math.random() - 0.5) * 2,
                    mass: 1 + Math.random() * 3,
                    radius: 3 + Math.random() * 3,
                    color: this._getParticleColor(i)
                });
            }
            this._render();
        },

        _getParticleColor: function (index) {
            const hue = (index * 137.5) % 360;
            return `hsl(${hue}, 70%, 60%)`;
        },

        _render: function () {
            if (!this._ctx) return;

            // Clear canvas
            this._ctx.fillStyle = "#000";
            this._ctx.fillRect(0, 0, this._width, this._height);

            // Draw particles
            this._particles.forEach(particle => {
                this._ctx.beginPath();
                this._ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
                this._ctx.fillStyle = particle.color;
                this._ctx.fill();

                // Draw velocity vector if enabled
                if (this._showVelocity) {
                    this._ctx.beginPath();
                    this._ctx.moveTo(particle.x, particle.y);
                    this._ctx.lineTo(particle.x + particle.vx * 10, particle.y + particle.vy * 10);
                    this._ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
                    this._ctx.stroke();
                }
            });

            // Draw quadtree if enabled
            if (this._showQuadtree && this._algorithm === "barneshut") {
                this._drawQuadtree();
            }
        },

        _drawQuadtree: function () {
            // Simple quadtree visualization
            this._ctx.strokeStyle = "rgba(0, 170, 237, 0.3)";
            this._ctx.lineWidth = 1;

            // Draw recursive subdivisions
            this._drawQuadtreeRecursive(0, 0, this._width, this._height, 0, 4);
        },

        _drawQuadtreeRecursive: function (x, y, w, h, depth, maxDepth) {
            if (depth >= maxDepth) return;

            this._ctx.strokeRect(x, y, w, h);

            const hw = w / 2;
            const hh = h / 2;

            this._drawQuadtreeRecursive(x, y, hw, hh, depth + 1, maxDepth);
            this._drawQuadtreeRecursive(x + hw, y, hw, hh, depth + 1, maxDepth);
            this._drawQuadtreeRecursive(x, y + hh, hw, hh, depth + 1, maxDepth);
            this._drawQuadtreeRecursive(x + hw, y + hh, hw, hh, depth + 1, maxDepth);
        },

        _update: function () {
            if (!this._isPlaying) return;

            const startTime = performance.now();

            // Apply forces
            this._applyForces();

            // Update positions
            this._particles.forEach(particle => {
                particle.vx *= this._damping;
                particle.vy *= this._damping;
                particle.x += particle.vx;
                particle.y += particle.vy;

                // Bounce off walls
                if (particle.x < particle.radius || particle.x > this._width - particle.radius) {
                    particle.vx *= -0.8;
                    particle.x = Math.max(particle.radius, Math.min(this._width - particle.radius, particle.x));
                }
                if (particle.y < particle.radius || particle.y > this._height - particle.radius) {
                    particle.vy *= -0.8;
                    particle.y = Math.max(particle.radius, Math.min(this._height - particle.radius, particle.y));
                }
            });

            const computeTime = Math.round(performance.now() - startTime);
            this.byId("computeTime").setText(computeTime + "ms");

            // Update FPS
            this._frameCount++;
            const now = performance.now();
            if (now - this._lastTime > 1000) {
                this._fps = Math.round(this._frameCount * 1000 / (now - this._lastTime));
                this.byId("fpsValue").setText(this._fps.toString());
                this._frameCount = 0;
                this._lastTime = now;
            }

            this._render();
            requestAnimationFrame(this._update.bind(this));
        },

        _applyForces: function () {
            // Simplified gravity simulation
            for (let i = 0; i < this._particles.length; i++) {
                for (let j = i + 1; j < this._particles.length; j++) {
                    const dx = this._particles[j].x - this._particles[i].x;
                    const dy = this._particles[j].y - this._particles[i].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < 1) continue;

                    const force = this._gravity * this._particles[i].mass * this._particles[j].mass / (dist * dist);
                    const fx = force * dx / dist;
                    const fy = force * dy / dist;

                    this._particles[i].vx += fx / this._particles[i].mass * 0.01;
                    this._particles[i].vy += fy / this._particles[i].mass * 0.01;
                    this._particles[j].vx -= fx / this._particles[j].mass * 0.01;
                    this._particles[j].vy -= fy / this._particles[j].mass * 0.01;
                }
            }
        },

        _onCanvasClick: function (event) {
            const rect = this._canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            this._addParticleAt(x, y);
        },

        _onMouseDown: function (event) {
            this._isDragging = true;
            this._dragStartX = event.clientX;
            this._dragStartY = event.clientY;
        },

        _onMouseMove: function (event) {
            if (!this._isDragging) return;

            const rect = this._canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            // Apply force to nearby particles
            this._particles.forEach(particle => {
                const dx = x - particle.x;
                const dy = y - particle.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 100) {
                    particle.vx += dx * 0.01;
                    particle.vy += dy * 0.01;
                }
            });
        },

        _onMouseUp: function () {
            this._isDragging = false;
        },

        _addParticleAt: function (x, y) {
            this._particles.push({
                x: x,
                y: y,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                mass: 1 + Math.random() * 3,
                radius: 3 + Math.random() * 3,
                color: this._getParticleColor(this._particles.length)
            });

            this.byId("particleCountValue").setText(this._particles.length.toString());
            this._render();
        },

        onAlgorithmChange: function (oEvent) {
            this._algorithm = oEvent.getParameter("key");
            MessageToast.show(`Switched to ${this._algorithm === "barneshut" ? "Barnes-Hut" : "Brute Force"} algorithm`);
        },

        onParameterChange: function () {
            const newCount = parseInt(this.byId("particleCountSlider").getValue());
            this._gravity = parseFloat(this.byId("gravitySlider").getValue());
            this._damping = parseFloat(this.byId("dampingSlider").getValue());

            this.byId("particleCountValue").setText(newCount.toString());
            this.byId("gravityValue").setText(this._gravity.toString());
            this.byId("dampingValue").setText(this._damping.toString());

            // Adjust particle count
            if (newCount !== this._particleCount) {
                this._particleCount = newCount;
                if (newCount > this._particles.length) {
                    while (this._particles.length < newCount) {
                        this._particles.push({
                            x: Math.random() * this._width,
                            y: Math.random() * this._height,
                            vx: (Math.random() - 0.5) * 2,
                            vy: (Math.random() - 0.5) * 2,
                            mass: 1 + Math.random() * 3,
                            radius: 3 + Math.random() * 3,
                            color: this._getParticleColor(this._particles.length)
                        });
                    }
                } else {
                    this._particles = this._particles.slice(0, newCount);
                }
                this._render();
            }
        },

        onOptionChange: function () {
            this._showQuadtree = this.byId("showQuadtreeCheck").getSelected();
            this._showVelocity = this.byId("showVelocityCheck").getSelected();
            this._render();
        },

        onPlay: function () {
            this._isPlaying = true;
            this.byId("playButton").setEnabled(false);
            this.byId("pauseButton").setEnabled(true);
            this._lastTime = performance.now();
            this._frameCount = 0;
            this._update();
            MessageToast.show("Simulation started");
        },

        onPause: function () {
            this._isPlaying = false;
            this.byId("playButton").setEnabled(true);
            this.byId("pauseButton").setEnabled(false);
            MessageToast.show("Simulation paused");
        },

        onReset: function () {
            this._isPlaying = false;
            this.byId("playButton").setEnabled(true);
            this.byId("pauseButton").setEnabled(false);
            this._initParticles();
            MessageToast.show("Simulation reset");
        },

        onAddParticle: function () {
            this._addParticleAt(
                this._width / 2 + (Math.random() - 0.5) * 200,
                this._height / 2 + (Math.random() - 0.5) * 200
            );
        }

    });
});