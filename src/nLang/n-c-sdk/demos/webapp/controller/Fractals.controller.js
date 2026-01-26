sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast"
], function (Controller, MessageToast) {
    "use strict";

    return Controller.extend("galaxy.sim.controller.Fractals", {

        onInit: function () {
            // Initialize fractal parameters
            this._fractalType = "mandelbrot";
            this._iterations = 100;
            this._zoom = 1;
            this._centerX = 0;
            this._centerY = 0;
            this._colorScheme = "rainbow";

            // Initialize after rendering with delay
            this.getView().addEventDelegate({
                onAfterRendering: function () {
                    setTimeout(function () {
                        this._initCanvas();
                        this._renderFractal();
                    }.bind(this), 300);
                }.bind(this)
            });
        },

        _initCanvas: function () {
            this._canvas = document.getElementById("fractalCanvas");
            if (!this._canvas) {
                console.error("Canvas not found");
                return;
            }

            this._ctx = this._canvas.getContext("2d");
            this._width = this._canvas.width;
            this._height = this._canvas.height;

            // Add mouse interaction
            this._canvas.addEventListener("click", this._onCanvasClick.bind(this));

            // Update pixel count
            this.byId("pixelCount").setText((this._width * this._height).toLocaleString());
        },

        _renderFractal: function () {
            if (!this._ctx) return;

            const startTime = performance.now();

            switch (this._fractalType) {
                case "mandelbrot":
                    this._renderMandelbrot();
                    break;
                case "julia":
                    this._renderJulia();
                    break;
                case "sierpinski":
                    this._renderSierpinski();
                    break;
            }

            const renderTime = Math.round(performance.now() - startTime);
            this.byId("renderTime").setText(renderTime + "ms");
            this._updateStatistics();
        },

        _renderMandelbrot: function () {
            const imageData = this._ctx.createImageData(this._width, this._height);
            const data = imageData.data;

            const minRe = -2.5 / this._zoom + this._centerX;
            const maxRe = 1.0 / this._zoom + this._centerX;
            const minIm = -1.0 / this._zoom + this._centerY;
            const maxIm = 1.0 / this._zoom + this._centerY;

            for (let y = 0; y < this._height; y++) {
                for (let x = 0; x < this._width; x++) {
                    const c_re = minRe + (x / this._width) * (maxRe - minRe);
                    const c_im = minIm + (y / this._height) * (maxIm - minIm);

                    let z_re = 0, z_im = 0;
                    let iteration = 0;

                    while (z_re * z_re + z_im * z_im <= 4 && iteration < this._iterations) {
                        const temp = z_re * z_re - z_im * z_im + c_re;
                        z_im = 2 * z_re * z_im + c_im;
                        z_re = temp;
                        iteration++;
                    }

                    const color = this._getColor(iteration, this._iterations);
                    const pixelIndex = (y * this._width + x) * 4;
                    data[pixelIndex] = color.r;
                    data[pixelIndex + 1] = color.g;
                    data[pixelIndex + 2] = color.b;
                    data[pixelIndex + 3] = 255;
                }
            }

            this._ctx.putImageData(imageData, 0, 0);
        },

        _renderJulia: function () {
            const imageData = this._ctx.createImageData(this._width, this._height);
            const data = imageData.data;

            // Julia set constant
            const c_re = -0.7;
            const c_im = 0.27015;

            const minRe = -1.5 / this._zoom + this._centerX;
            const maxRe = 1.5 / this._zoom + this._centerX;
            const minIm = -1.0 / this._zoom + this._centerY;
            const maxIm = 1.0 / this._zoom + this._centerY;

            for (let y = 0; y < this._height; y++) {
                for (let x = 0; x < this._width; x++) {
                    let z_re = minRe + (x / this._width) * (maxRe - minRe);
                    let z_im = minIm + (y / this._height) * (maxIm - minIm);

                    let iteration = 0;

                    while (z_re * z_re + z_im * z_im <= 4 && iteration < this._iterations) {
                        const temp = z_re * z_re - z_im * z_im + c_re;
                        z_im = 2 * z_re * z_im + c_im;
                        z_re = temp;
                        iteration++;
                    }

                    const color = this._getColor(iteration, this._iterations);
                    const pixelIndex = (y * this._width + x) * 4;
                    data[pixelIndex] = color.r;
                    data[pixelIndex + 1] = color.g;
                    data[pixelIndex + 2] = color.b;
                    data[pixelIndex + 3] = 255;
                }
            }

            this._ctx.putImageData(imageData, 0, 0);
        },

        _renderSierpinski: function () {
            this._ctx.fillStyle = "#fafafa";
            this._ctx.fillRect(0, 0, this._width, this._height);

            const depth = Math.min(Math.floor(this._iterations / 50), 10);

            const triangle = [
                { x: this._width / 2, y: 50 },
                { x: 50, y: this._height - 50 },
                { x: this._width - 50, y: this._height - 50 }
            ];

            this._drawSierpinski(triangle, depth);
        },

        _drawSierpinski: function (triangle, depth) {
            if (depth === 0) {
                this._ctx.beginPath();
                this._ctx.moveTo(triangle[0].x, triangle[0].y);
                this._ctx.lineTo(triangle[1].x, triangle[1].y);
                this._ctx.lineTo(triangle[2].x, triangle[2].y);
                this._ctx.closePath();
                this._ctx.fillStyle = "#0a6ed1";
                this._ctx.fill();
                this._ctx.strokeStyle = "#333";
                this._ctx.stroke();
            } else {
                const mid = [
                    { x: (triangle[0].x + triangle[1].x) / 2, y: (triangle[0].y + triangle[1].y) / 2 },
                    { x: (triangle[1].x + triangle[2].x) / 2, y: (triangle[1].y + triangle[2].y) / 2 },
                    { x: (triangle[2].x + triangle[0].x) / 2, y: (triangle[2].y + triangle[0].y) / 2 }
                ];

                this._drawSierpinski([triangle[0], mid[0], mid[2]], depth - 1);
                this._drawSierpinski([mid[0], triangle[1], mid[1]], depth - 1);
                this._drawSierpinski([mid[2], mid[1], triangle[2]], depth - 1);
            }
        },

        _getColor: function (iteration, maxIterations) {
            if (iteration === maxIterations) {
                return { r: 0, g: 0, b: 0 };
            }

            const t = iteration / maxIterations;

            switch (this._colorScheme) {
                case "rainbow":
                    return {
                        r: Math.floor(9 * (1 - t) * t * t * t * 255),
                        g: Math.floor(15 * (1 - t) * (1 - t) * t * t * 255),
                        b: Math.floor(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255)
                    };
                case "fire":
                    return {
                        r: Math.floor(255 * t),
                        g: Math.floor(128 * t * t),
                        b: 0
                    };
                case "ice":
                    return {
                        r: Math.floor(128 * (1 - t)),
                        g: Math.floor(200 * (1 - t)),
                        b: Math.floor(255 * t)
                    };
                case "grayscale":
                    const gray = Math.floor(255 * t);
                    return { r: gray, g: gray, b: gray };
                default:
                    return { r: 255, g: 255, b: 255 };
            }
        },

        _updateStatistics: function () {
            this.byId("centerCoords").setText(`(${this._centerX.toFixed(4)}, ${this._centerY.toFixed(4)})`);
            this.byId("zoomLevel").setText(this._zoom.toFixed(2) + "x");
            this.byId("zoomValue").setText(this._zoom.toFixed(2) + "x");
        },

        _onCanvasClick: function (event) {
            const rect = this._canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            // Convert to fractal coordinates
            const minRe = -2.5 / this._zoom + this._centerX;
            const maxRe = 1.0 / this._zoom + this._centerX;
            const minIm = -1.0 / this._zoom + this._centerY;
            const maxIm = 1.0 / this._zoom + this._centerY;

            this._centerX = minRe + (x / this._width) * (maxRe - minRe);
            this._centerY = minIm + (y / this._height) * (maxIm - minIm);
            this._zoom *= 2;

            this.byId("zoomSlider").setValue(Math.min(this._zoom, 100));
            this._renderFractal();
        },

        onFractalTypeChange: function (oEvent) {
            this._fractalType = oEvent.getParameter("key");
            this.onResetView();
        },

        onParameterChange: function () {
            this._iterations = parseInt(this.byId("iterationsSlider").getValue());
            this._zoom = parseFloat(this.byId("zoomSlider").getValue());

            this.byId("iterationsValue").setText(this._iterations.toString());
            this._renderFractal();
        },

        onColorSchemeChange: function (oEvent) {
            this._colorScheme = oEvent.getParameter("selectedItem").getKey();
            this._renderFractal();
        },

        onResetView: function () {
            this._centerX = 0;
            this._centerY = 0;
            this._zoom = 1;
            this._iterations = 100;

            this.byId("zoomSlider").setValue(1);
            this.byId("iterationsSlider").setValue(100);
            this.byId("iterationsValue").setText("100");

            this._renderFractal();
            MessageToast.show("View reset");
        },

        onZoomIn: function () {
            this._zoom *= 1.5;
            this.byId("zoomSlider").setValue(Math.min(this._zoom, 100));
            this._renderFractal();
        },

        onZoomOut: function () {
            this._zoom /= 1.5;
            this.byId("zoomSlider").setValue(Math.max(this._zoom, 0.1));
            this._renderFractal();
        },

        onPanLeft: function () {
            this._centerX -= 0.1 / this._zoom;
            this._renderFractal();
        },

        onPanRight: function () {
            this._centerX += 0.1 / this._zoom;
            this._renderFractal();
        },

        onDownload: function () {
            if (!this._canvas) return;

            const link = document.createElement('a');
            link.download = `fractal-${this._fractalType}-${Date.now()}.png`;
            link.href = this._canvas.toDataURL();
            link.click();

            MessageToast.show("Fractal image downloaded");
        }

    });
});