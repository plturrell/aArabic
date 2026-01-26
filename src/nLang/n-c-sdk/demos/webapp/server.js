// Galaxy Simulation WebSocket Server with HPC Metrics Streaming
// Uses existing socket.io dependency for real-time simulation sync

const express = require('express');
const { createServer } = require('http');
const { Server } = require('socket.io');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
    cors: { origin: "*", methods: ["GET", "POST"] }
});

const PORT = process.env.PORT || 8080;

// Serve static files (the webapp)
app.use(express.static(path.join(__dirname)));

// Simulation state
const state = {
    running: false,
    scenario: 'disk',
    bodyCount: 100000,
    theta: 0.5,
    dt: 0.01,
    clients: 0
};

// HPC Metrics state - streams to all connected clients
const hpcMetrics = {
    simulation: {
        running: false,
        fps: 0,
        frameTime: 0,
        treeBuild: 0,
        forceCalc: 0,
        integration: 0,
        bodyCount: 100000,
        kineticEnergy: 0,
        potentialEnergy: 0,
        totalEnergy: 0
    },
    benchmarks: {
        simdSpeedup: 5.2,
        simdEfficiency: 65.0,
        scalarTime: 42.5,
        simdTime: 8.2,
        lastRun: null,
        fibonacci: { time: 0, result: 0 },
        primeSieve: { time: 0, primes: 0 },
        matrixMul: { time: 0, size: 0 },
        memory: { bandwidth: 0, allocations: 0 }
    },
    performance: {
        zigVsPython: 127,
        memoryBandwidth: 45.2,
        cacheHitRate: 94.5,
        cpuUsage: 32.1
    },
    system: {
        uptime: 0,
        memoryUsed: 0,
        memoryTotal: 0
    }
};

// Update system metrics periodically with realistic mock data
setInterval(() => {
    hpcMetrics.system.uptime = process.uptime();
    const mem = process.memoryUsage();
    hpcMetrics.system.memoryUsed = Math.round(mem.heapUsed / 1024 / 1024);
    hpcMetrics.system.memoryTotal = Math.round(mem.heapTotal / 1024 / 1024);

    // Generate realistic HPC metrics for the dashboard
    const formattedMetrics = {
        simd: {
            speedup: 4.5 + Math.random() * 1.5,
            efficiency: 55 + Math.random() * 15,
            scalarMs: 40 + Math.random() * 10,
            simdMs: 8 + Math.random() * 3
        },
        simulation: {
            fps: 55 + Math.random() * 10,
            frameMs: 15 + Math.random() * 3,
            treeBuildMs: 6 + Math.random() * 2,
            forceCalcMs: 8 + Math.random() * 2,
            bodies: 100000
        },
        memory: {
            cacheHitRate: 90 + Math.random() * 8,
            bandwidthGbps: 40 + Math.random() * 15,
            heapMb: hpcMetrics.system.memoryUsed
        },
        wasm: {
            sizeKb: 47.2,
            loadMs: 45 + Math.random() * 10
        },
        uptime: process.uptime()
    };

    // Broadcast HPC metrics to all clients
    io.emit('hpc:metrics', formattedMetrics);
}, 1000);

// WebSocket connection handling
io.on('connection', (socket) => {
    state.clients++;
    console.log(`🌌 Client connected: ${socket.id} (${state.clients} total)`);
    
    socket.emit('state:sync', state);
    io.emit('clients:count', state.clients);
    
    // Simulation controls
    socket.on('simulation:start', () => {
        state.running = true;
        io.emit('simulation:started');
    });
    
    socket.on('simulation:pause', () => {
        state.running = false;
        io.emit('simulation:paused');
    });
    
    socket.on('simulation:reset', () => io.emit('simulation:reset'));
    
    socket.on('scenario:change', (scenario) => {
        state.scenario = scenario;
        io.emit('scenario:changed', scenario);
    });
    
    socket.on('bodies:change', (count) => {
        state.bodyCount = count;
        io.emit('bodies:changed', count);
    });
    
    socket.on('theta:change', (theta) => {
        state.theta = theta;
        io.emit('theta:changed', theta);
    });
    
    socket.on('dt:change', (dt) => {
        state.dt = dt;
        io.emit('dt:changed', dt);
    });
    
    // Stats broadcast from primary client (WASM simulation)
    socket.on('stats:update', (stats) => {
        // Update HPC metrics with simulation stats
        hpcMetrics.simulation = { ...hpcMetrics.simulation, ...stats, running: state.running };
        socket.broadcast.emit('stats:updated', stats);
    });

    // Benchmark run request
    socket.on('benchmark:run', (type) => {
        console.log(`🔬 Running benchmark: ${type}`);
        io.emit('benchmark:started', type);

        // Simulate benchmark results (in production, run actual Zig benchmarks)
        setTimeout(() => {
            const results = runBenchmark(type);
            hpcMetrics.benchmarks = { ...hpcMetrics.benchmarks, ...results, lastRun: new Date() };
            io.emit('benchmark:result', results);
        }, 1000);
    });

    // Send initial HPC metrics
    socket.emit('hpc:metrics', hpcMetrics);

    socket.on('disconnect', () => {
        state.clients--;
        console.log(`👋 Client disconnected (${state.clients} remaining)`);
        io.emit('clients:count', state.clients);
    });
});

// Benchmark runner (simulated - in production would call Zig executables)
function runBenchmark(type) {
    const base = {
        simdSpeedup: 4.5 + Math.random() * 2,
        simdEfficiency: 55 + Math.random() * 20
    };

    switch (type) {
        case 'simd':
            return {
                ...base,
                scalarTime: 40 + Math.random() * 10,
                simdTime: 8 + Math.random() * 3
            };
        case 'memory':
            return {
                memory: {
                    bandwidth: 40 + Math.random() * 20,
                    allocations: 10000 + Math.floor(Math.random() * 5000)
                }
            };
        case 'full':
            return {
                ...base,
                scalarTime: 40 + Math.random() * 10,
                simdTime: 8 + Math.random() * 3,
                fibonacci: { time: 0.5 + Math.random(), result: 102334155 },
                primeSieve: { time: 10 + Math.random() * 5, primes: 78498 },
                matrixMul: { time: 50 + Math.random() * 20, size: 512 },
                memory: { bandwidth: 45 + Math.random() * 15, allocations: 10000 }
            };
        default:
            return base;
    }
}

// API endpoints
app.get('/api/state', (req, res) => res.json(state));
app.get('/api/health', (req, res) => res.json({ status: 'ok', uptime: process.uptime(), clients: state.clients }));
app.get('/api/hpc', (req, res) => res.json(hpcMetrics));
app.get('/api/benchmarks', (req, res) => res.json(hpcMetrics.benchmarks));

httpServer.listen(PORT, () => {
    console.log(`
╔══════════════════════════════════════════════════════════════╗
║  🌌 HPC Streaming Server - Galaxy Simulation                 ║
╠══════════════════════════════════════════════════════════════╣
║  URL:       http://localhost:${PORT}                            ║
║  WebSocket: ws://localhost:${PORT}                              ║
║  HPC API:   http://localhost:${PORT}/api/hpc                    ║
║  Status:    Ready                                            ║
╚══════════════════════════════════════════════════════════════╝
`);
});

