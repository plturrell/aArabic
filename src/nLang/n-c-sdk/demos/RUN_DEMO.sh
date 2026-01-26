#!/bin/bash

# 🚀 Zig Performance Demo Suite - Quick Launch Script

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║        🚀 ZIG PERFORMANCE DEMO SUITE 🚀                  ║"
echo "║                                                          ║"
echo "║  Visual Performance Demonstration                        ║"
echo "║  50,000 Particles • Real-time Metrics                   ║"
echo "║  Language Comparisons • Interactive Controls            ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check if SDL2 is installed
if ! brew list sdl2 &>/dev/null; then
    echo "⚠️  SDL2 not found! Installing..."
    brew install sdl2
fi

# Build the demo if needed
if [ ! -f "visual_particle_demo_complete" ] || [ "visual_particle_demo_complete.zig" -nt "visual_particle_demo_complete" ]; then
    echo "🔨 Building demo (ReleaseFast mode)..."
    zig build-exe visual_particle_demo_complete.zig \
        -lc -lSDL2 \
        -I/opt/homebrew/include/SDL2 \
        -L/opt/homebrew/lib \
        -OReleaseFast
    echo "✅ Build complete!"
    echo ""
fi

# Display instructions
echo "🎮 CONTROLS:"
echo "  • LEFT MOUSE:  Attract particles"
echo "  • RIGHT MOUSE: Repel particles"
echo "  • SPACE:       Pause/Resume"
echo "  • M:           Toggle metrics overlay"
echo "  • R:           Reset simulation"
echo "  • ESC/Q:       Exit"
echo ""
echo "📊 METRICS DISPLAYED:"
echo "  • Real-time FPS and frame timing"
echo "  • Physics update and render times"
echo "  • Particles per second throughput"
echo "  • Memory usage"
echo "  • Language performance comparison bars"
echo ""
echo "🚀 Launching demo..."
echo ""

# Run the demo
./visual_particle_demo_complete

echo ""
echo "👋 Demo complete! Thanks for exploring Zig's performance!"
echo ""