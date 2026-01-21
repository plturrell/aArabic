#!/bin/bash
# Install all Rust CLI binaries to system PATH
# This script copies compiled binaries to /usr/local/bin

set -e  # Exit on error

echo "📦 Installing Rust CLI clients..."
echo "=================================="

# Check if running with appropriate permissions
if [ ! -w "/usr/local/bin" ]; then
    echo "⚠️  Warning: /usr/local/bin is not writable"
    echo "   You may need to run with sudo: sudo ./install_rust_clients.sh"
    echo "   Or install to user directory: ./install_rust_clients.sh --user"
    
    if [ "$1" != "--user" ] && [ "$EUID" -ne 0 ]; then
        echo ""
        read -p "Continue with user install (~/.local/bin)? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        INSTALL_DIR="$HOME/.local/bin"
        mkdir -p "$INSTALL_DIR"
    else
        INSTALL_DIR="/usr/local/bin"
    fi
else
    INSTALL_DIR="/usr/local/bin"
fi

cd "$(dirname "$0")/../src/serviceAutomation"

# Counter for tracking
total=0
installed=0
skipped=0

# CLI binaries to install
cli_binaries=(
    "gitea-api-client/target/release/gitea-cli"
    "git-api-client/target/release/git-cli"
    "postgres-api-client/target/release/postgres-cli"
    "kafka-api-client/target/release/kafka-cli"
    "apisix-api-client/target/release/apisix-cli"
    "keycloak-api-client/target/release/keycloak-cli"
    "filesystem-api-client/target/release/fs-cli"
    "memory-api-client/target/release/memory-cli"
    "shimmy-api-client/target/release/shimmy-cli"
    "marquez-api-client/target/release/marquez-cli"
    "n8n-api-client/target/release/n8n-cli"
    "opencanvas-api-client/target/release/opencanvas-cli"
    "hyperbook-api-client/target/release/hyperbook-cli"
    "qdrant-api-client/target/release/qdrant-cli"
    "memgraph-api-client/target/release/memgraph-cli"
    "dragonflydb-api-client/target/release/dragonflydb-cli"
    "ncode-api-client/target/release/ncode-cli"
    "lean4-api-client/target/release/lean4-cli"
)

for binary_path in "${cli_binaries[@]}"; do
    total=$((total + 1))
    binary_name=$(basename "$binary_path")
    
    if [ -f "$binary_path" ]; then
        echo "Installing $binary_name..."
        cp "$binary_path" "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/$binary_name"
        installed=$((installed + 1))
        echo "✅ $binary_name installed to $INSTALL_DIR"
    else
        skipped=$((skipped + 1))
        echo "⚠️  $binary_name not found (not built yet?), skipping..."
    fi
done

echo ""
echo "=================================="
echo "📊 Installation Summary:"
echo "   Total:     $total CLIs"
echo "   Installed: $installed CLIs"
echo "   Skipped:   $skipped CLIs"
echo "   Location:  $INSTALL_DIR"
echo "=================================="

if [ $installed -gt 0 ]; then
    echo "✅ Installation complete!"
    echo ""
    echo "Verify installation:"
    echo "  gitea-cli --version"
    echo "  postgres-cli --help"
    echo ""
    
    if [ "$INSTALL_DIR" = "$HOME/.local/bin" ]; then
        echo "⚠️  User install detected. Make sure $INSTALL_DIR is in your PATH:"
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo ""
        echo "Add to ~/.bashrc or ~/.zshrc to make permanent"
    fi
    
    echo "Next step: Update Python service adapters to use rust_cli_adapter.py"
    exit 0
else
    echo "❌ No CLIs were installed"
    echo "   Run ./build_rust_clients.sh first to build the binaries"
    exit 1
fi
