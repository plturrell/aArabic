#!/bin/bash

# Install Mojo on macOS
# Official installation method from docs.modular.com

set -e

echo "=========================================="
echo "Mojo Installation Script"
echo "=========================================="
echo ""

# Check if already installed
if command -v mojo &> /dev/null; then
    echo "✓ Mojo is already installed:"
    mojo --version
    exit 0
fi

echo "Mojo not found. Installing..."
echo ""

# Check system requirements
echo "Checking system requirements..."
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "✗ This script is for macOS only"
    exit 1
fi

# Get CPU architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

if [[ "$ARCH" != "arm64" ]]; then
    echo "WARNING: Mojo is optimized for Apple Silicon (arm64)"
    echo "You're running on $ARCH - performance may vary"
fi

echo ""
echo "=========================================="
echo "Installation Options"
echo "=========================================="
echo ""
echo "Mojo requires the Modular CLI. You have two options:"
echo ""
echo "Option 1: Manual Installation (Recommended)"
echo "  1. Visit: https://developer.modular.com/download"
echo "  2. Sign up for a free account"
echo "  3. Download and run the installer"
echo "  4. Run: modular install mojo"
echo ""
echo "Option 2: Use magic CLI (if you have an auth token)"
echo "  curl -ssL https://magic.modular.com/install.sh | bash"
echo "  magic install mojo"
echo ""
echo "=========================================="
echo ""

# Create a helper script for post-installation
cat > "$HOME/.mojo_install_complete.sh" << 'EOF'
#!/bin/bash
# Run this after installing Mojo via the official installer

# Add to PATH if not already there
if ! grep -q 'modular' ~/.zshrc 2>/dev/null; then
    echo 'export MODULAR_HOME="$HOME/.modular"' >> ~/.zshrc
    echo 'export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"' >> ~/.zshrc
    echo "Added Mojo to ~/.zshrc"
fi

# Reload shell config
source ~/.zshrc 2>/dev/null || true

# Verify installation
if command -v mojo &> /dev/null; then
    echo "✓ Mojo installation verified:"
    mojo --version
else
    echo "✗ Mojo not found in PATH"
    echo "Try running: source ~/.zshrc"
fi
EOF

chmod +x "$HOME/.mojo_install_complete.sh"

echo "Installation helper script created at:"
echo "  $HOME/.mojo_install_complete.sh"
echo ""
echo "After installing Mojo via the official installer, run:"
echo "  bash $HOME/.mojo_install_complete.sh"
echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Install Mojo using one of the methods above"
echo "2. Run: bash $HOME/.mojo_install_complete.sh"
echo "3. Verify: mojo --version"
echo "4. Return to nAudioLab and run Day 2 tests"
echo ""
