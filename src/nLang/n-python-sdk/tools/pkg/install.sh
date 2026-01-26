#!/bin/bash

# Mojo Package Manager - Installation Script
# Installs mojo-pkg to your system

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if running from correct directory
if [ ! -f "build.zig" ] || [ ! -f "VERSION" ]; then
    print_error "Please run this script from the mojo-pkg directory"
    print_info "cd src/serviceCore/serviceShimmy-mojo/mojo-sdk/tools/pkg"
    exit 1
fi

print_header "Mojo Package Manager Installer"
VERSION=$(cat VERSION | tr -d '\n' || echo 'unknown')
echo "Version: $VERSION"
echo ""

# Check for Zig
print_info "Checking for Zig..."
if ! command -v zig &> /dev/null; then
    print_error "Zig is not installed"
    print_info "Please install Zig 0.15.2 or later from https://ziglang.org/"
    exit 1
fi

ZIG_VERSION=$(zig version)
print_success "Found Zig $ZIG_VERSION"

# Build mojo-pkg
print_info "Building mojo-pkg..."
if zig build 2>&1 | grep -q "error"; then
    print_error "Build failed"
    exit 1
fi

if [ ! -f "zig-out/bin/mojo-pkg" ]; then
    print_error "Build succeeded but executable not found"
    exit 1
fi

print_success "Build successful"

# Check executable
SIZE=$(ls -lh zig-out/bin/mojo-pkg | awk '{print $5}')
print_info "Executable size: $SIZE"

# Installation options
echo ""
print_header "Installation Options"
echo ""
echo "1. Install to /usr/local/bin (system-wide, requires sudo)"
echo "2. Install to ~/.local/bin (user-only, no sudo)"
echo "3. Skip installation (just build)"
echo ""

read -p "Select option [1-3]: " choice

case $choice in
    1)
        # System-wide installation
        INSTALL_DIR="/usr/local/bin"
        print_info "Installing to $INSTALL_DIR..."
        
        if [ ! -w "$INSTALL_DIR" ]; then
            print_info "Requires sudo for system-wide installation"
            sudo cp zig-out/bin/mojo-pkg "$INSTALL_DIR/"
        else
            cp zig-out/bin/mojo-pkg "$INSTALL_DIR/"
        fi
        
        if [ -f "$INSTALL_DIR/mojo-pkg" ]; then
            print_success "Installed to $INSTALL_DIR/mojo-pkg"
        else
            print_error "Installation failed"
            exit 1
        fi
        ;;
        
    2)
        # User installation
        INSTALL_DIR="$HOME/.local/bin"
        mkdir -p "$INSTALL_DIR"
        
        print_info "Installing to $INSTALL_DIR..."
        cp zig-out/bin/mojo-pkg "$INSTALL_DIR/"
        
        if [ -f "$INSTALL_DIR/mojo-pkg" ]; then
            print_success "Installed to $INSTALL_DIR/mojo-pkg"
            
            # Check if in PATH
            if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
                print_warning "$INSTALL_DIR is not in your PATH"
                print_info "Add this to your shell profile (~/.zshrc, ~/.bashrc, etc.):"
                echo ""
                echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
                echo ""
            fi
        else
            print_error "Installation failed"
            exit 1
        fi
        ;;
        
    3)
        # Skip installation
        print_info "Skipping installation"
        print_info "You can run mojo-pkg from: $(pwd)/zig-out/bin/mojo-pkg"
        print_info "Or add to PATH:"
        echo ""
        echo "    export PATH=\"$(pwd)/zig-out/bin:\$PATH\""
        echo ""
        exit 0
        ;;
        
    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

# Verify installation
echo ""
print_header "Verification"

if command -v mojo-pkg &> /dev/null; then
    INSTALLED_VERSION=$(mojo-pkg help 2>&1 | head -1 | grep -o "Mojo Package Manager" || echo "")
    if [ -n "$INSTALLED_VERSION" ]; then
        print_success "mojo-pkg is installed and working!"
        echo ""
        print_info "Try these commands:"
        echo "    mojo-pkg help"
        echo "    mojo-pkg init my-project"
        echo ""
    else
        print_warning "mojo-pkg found but may not be working correctly"
    fi
else
    print_warning "mojo-pkg not found in PATH"
    print_info "You may need to:"
    echo "  1. Restart your shell"
    echo "  2. Or run: source ~/.zshrc (or ~/.bashrc)"
fi

# Optional: Run tests
echo ""
read -p "Run integration tests? [y/N]: " run_tests

if [[ $run_tests =~ ^[Yy]$ ]]; then
    print_info "Running integration tests..."
    if [ -x "integration_test.sh" ]; then
        ./integration_test.sh
    else
        print_warning "Integration test script not found or not executable"
    fi
fi

echo ""
print_header "Installation Complete!"
echo ""
print_success "mojo-pkg v$VERSION is ready to use!"
echo ""
print_info "Quick start:"
echo "  1. See QUICKSTART.md for a 5-minute guide"
echo "  2. See README.md for full documentation"
echo "  3. See EXAMPLES.md for practical examples"
echo ""
print_info "Get help anytime with: mojo-pkg help"
echo ""
