#!/bin/bash

# Mojo Package Manager - Uninstallation Script
# Removes mojo-pkg from your system

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

print_header "Mojo Package Manager Uninstaller"
echo ""

# Check installation locations
SYSTEM_INSTALL="/usr/local/bin/mojo-pkg"
USER_INSTALL="$HOME/.local/bin/mojo-pkg"

found=false

# Check system installation
if [ -f "$SYSTEM_INSTALL" ]; then
    found=true
    print_info "Found system installation: $SYSTEM_INSTALL"
fi

# Check user installation
if [ -f "$USER_INSTALL" ]; then
    found=true
    print_info "Found user installation: $USER_INSTALL"
fi

if [ "$found" = false ]; then
    print_warning "mojo-pkg not found in standard locations"
    print_info "Checked:"
    echo "  - $SYSTEM_INSTALL"
    echo "  - $USER_INSTALL"
    echo ""
    exit 0
fi

echo ""
read -p "Remove mojo-pkg? [y/N]: " confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    print_info "Uninstallation cancelled"
    exit 0
fi

echo ""

# Remove system installation
if [ -f "$SYSTEM_INSTALL" ]; then
    print_info "Removing $SYSTEM_INSTALL..."
    if [ ! -w "$(dirname $SYSTEM_INSTALL)" ]; then
        sudo rm "$SYSTEM_INSTALL"
    else
        rm "$SYSTEM_INSTALL"
    fi
    
    if [ ! -f "$SYSTEM_INSTALL" ]; then
        print_success "Removed system installation"
    else
        print_error "Failed to remove system installation"
    fi
fi

# Remove user installation
if [ -f "$USER_INSTALL" ]; then
    print_info "Removing $USER_INSTALL..."
    rm "$USER_INSTALL"
    
    if [ ! -f "$USER_INSTALL" ]; then
        print_success "Removed user installation"
    else
        print_error "Failed to remove user installation"
    fi
fi

# Verify removal
echo ""
print_header "Verification"

if command -v mojo-pkg &> /dev/null; then
    print_warning "mojo-pkg still found in PATH"
    WHICH_OUTPUT=$(which mojo-pkg)
    print_info "Location: $WHICH_OUTPUT"
    print_info "You may need to manually remove it"
else
    print_success "mojo-pkg successfully uninstalled"
fi

echo ""
print_header "Uninstallation Complete"
echo ""
print_info "Note: This script only removes the mojo-pkg executable."
print_info "Your projects and their mojo.toml files remain untouched."
echo ""
