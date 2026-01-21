#!/bin/bash
# Install SCIP indexers for all supported languages

set -e

echo "Installing SCIP indexers for nCode..."
echo "======================================="
echo ""

# Track what was installed
INSTALLED=()
SKIPPED=()

# TypeScript/JavaScript
if command -v npm &> /dev/null; then
    echo "Installing scip-typescript..."
    npm install -g @sourcegraph/scip-typescript
    INSTALLED+=("scip-typescript")
else
    SKIPPED+=("scip-typescript (npm not found)")
fi

# Python
if command -v pip &> /dev/null; then
    echo "Installing scip-python..."
    pip install scip-python
    INSTALLED+=("scip-python")
else
    SKIPPED+=("scip-python (pip not found)")
fi

# Go
if command -v go &> /dev/null; then
    echo "Installing scip-go..."
    go install github.com/sourcegraph/scip-go/cmd/scip-go@latest
    INSTALLED+=("scip-go")
else
    SKIPPED+=("scip-go (go not found)")
fi

# Rust (rust-analyzer)
if command -v rustup &> /dev/null; then
    echo "Installing rust-analyzer..."
    rustup component add rust-analyzer
    INSTALLED+=("rust-analyzer")
else
    SKIPPED+=("rust-analyzer (rustup not found)")
fi

# .NET
if command -v dotnet &> /dev/null; then
    echo "Installing scip-dotnet..."
    dotnet tool install -g scip-dotnet || dotnet tool update -g scip-dotnet
    INSTALLED+=("scip-dotnet")
else
    SKIPPED+=("scip-dotnet (dotnet not found)")
fi

# Ruby
if command -v gem &> /dev/null; then
    echo "Installing scip-ruby..."
    gem install scip-ruby || true
    INSTALLED+=("scip-ruby")
else
    SKIPPED+=("scip-ruby (gem not found)")
fi

# PHP
if command -v composer &> /dev/null; then
    echo "Installing scip-php..."
    composer global require sourcegraph/scip-php || true
    INSTALLED+=("scip-php")
else
    SKIPPED+=("scip-php (composer not found)")
fi

# Tree-sitter CLI for data languages
if command -v cargo &> /dev/null; then
    echo "Installing tree-sitter-cli..."
    cargo install tree-sitter-cli
    INSTALLED+=("tree-sitter-cli")
else
    SKIPPED+=("tree-sitter-cli (cargo not found)")
fi

# Print summary
echo ""
echo "======================================="
echo "Installation Summary"
echo "======================================="
echo ""

if [ ${#INSTALLED[@]} -gt 0 ]; then
    echo "Successfully installed:"
    for item in "${INSTALLED[@]}"; do
        echo "  ✓ $item"
    done
else
    echo "No indexers were installed."
fi

echo ""

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "Skipped (missing package manager):"
    for item in "${SKIPPED[@]}"; do
        echo "  ✗ $item"
    done
fi

echo ""
echo "Done!"

