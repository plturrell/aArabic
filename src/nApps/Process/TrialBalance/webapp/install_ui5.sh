#!/bin/bash

# Script to download and install OpenUI5 locally
# This eliminates CDN dependency

echo "📦 Downloading OpenUI5 SDK..."

UI5_VERSION="1.120.0"
UI5_URL="https://sdk.openui5.org/downloads/openui5-runtime-${UI5_VERSION}.zip"

# Create resources directory
mkdir -p resources

# Download UI5 SDK
echo "Downloading from ${UI5_URL}..."
curl -L -o openui5.zip "${UI5_URL}"

if [ $? -eq 0 ]; then
    echo "✅ Download successful"
    
    # Extract
    echo "📂 Extracting..."
    unzip -q openui5.zip -d resources/
    
    # Cleanup
    rm openui5.zip
    
    echo "✅ OpenUI5 installed successfully in ./resources/"
    echo ""
    echo "Update your index.html to use:"
    echo '  src="resources/resources/sap-ui-core.js"'
    echo ""
else
    echo "❌ Download failed. CDN might be blocked."
    echo ""
    echo "Alternative: Use npx to install via npm:"
    echo "  npx @ui5/cli@latest add sap.ui.core sap.m sap.f sap.ui.layout"
    echo ""
fi