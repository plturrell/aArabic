#!/bin/bash
# Fix broken adapter aliases

echo "Fixing adapter aliases..."

# Fix a2ui_enhanced
sed -i '' 's/A2UIEnhancedAdapter = A2UIEnhancedService/# A2UIEnhancedAdapter alias removed - no Service class exists/' src/serviceCore/adapters/a2ui_enhanced.py

# Fix apisix
sed -i '' 's/APISIXAdapter = APISIXService/# APISIXAdapter alias removed - no Service class exists/' src/serviceCore/adapters/apisix.py

# Fix keycloak  
sed -i '' 's/KeycloakAdapter = KeycloakService/# KeycloakAdapter alias removed - no Service class exists/' src/serviceCore/adapters/keycloak.py

# Fix marquez
sed -i '' 's/MarquezAdapter = MarquezService/# MarquezAdapter alias removed - no Service class exists/' src/serviceCore/adapters/marquez.py

# Fix nucleusgraph
sed -i '' 's/NucleusGraphAdapter = NucleusGraphService/# NucleusGraphAdapter alias removed - no Service class exists/' src/serviceCore/adapters/nucleusgraph.py

echo "✅ Fixed 5 broken adapter aliases!"