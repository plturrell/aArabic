#!/bin/bash
# Fix all adapter aliases correctly

echo "Fixing all adapter aliases..."

# apisix - check what class exists
if grep -q "^class APISIX[^S]" src/serviceCore/adapters/apisix.py; then
    echo "apisix: Found APISIX class"
    sed -i '' 's/# APISIXAdapter alias removed.*/APISIXAdapter = APISIX/' src/serviceCore/adapters/apisix.py
else
    echo "apisix: No APISIX class found, keeping comment"
fi

# keycloak  
if grep -q "^class Keycloak[^S]" src/serviceCore/adapters/keycloak.py; then
    echo "keycloak: Found Keycloak class"
    sed -i '' 's/# KeycloakAdapter alias removed.*/KeycloakAdapter = Keycloak/' src/serviceCore/adapters/keycloak.py
else
    echo "keycloak: No Keycloak class found, keeping comment"
fi

# marquez
if grep -q "^class Marquez[^S]" src/serviceCore/adapters/marquez.py; then
    echo "marquez: Found Marquez class"
    sed -i '' 's/# MarquezAdapter alias removed.*/MarquezAdapter = Marquez/' src/serviceCore/adapters/marquez.py
else
    echo "marquez: No Marquez class found, keeping comment"
fi

# nucleusgraph
if grep -q "^class NucleusGraph[^S]" src/serviceCore/adapters/nucleusgraph.py; then
    echo "nucleusgraph: Found NucleusGraph class"
    sed -i '' 's/# NucleusGraphAdapter alias removed.*/NucleusGraphAdapter = NucleusGraph/' src/serviceCore/adapters/nucleusgraph.py
else
    echo "nucleusgraph: No NucleusGraph class found, keeping comment"
fi

echo "✅ All adapter aliases fixed!"
