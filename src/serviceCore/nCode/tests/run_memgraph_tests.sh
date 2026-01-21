#!/bin/bash

# Memgraph Integration Test Runner for nCode
# This script checks prerequisites and runs the Memgraph test suite

set -e

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}nCode Memgraph Integration Test Runner${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    echo -e "${YELLOW}Please install Python 3 to run these tests${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found: $(python3 --version)${NC}"

# Check if neo4j Python driver is installed
if ! python3 -c "import neo4j" 2>/dev/null; then
    echo -e "${YELLOW}⚠ neo4j Python driver not found${NC}"
    echo -e "${BLUE}Installing neo4j driver...${NC}"
    pip3 install neo4j
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ neo4j driver installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install neo4j driver${NC}"
        echo -e "${YELLOW}Please install manually: pip3 install neo4j${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ neo4j Python driver found${NC}"
fi

# Check if Memgraph is running
echo -e "\n${BLUE}Checking Memgraph status...${NC}"
if ! docker ps | grep -q memgraph; then
    echo -e "${YELLOW}⚠ Memgraph container not running${NC}"
    echo -e "${BLUE}Attempting to start Memgraph...${NC}"
    
    # Check if we're in the project root
    if [ -f "docker-compose.yml" ]; then
        docker-compose up -d memgraph
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Memgraph started successfully${NC}"
            echo -e "${YELLOW}Waiting 5 seconds for Memgraph to initialize...${NC}"
            sleep 5
        else
            echo -e "${RED}✗ Failed to start Memgraph${NC}"
            echo -e "${YELLOW}Please start Memgraph manually:${NC}"
            echo -e "  docker-compose up -d memgraph"
            exit 1
        fi
    else
        echo -e "${YELLOW}docker-compose.yml not found${NC}"
        echo -e "${YELLOW}Please start Memgraph manually:${NC}"
        echo -e "  docker run -p 7687:7687 memgraph/memgraph"
        echo -e "  OR"
        echo -e "  cd to project root and run: docker-compose up -d memgraph"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Memgraph is running${NC}"
fi

# Verify Memgraph is accessible
echo -e "\n${BLUE}Verifying Memgraph connection...${NC}"
if python3 -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687'); driver.verify_connectivity(); driver.close()" 2>/dev/null; then
    echo -e "${GREEN}✓ Successfully connected to Memgraph${NC}"
else
    echo -e "${RED}✗ Cannot connect to Memgraph${NC}"
    echo -e "${YELLOW}Memgraph may still be starting up. Wait a few seconds and try again.${NC}"
    exit 1
fi

# Run the tests
echo -e "\n${BLUE}======================================${NC}"
echo -e "${BLUE}Running Memgraph Integration Tests${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the test suite
python3 "${SCRIPT_DIR}/memgraph_integration_test.py"
TEST_EXIT_CODE=$?

echo -e "\n${BLUE}======================================${NC}"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed successfully!${NC}"
    echo -e "${BLUE}======================================${NC}\n"
    echo -e "${GREEN}Next steps:${NC}"
    echo -e "  1. Load a real SCIP index into Memgraph:"
    echo -e "     ${BLUE}python scripts/load_to_databases.py index.scip --memgraph${NC}"
    echo -e "  2. Query the graph with Cypher:"
    echo -e "     ${BLUE}# Find all functions"
    echo -e "     MATCH (s:Symbol {kind: 'Function'}) RETURN s.name${NC}"
    echo -e "  3. Explore relationships:"
    echo -e "     ${BLUE}# Find what authenticate() calls"
    echo -e "     MATCH (s:Symbol {name: 'authenticate'})-[:REFERENCES]->(target)"
    echo -e "     RETURN target.name${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo -e "${BLUE}======================================${NC}\n"
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "  1. Check Memgraph logs:"
    echo -e "     ${BLUE}docker logs \$(docker ps | grep memgraph | awk '{print \$1}')${NC}"
    echo -e "  2. Verify Memgraph is running properly:"
    echo -e "     ${BLUE}docker ps | grep memgraph${NC}"
    echo -e "  3. Check connection settings in test file"
    echo -e "  4. Review test output above for specific errors"
fi

exit $TEST_EXIT_CODE
