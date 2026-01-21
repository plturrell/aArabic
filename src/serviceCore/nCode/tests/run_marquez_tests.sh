#!/bin/bash

# Marquez Integration Test Runner for nCode
# This script checks prerequisites and runs the Marquez test suite

set -e

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}nCode Marquez Integration Test Runner${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    echo -e "${YELLOW}Please install Python 3 to run these tests${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found: $(python3 --version)${NC}"

# Check if requests library is installed
if ! python3 -c "import requests" 2>/dev/null; then
    echo -e "${YELLOW}⚠ requests library not found${NC}"
    echo -e "${BLUE}Installing requests library...${NC}"
    pip3 install requests
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ requests library installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install requests library${NC}"
        echo -e "${YELLOW}Please install manually: pip3 install requests${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ requests library found${NC}"
fi

# Check if Marquez is running
echo -e "\n${BLUE}Checking Marquez status...${NC}"
if ! docker ps | grep -q marquez-api; then
    echo -e "${YELLOW}⚠ Marquez container not running${NC}"
    echo -e "${BLUE}Attempting to start Marquez...${NC}"
    
    # Check if we're in the project root
    if [ -f "docker-compose.yml" ]; then
        docker-compose up -d marquez marquez-db
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Marquez started successfully${NC}"
            echo -e "${YELLOW}Waiting 10 seconds for Marquez to initialize...${NC}"
            sleep 10
        else
            echo -e "${RED}✗ Failed to start Marquez${NC}"
            echo -e "${YELLOW}Please start Marquez manually:${NC}"
            echo -e "  docker-compose up -d marquez marquez-db"
            exit 1
        fi
    else
        echo -e "${YELLOW}docker-compose.yml not found${NC}"
        echo -e "${YELLOW}Please start Marquez manually:${NC}"
        echo -e "  docker run -p 5000:5000 -p 5001:5001 marquezproject/marquez"
        echo -e "  OR"
        echo -e "  cd to project root and run: docker-compose up -d marquez marquez-db"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Marquez is running${NC}"
fi

# Verify Marquez is accessible
echo -e "\n${BLUE}Verifying Marquez connection...${NC}"
if curl -s -f http://localhost:5000/api/v1/namespaces > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Successfully connected to Marquez${NC}"
else
    echo -e "${RED}✗ Cannot connect to Marquez API${NC}"
    echo -e "${YELLOW}Marquez may still be starting up. Wait a few seconds and try again.${NC}"
    echo -e "${YELLOW}You can check Marquez logs with:${NC}"
    echo -e "  docker logs \$(docker ps | grep marquez-api | awk '{print \$1}')"
    exit 1
fi

# Run the tests
echo -e "\n${BLUE}======================================${NC}"
echo -e "${BLUE}Running Marquez Integration Tests${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the test suite
python3 "${SCRIPT_DIR}/marquez_integration_test.py"
TEST_EXIT_CODE=$?

echo -e "\n${BLUE}======================================${NC}"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed successfully!${NC}"
    echo -e "${BLUE}======================================${NC}\n"
    echo -e "${GREEN}Next steps:${NC}"
    echo -e "  1. Track real indexing runs in Marquez:"
    echo -e "     ${BLUE}python scripts/load_to_databases.py index.scip --marquez${NC}"
    echo -e "  2. View lineage in Marquez UI:"
    echo -e "     ${BLUE}open http://localhost:3000${NC}"
    echo -e "  3. Query lineage via API:"
    echo -e "     ${BLUE}curl http://localhost:5000/api/v1/lineage?nodeId=dataset:ncode:scip-index${NC}"
    echo -e "  4. Explore datasets and jobs:"
    echo -e "     ${BLUE}curl http://localhost:5000/api/v1/namespaces/ncode/datasets${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo -e "${BLUE}======================================${NC}\n"
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "  1. Check Marquez logs:"
    echo -e "     ${BLUE}docker logs \$(docker ps | grep marquez-api | awk '{print \$1}')${NC}"
    echo -e "  2. Check Marquez database logs:"
    echo -e "     ${BLUE}docker logs \$(docker ps | grep marquez-db | awk '{print \$1}')${NC}"
    echo -e "  3. Verify Marquez is running properly:"
    echo -e "     ${BLUE}docker ps | grep marquez${NC}"
    echo -e "  4. Check API accessibility:"
    echo -e "     ${BLUE}curl http://localhost:5000/api/v1/namespaces${NC}"
    echo -e "  5. Review test output above for specific errors"
fi

exit $TEST_EXIT_CODE
