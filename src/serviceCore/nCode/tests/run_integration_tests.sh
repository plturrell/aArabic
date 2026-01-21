#!/bin/bash
# nCode Integration Testing Runner
# Day 14 - Integration Testing & Deployment

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}nCode Integration Testing${NC}"
echo -e "${BLUE}Day 14 - Integration Testing & Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

# Check required Python packages
python3 -c "import requests" 2>/dev/null || {
    echo -e "${YELLOW}⚠ Installing requests package...${NC}"
    pip3 install requests
}
echo -e "${GREEN}✓ requests package available${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠ Docker not found - some tests will be skipped${NC}"
else
    echo -e "${GREEN}✓ Docker found${NC}"
fi

# Check if services are running
echo ""
echo -e "${BLUE}Checking service availability...${NC}"

# Check nCode server
if curl -s http://localhost:18003/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ nCode server is running${NC}"
else
    echo -e "${YELLOW}⚠ nCode server not accessible - integration tests may fail${NC}"
    echo -e "${YELLOW}  Start with: cd src/serviceCore/nCode && docker-compose up -d${NC}"
fi

# Check Qdrant
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Qdrant is running${NC}"
else
    echo -e "${YELLOW}⚠ Qdrant not accessible${NC}"
fi

# Check Marquez
if curl -s http://localhost:5000/api/v1/namespaces > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Marquez is running${NC}"
else
    echo -e "${YELLOW}⚠ Marquez not accessible${NC}"
fi

# Run integration tests
echo ""
echo -e "${BLUE}Running integration tests...${NC}"
echo ""

cd "$(dirname "$0")/.."
python3 tests/integration_test_toolorchestra.py

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Integration tests completed successfully${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Integration tests failed${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $exit_code
