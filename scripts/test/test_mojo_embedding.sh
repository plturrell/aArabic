#!/bin/bash
# Test Mojo Embedding Service
# Usage: ./scripts/test_mojo_embedding.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BASE_URL="http://localhost:8007"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🧪 Testing Mojo Embedding Service${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Test 1: Health Check
echo -e "${BLUE}Test 1: Health Check${NC}"
echo -e "${YELLOW}GET ${BASE_URL}/health${NC}"
response=$(curl -s "${BASE_URL}/health")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Health check successful${NC}"
    echo "$response" | python3 -m json.tool
else
    echo -e "${RED}✗ Health check failed${NC}"
    exit 1
fi
echo ""

# Test 2: Single Embedding
echo -e "${BLUE}Test 2: Single Embedding${NC}"
echo -e "${YELLOW}POST ${BASE_URL}/embed/single${NC}"
response=$(curl -s -X POST "${BASE_URL}/embed/single" \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello world","model_type":"general"}')
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Single embedding successful${NC}"
    echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Model: {d[\"model_used\"]}, Dimensions: {d[\"dimensions\"]}, Time: {d[\"processing_time_ms\"]}ms')"
else
    echo -e "${RED}✗ Single embedding failed${NC}"
fi
echo ""

# Test 3: Batch Embedding (General)
echo -e "${BLUE}Test 3: Batch Embedding (General Model)${NC}"
echo -e "${YELLOW}POST ${BASE_URL}/embed/batch${NC}"
response=$(curl -s -X POST "${BASE_URL}/embed/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "texts": ["Hello world", "مرحبا بك", "This is a test"],
        "model_type": "general"
    }')
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Batch embedding successful${NC}"
    echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Count: {d[\"count\"]}, Dimensions: {d[\"dimensions\"]}, Time: {d[\"processing_time_ms\"]}ms, Model: {d[\"model_used\"]}')"
else
    echo -e "${RED}✗ Batch embedding failed${NC}"
fi
echo ""

# Test 4: Batch Embedding (Financial Model)
echo -e "${BLUE}Test 4: Batch Embedding (Financial Model)${NC}"
echo -e "${YELLOW}POST ${BASE_URL}/embed/batch${NC}"
response=$(curl -s -X POST "${BASE_URL}/embed/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "texts": ["Invoice amount 1000 SAR", "فاتورة بمبلغ 5000 ريال"],
        "model_type": "financial"
    }')
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Financial embedding successful${NC}"
    echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Count: {d[\"count\"]}, Dimensions: {d[\"dimensions\"]}, Time: {d[\"processing_time_ms\"]}ms, Model: {d[\"model_used\"]}')"
else
    echo -e "${RED}✗ Financial embedding failed${NC}"
fi
echo ""

# Test 5: Workflow Embedding
echo -e "${BLUE}Test 5: Workflow Embedding${NC}"
echo -e "${YELLOW}POST ${BASE_URL}/embed/workflow${NC}"
response=$(curl -s -X POST "${BASE_URL}/embed/workflow" \
    -H "Content-Type: application/json" \
    -d '{
        "workflow_text": "Process invoice and extract data",
        "workflow_metadata": {
            "name": "Invoice Processing",
            "description": "Automated invoice extraction workflow"
        }
    }')
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Workflow embedding successful${NC}"
    echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Dimensions: {d[\"dimensions\"]}, Model: {d[\"model\"]}')"
else
    echo -e "${RED}✗ Workflow embedding failed${NC}"
fi
echo ""

# Test 6: Invoice Embedding
echo -e "${BLUE}Test 6: Invoice Embedding${NC}"
echo -e "${YELLOW}POST ${BASE_URL}/embed/invoice${NC}"
response=$(curl -s -X POST "${BASE_URL}/embed/invoice" \
    -H "Content-Type: application/json" \
    -d '{
        "invoice_text": "Invoice #12345 from ACME Corp",
        "extracted_data": {
            "vendor_name": "ACME Corp",
            "total_amount": "1000",
            "currency": "SAR"
        }
    }')
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Invoice embedding successful${NC}"
    echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Dimensions: {d[\"dimensions\"]}, Model: {d[\"model\"]}')"
else
    echo -e "${RED}✗ Invoice embedding failed${NC}"
fi
echo ""

# Test 7: Document Embedding (Chunked)
echo -e "${BLUE}Test 7: Document Embedding (Chunked)${NC}"
echo -e "${YELLOW}POST ${BASE_URL}/embed/document${NC}"
response=$(curl -s -X POST "${BASE_URL}/embed/document" \
    -H "Content-Type: application/json" \
    -d '{
        "document_text": "This is a long document that will be split into chunks. It contains multiple sentences and paragraphs. The system will automatically chunk it based on the specified chunk size. This allows for efficient processing of large documents.",
        "chunk_size": 10
    }')
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Document embedding successful${NC}"
    echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Chunks: {d[\"chunks\"]}, Dimensions: {d[\"dimensions\"]}, Model: {d[\"model\"]}')"
else
    echo -e "${RED}✗ Document embedding failed${NC}"
fi
echo ""

# Test 8: List Models
echo -e "${BLUE}Test 8: List Available Models${NC}"
echo -e "${YELLOW}GET ${BASE_URL}/models${NC}"
response=$(curl -s "${BASE_URL}/models")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Models list retrieved${NC}"
    echo "$response" | python3 -m json.tool
else
    echo -e "${RED}✗ Failed to retrieve models${NC}"
fi
echo ""

# Test 9: Metrics
echo -e "${BLUE}Test 9: Service Metrics${NC}"
echo -e "${YELLOW}GET ${BASE_URL}/metrics${NC}"
response=$(curl -s "${BASE_URL}/metrics")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Metrics retrieved${NC}"
    echo "$response" | python3 -m json.tool
else
    echo -e "${RED}✗ Failed to retrieve metrics${NC}"
fi
echo ""

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ All tests completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}📚 API Documentation:${NC}"
echo -e "   ${BASE_URL}/docs"
echo ""
echo -e "${YELLOW}🎯 Next Steps:${NC}"
echo -e "   1. Phase 2: Add input validation"
echo -e "   2. Phase 3: Integrate real models with MAX Engine"
echo -e "   3. Phase 4: Implement SIMD optimizations"
echo -e "   4. Phase 5: Production deployment"
