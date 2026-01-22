#!/bin/bash

# Model Testing Script for nOpenaiServer
# Tests all 5 GGUF models with tiering configurations
# Last Updated: January 20, 2026

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_DIR="/Users/user/Documents/arabic_folder"
MODEL_DIR="${BASE_DIR}/vendor/layerModels"
SERVER_DIR="${BASE_DIR}/src/serviceCore/nOpenaiServer"
SERVER_BIN="${SERVER_DIR}/openai_http_server"
SERVER_URL="http://localhost:11434"
LOG_DIR="${SERVER_DIR}/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Test configuration
TEST_PROMPT="What is 2+2?"
TEST_MAX_TOKENS=10
TEST_TEMPERATURE=0.1

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  nOpenaiServer Model Testing Suite${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to check if server is running
check_server() {
    curl -s "${SERVER_URL}/health" > /dev/null 2>&1
    return $?
}

# Function to stop server
stop_server() {
    echo -e "${YELLOW}Stopping server...${NC}"
    pkill -f openai_http_server 2>/dev/null || true
    sleep 2
}

# Function to wait for server
wait_for_server() {
    local max_attempts=30
    local attempt=0
    
    echo -e "${YELLOW}Waiting for server to start...${NC}"
    while [ $attempt -lt $max_attempts ]; do
        if check_server; then
            echo -e "${GREEN}✓ Server is ready${NC}"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
        echo -n "."
    done
    
    echo -e "\n${RED}✗ Server failed to start${NC}"
    return 1
}

# Function to test a model
test_model() {
    local model_id="$1"
    local model_path="$2"
    local description="$3"
    local expected_behavior="$4"
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Testing: ${description}${NC}"
    echo -e "${BLUE}Model ID: ${model_id}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Stop any existing server
    stop_server
    
    # Start server with model
    local log_file="${LOG_DIR}/test_${model_id}.log"
    echo -e "${YELLOW}Starting server with ${model_id}...${NC}"
    echo -e "${YELLOW}Log file: ${log_file}${NC}"
    
    cd "${SERVER_DIR}"
    
    SHIMMY_DEBUG=1 \
    SHIMMY_MODEL_PATH="${MODEL_DIR}/${model_path}" \
    SHIMMY_MODEL_ID="${model_id}" \
    ./openai_http_server > "${log_file}" 2>&1 &
    
    local server_pid=$!
    echo -e "${YELLOW}Server PID: ${server_pid}${NC}"
    
    # Wait for server to start
    if ! wait_for_server; then
        echo -e "${RED}✗ Failed to start server for ${model_id}${NC}"
        echo -e "${YELLOW}Check log file: ${log_file}${NC}"
        return 1
    fi
    
    # Test 1: Health check
    echo -e "\n${YELLOW}Test 1: Health Check${NC}"
    if curl -s "${SERVER_URL}/health" | grep -q "ok"; then
        echo -e "${GREEN}✓ Health check passed${NC}"
    else
        echo -e "${RED}✗ Health check failed${NC}"
        stop_server
        return 1
    fi
    
    # Test 2: Models endpoint
    echo -e "\n${YELLOW}Test 2: Models Endpoint${NC}"
    local models_response=$(curl -s "${SERVER_URL}/v1/models")
    if echo "${models_response}" | grep -q "${model_id}"; then
        echo -e "${GREEN}✓ Model listed in /v1/models${NC}"
        echo "${models_response}" | python3 -m json.tool | head -20
    else
        echo -e "${RED}✗ Model not found in /v1/models${NC}"
    fi
    
    # Test 3: Simple completion
    echo -e "\n${YELLOW}Test 3: Simple Completion${NC}"
    echo -e "${YELLOW}Prompt: ${TEST_PROMPT}${NC}"
    echo -e "${YELLOW}Expected: ${expected_behavior}${NC}"
    
    local completion_response=$(curl -s -X POST "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model_id}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${TEST_PROMPT}\"}],
            \"max_tokens\": ${TEST_MAX_TOKENS},
            \"temperature\": ${TEST_TEMPERATURE}
        }")
    
    if echo "${completion_response}" | grep -q "choices"; then
        echo -e "${GREEN}✓ Completion request successful${NC}"
        echo -e "${YELLOW}Response:${NC}"
        echo "${completion_response}" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'choices' in data and len(data['choices']) > 0:
        content = data['choices'][0]['message']['content']
        print(f'  Content: {repr(content)[:100]}')
        print(f'  Length: {len(content)} chars')
        if content.strip() == '' or content == '\n' * len(content):
            print('  ⚠️  WARNING: Output is only whitespace/newlines')
        else:
            print('  ✓ Output contains text')
    else:
        print('  ✗ No choices in response')
except Exception as e:
    print(f'  ✗ Error parsing response: {e}')
"
    else
        echo -e "${RED}✗ Completion request failed${NC}"
        echo "${completion_response}"
    fi
    
    # Test 4: Memory usage (if available)
    echo -e "\n${YELLOW}Test 4: Memory Usage${NC}"
    local memory_response=$(curl -s "${SERVER_URL}/admin/memory" 2>/dev/null || echo "{}")
    if [ "${memory_response}" != "{}" ]; then
        echo "${memory_response}" | python3 -m json.tool 2>/dev/null || echo "${memory_response}"
    else
        echo -e "${YELLOW}Memory endpoint not available${NC}"
    fi
    
    # Test 5: Check logs for errors
    echo -e "\n${YELLOW}Test 5: Log Analysis${NC}"
    if grep -i "error\|panic\|fatal" "${log_file}" > /dev/null 2>&1; then
        echo -e "${RED}✗ Errors found in logs:${NC}"
        grep -i "error\|panic\|fatal" "${log_file}" | tail -5
    else
        echo -e "${GREEN}✓ No critical errors in logs${NC}"
    fi
    
    # Check for specific patterns
    if grep -q "Logits\[" "${log_file}"; then
        echo -e "${GREEN}✓ Generation process visible in logs${NC}"
    fi
    
    if grep -q "Top logits" "${log_file}"; then
        echo -e "${YELLOW}Top logits sample:${NC}"
        grep "Top logits" "${log_file}" | head -3
    fi
    
    # Stop server
    stop_server
    
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✓ Testing complete for ${model_id}${NC}"
    echo -e "${YELLOW}Full logs: ${log_file}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    return 0
}

# Main testing sequence
main() {
    echo -e "${YELLOW}Starting model testing sequence...${NC}"
    echo -e "${YELLOW}Base directory: ${BASE_DIR}${NC}"
    echo -e "${YELLOW}Model directory: ${MODEL_DIR}${NC}"
    echo ""
    
    # Check if server binary exists
    if [ ! -f "${SERVER_BIN}" ]; then
        echo -e "${RED}✗ Server binary not found: ${SERVER_BIN}${NC}"
        echo -e "${YELLOW}Please build the server first.${NC}"
        exit 1
    fi
    
    # Test each model
    local test_count=0
    local pass_count=0
    
    # Model 1: LFM2.5-Q4_0 (Debug priority)
    echo -e "\n${GREEN}═══ Test 1/5: LFM2.5-1.2B-Q4_0 ═══${NC}"
    if test_model \
        "lfm2.5-1.2b-q4_0" \
        "LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_0.gguf" \
        "LFM2.5 1.2B Q4_0 - Most compressed variant" \
        "Currently outputs only newlines - needs debugging"; then
        pass_count=$((pass_count + 1))
    fi
    test_count=$((test_count + 1))
    
    # Model 2: LFM2.5-Q4_K_M (Production)
    echo -e "\n${GREEN}═══ Test 2/5: LFM2.5-1.2B-Q4_K_M ═══${NC}"
    if test_model \
        "lfm2.5-1.2b-q4_k_m" \
        "LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf" \
        "LFM2.5 1.2B Q4_K_M - Balanced quality/size" \
        "Should produce valid text output"; then
        pass_count=$((pass_count + 1))
    fi
    test_count=$((test_count + 1))
    
    # Model 3: LFM2.5-F16 (Quality benchmark)
    echo -e "\n${GREEN}═══ Test 3/5: LFM2.5-1.2B-F16 ═══${NC}"
    if test_model \
        "lfm2.5-1.2b-f16" \
        "LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-F16.gguf" \
        "LFM2.5 1.2B F16 - Full precision" \
        "Highest quality output"; then
        pass_count=$((pass_count + 1))
    fi
    test_count=$((test_count + 1))
    
    # Ask user if they want to test large models
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Small model testing complete. Test large models?${NC}"
    echo -e "${YELLOW}  - DeepSeek-Coder-33B (19GB, requires tiering)${NC}"
    echo -e "${YELLOW}  - Llama-3.3-70B (40GB, requires aggressive tiering)${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    read -p "Test large models? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Model 4: DeepSeek-Coder-33B (Tiering validation)
        echo -e "\n${GREEN}═══ Test 4/5: DeepSeek-Coder-33B ═══${NC}"
        if test_model \
            "deepseek-coder-33b" \
            "deepseek-coder-33b-instruct-q4_k_m/deepseek-coder-33b-instruct-q4_k_m.gguf" \
            "DeepSeek Coder 33B - RAM+SSD tiering" \
            "Code generation with tiering"; then
            pass_count=$((pass_count + 1))
        fi
        test_count=$((test_count + 1))
        
        # Model 5: Llama-3.3-70B (Stress test)
        echo -e "\n${GREEN}═══ Test 5/5: Llama-3.3-70B ═══${NC}"
        echo -e "${RED}WARNING: This model requires significant resources!${NC}"
        echo -e "${YELLOW}  - Minimum: 10GB RAM, 50GB SSD${NC}"
        echo -e "${YELLOW}  - First load will be slow (mmap initialization)${NC}"
        read -p "Proceed with Llama-70B test? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if test_model \
                "llama-3.3-70b" \
                "Llama-3.3-70B-Instruct-Q4_K_M.gguf/Llama-3.3-70B-Instruct-Q4_K_M.gguf" \
                "Llama 3.3 70B - Aggressive SSD tiering" \
                "Advanced reasoning with tiering"; then
                pass_count=$((pass_count + 1))
            fi
            test_count=$((test_count + 1))
        fi
    fi
    
    # Summary
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Testing Summary${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Tests passed: ${pass_count}/${test_count}${NC}"
    echo -e "${YELLOW}Logs location: ${LOG_DIR}${NC}"
    echo ""
    
    if [ ${pass_count} -eq ${test_count} ]; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Some tests had issues. Check logs for details.${NC}"
        return 1
    fi
}

# Run main function
main

# Cleanup
stop_server
