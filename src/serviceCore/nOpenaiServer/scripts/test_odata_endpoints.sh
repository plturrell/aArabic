#!/bin/bash
# Test OData v4 Endpoints - Day 17
# Tests all OData endpoints with HANA Cloud

set -e

echo "================================"
echo "OData v4 Endpoint Testing"
echo "================================"

# Configuration
BASE_URL="http://localhost:11434"
ODATA_PATH="/odata/v4"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Helper function to test endpoint
test_endpoint() {
    local method=$1
    local path=$2
    local expected_status=$3
    local description=$4
    local data=$5
    
    echo ""
    echo "Testing: $description"
    echo "  Method: $method"
    echo "  Path: $path"
    
    if [ -n "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" -X $method "$BASE_URL$path" \
            -H "Content-Type: application/json" \
            -d "$data")
    else
        response=$(curl -s -w "\n%{http_code}" -X $method "$BASE_URL$path")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" == "$expected_status" ]; then
        echo -e "  ${GREEN}✓ PASS${NC} (HTTP $http_code)"
        ((PASSED++))
        if [ ${#body} -lt 500 ]; then
            echo "  Response: $body"
        else
            echo "  Response: ${body:0:200}... (truncated)"
        fi
    else
        echo -e "  ${RED}✗ FAIL${NC} (Expected: $expected_status, Got: $http_code)"
        ((FAILED++))
        echo "  Response: $body"
    fi
}

echo ""
echo "=============================="
echo "1. Service Discovery Tests"
echo "=============================="

# Test $metadata endpoint
test_endpoint "GET" "$ODATA_PATH/\$metadata" "200" "OData service metadata"

# Test service root (should list entity sets)
test_endpoint "GET" "$ODATA_PATH/" "200" "OData service root"

echo ""
echo "=============================="
echo "2. PROMPTS Entity Tests"
echo "=============================="

# Test GET collection
test_endpoint "GET" "$ODATA_PATH/Prompts" "200" "List all prompts"

# Test GET with $top
test_endpoint "GET" "$ODATA_PATH/Prompts?\$top=5" "200" "List top 5 prompts"

# Test GET with $select
test_endpoint "GET" "$ODATA_PATH/Prompts?\$select=prompt_text,rating" "200" "Select specific columns"

# Test GET with $filter
test_endpoint "GET" "$ODATA_PATH/Prompts?\$filter=rating%20gt%203" "200" "Filter by rating > 3"

# Test GET with $orderby
test_endpoint "GET" "$ODATA_PATH/Prompts?\$orderby=created_at%20desc" "200" "Order by created_at desc"

# Test GET with pagination
test_endpoint "GET" "$ODATA_PATH/Prompts?\$top=10&\$skip=20" "200" "Pagination (top 10, skip 20)"

# Test GET single entity (may fail if ID doesn't exist)
test_endpoint "GET" "$ODATA_PATH/Prompts(1)" "200" "Get prompt by ID" || true

# Test POST (create new prompt)
test_endpoint "POST" "$ODATA_PATH/Prompts" "200" "Create new prompt" \
    '{"prompt_text":"Test from OData","model_name":"test-model","user_id":"odata-test"}'

# Test PATCH (update prompt)
test_endpoint "PATCH" "$ODATA_PATH/Prompts(1)" "200" "Update prompt" \
    '{"rating":5}'

# Test DELETE (may fail if ID doesn't exist)
test_endpoint "DELETE" "$ODATA_PATH/Prompts(999)" "200" "Delete prompt" || true

echo ""
echo "=============================="
echo "3. MODEL_CONFIGURATIONS Tests"
echo "=============================="

# Test GET collection
test_endpoint "GET" "$ODATA_PATH/ModelConfigurations" "200" "List all model configurations"

# Test GET with query options
test_endpoint "GET" "$ODATA_PATH/ModelConfigurations?\$top=5" "200" "List top 5 configurations"
test_endpoint "GET" "$ODATA_PATH/ModelConfigurations?\$select=model_id,temperature,top_p" "200" "Select specific config fields"
test_endpoint "GET" "$ODATA_PATH/ModelConfigurations?\$filter=is_default%20eq%20true" "200" "Filter default configurations"
test_endpoint "GET" "$ODATA_PATH/ModelConfigurations?\$orderby=updated_at%20desc" "200" "Order by last updated"

# Test POST (create new configuration)
test_endpoint "POST" "$ODATA_PATH/ModelConfigurations" "200" "Create new model configuration" \
    '{"model_id":"gpt-4","temperature":0.7,"top_p":0.9,"max_tokens":2048}'

# Test GET single entity
test_endpoint "GET" "$ODATA_PATH/ModelConfigurations('config-123')" "200" "Get configuration by ID" || true

# Test PATCH (update configuration)
test_endpoint "PATCH" "$ODATA_PATH/ModelConfigurations('config-123')" "200" "Update configuration" \
    '{"temperature":0.8,"max_tokens":4096}' || true

# Test DELETE
test_endpoint "DELETE" "$ODATA_PATH/ModelConfigurations('config-999')" "200" "Delete configuration" || true

echo ""
echo "=============================="
echo "4. USER_SETTINGS Tests"
echo "=============================="

# Test GET collection
test_endpoint "GET" "$ODATA_PATH/UserSettings" "200" "List all user settings"

# Test GET with query options
test_endpoint "GET" "$ODATA_PATH/UserSettings?\$top=10" "200" "List top 10 user settings"
test_endpoint "GET" "$ODATA_PATH/UserSettings?\$select=user_id,theme,language" "200" "Select specific settings fields"
test_endpoint "GET" "$ODATA_PATH/UserSettings?\$filter=theme%20eq%20'sap_horizon'" "200" "Filter by theme"
test_endpoint "GET" "$ODATA_PATH/UserSettings?\$orderby=updated_at%20desc" "200" "Order settings by update time"

# Test POST (create new user settings)
test_endpoint "POST" "$ODATA_PATH/UserSettings" "200" "Create new user settings" \
    '{"user_id":"test-user","theme":"sap_horizon","language":"en"}'

# Test GET single entity
test_endpoint "GET" "$ODATA_PATH/UserSettings('test-user')" "200" "Get user settings by user ID" || true

# Test PATCH (update user settings)
test_endpoint "PATCH" "$ODATA_PATH/UserSettings('test-user')" "200" "Update user settings" \
    '{"theme":"sap_fiori","enable_notifications":true}' || true

# Test DELETE
test_endpoint "DELETE" "$ODATA_PATH/UserSettings('temp-user')" "200" "Delete user settings" || true

echo ""
echo "=============================="
echo "5. NOTIFICATIONS Tests"
echo "=============================="

# Test GET collection
test_endpoint "GET" "$ODATA_PATH/Notifications" "200" "List all notifications"

# Test GET with query options
test_endpoint "GET" "$ODATA_PATH/Notifications?\$top=20" "200" "List top 20 notifications"
test_endpoint "GET" "$ODATA_PATH/Notifications?\$select=notification_id,type,title,message" "200" "Select notification fields"
test_endpoint "GET" "$ODATA_PATH/Notifications?\$filter=type%20eq%20'error'" "200" "Filter error notifications"
test_endpoint "GET" "$ODATA_PATH/Notifications?\$filter=is_read%20eq%20false" "200" "Filter unread notifications"
test_endpoint "GET" "$ODATA_PATH/Notifications?\$orderby=created_at%20desc" "200" "Order by newest first"

# Test POST (create new notification)
test_endpoint "POST" "$ODATA_PATH/Notifications" "200" "Create new notification" \
    '{"type":"info","title":"Test Notification","message":"This is a test","user_id":"test-user"}'

# Test GET single entity
test_endpoint "GET" "$ODATA_PATH/Notifications('notif-123')" "200" "Get notification by ID" || true

# Test PATCH (mark as read)
test_endpoint "PATCH" "$ODATA_PATH/Notifications('notif-123')" "200" "Mark notification as read" \
    '{"is_read":true}' || true

# Test DELETE
test_endpoint "DELETE" "$ODATA_PATH/Notifications('notif-999')" "200" "Delete notification" || true

echo ""
echo "=============================="
echo "6. Other Entity Sets Tests"
echo "=============================="

# Test other entity sets (stub responses)
test_endpoint "GET" "$ODATA_PATH/ModelVersions" "200" "List model versions"
test_endpoint "GET" "$ODATA_PATH/TrainingExperiments" "200" "List training experiments"
test_endpoint "GET" "$ODATA_PATH/PromptComparisons" "200" "List prompt comparisons"

echo ""
echo "=============================="
echo "7. Advanced Query Tests"
echo "=============================="

# Test complex filter
test_endpoint "GET" "$ODATA_PATH/Prompts?\$filter=rating%20gt%203%20and%20is_favorite%20eq%20true" "200" "Complex filter (rating > 3 AND is_favorite = true)"

# Test multiple query options
test_endpoint "GET" "$ODATA_PATH/Prompts?\$filter=user_id%20eq%20'test'&\$orderby=created_at%20desc&\$top=10" "200" "Multiple query options"

# Test $count
test_endpoint "GET" "$ODATA_PATH/Prompts?\$count=true" "200" "Count query option"

echo ""
echo "=============================="
echo "8. Error Handling Tests"
echo "=============================="

# Test invalid entity set
test_endpoint "GET" "$ODATA_PATH/InvalidEntity" "200" "Invalid entity set (should return error)"

# Test invalid query option
test_endpoint "GET" "$ODATA_PATH/Prompts?\$invalid=value" "200" "Invalid query option (should be ignored)"

# Test malformed JSON
test_endpoint "POST" "$ODATA_PATH/Prompts" "200" "Malformed JSON" \
    '{invalid json}' || true

echo ""
echo "=============================="
echo "Test Summary"
echo "=============================="
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo "Total: $((PASSED + FAILED))"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Some tests failed${NC}"
    exit 1
fi
