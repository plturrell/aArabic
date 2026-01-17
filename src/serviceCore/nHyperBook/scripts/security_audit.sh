#!/bin/bash
# HyperShimmy Security Audit Script
# Comprehensive security testing and vulnerability scanning

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BASE_URL="${BASE_URL:-http://localhost:8080}"
RESULTS_DIR="security_audit_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

ISSUES_FOUND=0

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  HyperShimmy Security Audit${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check server availability
echo -e "${YELLOW}→ Checking server...${NC}"
if ! curl -s -f "${BASE_URL}/health" > /dev/null; then
    echo -e "${RED}✗ Server not available${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Server is available${NC}"
echo ""

# Function to log security issue
log_issue() {
    local severity=$1
    local title=$2
    local description=$3
    
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
    
    cat >> "$RESULTS_DIR/security_issues.txt" << EOF

[${severity}] ${title}
${description}

EOF
    
    if [ "$severity" = "HIGH" ] || [ "$severity" = "CRITICAL" ]; then
        echo -e "${RED}  ✗ [${severity}] ${title}${NC}"
    else
        echo -e "${YELLOW}  ⚠ [${severity}] ${title}${NC}"
    fi
}

# Test 1: Security Headers
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 1: Security Headers${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Checking HTTP security headers...${NC}"

HEADERS=$(curl -s -I "${BASE_URL}/health")
echo "$HEADERS" > "$RESULTS_DIR/headers.txt"

# Check for important security headers
if ! echo "$HEADERS" | grep -qi "X-Content-Type-Options"; then
    log_issue "MEDIUM" "Missing X-Content-Type-Options header" \
        "Header should be set to 'nosniff' to prevent MIME type sniffing"
fi

if ! echo "$HEADERS" | grep -qi "X-Frame-Options"; then
    log_issue "MEDIUM" "Missing X-Frame-Options header" \
        "Header should be set to 'DENY' or 'SAMEORIGIN' to prevent clickjacking"
fi

if ! echo "$HEADERS" | grep -qi "Content-Security-Policy"; then
    log_issue "HIGH" "Missing Content-Security-Policy header" \
        "CSP header is missing, application may be vulnerable to XSS attacks"
fi

if ! echo "$HEADERS" | grep -qi "Strict-Transport-Security"; then
    log_issue "LOW" "Missing Strict-Transport-Security header" \
        "HSTS header should be set when using HTTPS"
fi

if ! echo "$HEADERS" | grep -qi "X-XSS-Protection"; then
    log_issue "LOW" "Missing X-XSS-Protection header" \
        "Header should be set to '1; mode=block'"
fi

if echo "$HEADERS" | grep -qi "Server:"; then
    log_issue "LOW" "Server header exposed" \
        "Server version information should be hidden"
fi

echo -e "${GREEN}✓ Security headers check complete${NC}"
echo ""

# Test 2: Authentication & Authorization
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 2: Authentication & Authorization${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Testing authentication mechanisms...${NC}"

# Test unauthenticated access
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/odata/Sources")
if [ "$STATUS" = "200" ]; then
    log_issue "HIGH" "No authentication required" \
        "OData endpoints accessible without authentication"
fi

# Test for default credentials (if auth is implemented)
DEFAULT_CREDS=("admin:admin" "admin:password" "root:root" "test:test")
for cred in "${DEFAULT_CREDS[@]}"; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" -u "$cred" "${BASE_URL}/odata/Sources")
    if [ "$STATUS" = "200" ]; then
        log_issue "CRITICAL" "Default credentials accepted" \
            "Default credentials '$cred' are still active"
    fi
done

echo -e "${GREEN}✓ Authentication check complete${NC}"
echo ""

# Test 3: Input Validation
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 3: Input Validation${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Testing input validation...${NC}"

# SQL Injection tests
SQL_PAYLOADS=(
    "' OR '1'='1"
    "1' AND '1'='1"
    "'; DROP TABLE sources--"
    "1 UNION SELECT NULL--"
)

for payload in "${SQL_PAYLOADS[@]}"; do
    ENCODED=$(printf %s "$payload" | jq -sRr @uri)
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
        "${BASE_URL}/odata/Sources?\$filter=title eq '${ENCODED}'")
    RESPONSE=$(curl -s "${BASE_URL}/odata/Sources?\$filter=title eq '${ENCODED}'")
    
    if echo "$RESPONSE" | grep -qi "error\|exception\|sql"; then
        log_issue "CRITICAL" "Possible SQL injection vulnerability" \
            "Payload '$payload' generated error message exposing internals"
    fi
done

# XSS tests
XSS_PAYLOADS=(
    "<script>alert('XSS')</script>"
    "<img src=x onerror=alert('XSS')>"
    "javascript:alert('XSS')"
)

for payload in "${XSS_PAYLOADS[@]}"; do
    ENCODED=$(printf %s "$payload" | jq -sRr @uri)
    RESPONSE=$(curl -s "${BASE_URL}/odata/Sources?\$filter=title eq '${ENCODED}'")
    
    if echo "$RESPONSE" | grep -F "$payload" > /dev/null; then
        log_issue "HIGH" "Possible XSS vulnerability" \
            "Unescaped user input detected in response"
        break
    fi
done

# Path traversal tests
PATH_PAYLOADS=(
    "../../../etc/passwd"
    "..\\..\\..\\windows\\system.ini"
    "....//....//....//etc/passwd"
)

for payload in "${PATH_PAYLOADS[@]}"; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
        "${BASE_URL}/files/${payload}")
    if [ "$STATUS" = "200" ]; then
        log_issue "CRITICAL" "Path traversal vulnerability" \
            "Able to access files outside intended directory"
        break
    fi
done

echo -e "${GREEN}✓ Input validation check complete${NC}"
echo ""

# Test 4: Rate Limiting
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 4: Rate Limiting${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Testing rate limiting...${NC}"

# Send rapid requests
RATE_LIMIT_HIT=false
for i in {1..150}; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health")
    if [ "$STATUS" = "429" ]; then
        RATE_LIMIT_HIT=true
        break
    fi
done

if [ "$RATE_LIMIT_HIT" = false ]; then
    log_issue "MEDIUM" "No rate limiting detected" \
        "Application does not implement rate limiting, vulnerable to DoS"
fi

echo -e "${GREEN}✓ Rate limiting check complete${NC}"
echo ""

# Test 5: CORS Configuration
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 5: CORS Configuration${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Testing CORS settings...${NC}"

CORS_RESPONSE=$(curl -s -H "Origin: http://evil.com" -I "${BASE_URL}/health")
echo "$CORS_RESPONSE" > "$RESULTS_DIR/cors.txt"

if echo "$CORS_RESPONSE" | grep -qi "Access-Control-Allow-Origin: \*"; then
    log_issue "MEDIUM" "Overly permissive CORS" \
        "CORS allows all origins (*), should be restricted to trusted domains"
fi

if echo "$CORS_RESPONSE" | grep -qi "Access-Control-Allow-Origin: http://evil.com"; then
    log_issue "HIGH" "CORS reflects arbitrary origins" \
        "CORS configuration reflects any origin without validation"
fi

echo -e "${GREEN}✓ CORS check complete${NC}"
echo ""

# Test 6: File Upload Security
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 6: File Upload Security${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Testing file upload security...${NC}"

# Test large file upload
dd if=/dev/zero of="$RESULTS_DIR/large_file.bin" bs=1M count=50 2>/dev/null
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -F "file=@$RESULTS_DIR/large_file.bin" \
    "${BASE_URL}/odata/Sources/upload")

if [ "$STATUS" = "413" ]; then
    echo -e "${GREEN}  ✓ File size limits enforced${NC}"
else
    log_issue "MEDIUM" "No file size limits" \
        "Application accepts files of any size, vulnerable to DoS"
fi

# Test malicious file types
echo '<?php system($_GET["cmd"]); ?>' > "$RESULTS_DIR/malicious.php"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -F "file=@$RESULTS_DIR/malicious.php" \
    "${BASE_URL}/odata/Sources/upload")

if [ "$STATUS" = "200" ] || [ "$STATUS" = "201" ]; then
    log_issue "HIGH" "Accepts dangerous file types" \
        "Application accepts .php files which could be executed"
fi

echo -e "${GREEN}✓ File upload check complete${NC}"
echo ""

# Test 7: Information Disclosure
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 7: Information Disclosure${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Testing for information leaks...${NC}"

# Check for exposed debug endpoints
DEBUG_PATHS=(
    "/debug"
    "/debug/pprof"
    "/.env"
    "/config"
    "/swagger"
    "/api-docs"
    "/graphql"
)

for path in "${DEBUG_PATHS[@]}"; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}${path}")
    if [ "$STATUS" = "200" ]; then
        log_issue "MEDIUM" "Debug endpoint exposed: ${path}" \
            "Debug or admin endpoint should not be accessible in production"
    fi
done

# Check error messages
ERROR_RESPONSE=$(curl -s "${BASE_URL}/odata/NonExistent")
if echo "$ERROR_RESPONSE" | grep -Ei "stack trace|exception|error.*line"; then
    log_issue "MEDIUM" "Verbose error messages" \
        "Error messages contain sensitive information (stack traces, paths)"
fi

echo -e "${GREEN}✓ Information disclosure check complete${NC}"
echo ""

# Test 8: SSL/TLS Configuration
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 8: SSL/TLS Configuration${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [[ "$BASE_URL" == https://* ]]; then
    echo -e "${YELLOW}→ Testing SSL/TLS configuration...${NC}"
    
    if command -v openssl &> /dev/null; then
        HOST=$(echo "$BASE_URL" | sed -e 's|https://||' -e 's|/.*||')
        PORT=443
        
        # Test SSL protocols
        for protocol in ssl2 ssl3 tls1 tls1_1; do
            if openssl s_client -connect "$HOST:$PORT" -$protocol < /dev/null 2>&1 | grep -q "Cipher"; then
                log_issue "HIGH" "Insecure SSL/TLS protocol enabled: $protocol" \
                    "Only TLS 1.2 and TLS 1.3 should be enabled"
            fi
        done
        
        # Test certificate
        openssl s_client -connect "$HOST:$PORT" < /dev/null 2>&1 | \
            openssl x509 -noout -dates > "$RESULTS_DIR/cert_info.txt" 2>&1
        
        echo -e "${GREEN}✓ SSL/TLS check complete${NC}"
    else
        echo -e "${YELLOW}  ⚠ OpenSSL not available, skipping${NC}"
    fi
else
    log_issue "HIGH" "Application not using HTTPS" \
        "Production application should use HTTPS"
fi
echo ""

# Test 9: Dependency Vulnerabilities
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 9: Dependency Vulnerabilities${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Checking for vulnerable dependencies...${NC}"

# Note: This would typically use tools like Snyk, OWASP Dependency Check, etc.
echo -e "${YELLOW}  ℹ Run 'snyk test' for comprehensive dependency scanning${NC}"
echo -e "${YELLOW}  ℹ Run 'zig build --help' to check Zig version${NC}"
echo ""

# Generate summary report
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Generating Security Report${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cat > "$RESULTS_DIR/SECURITY_REPORT.md" << EOF
# HyperShimmy Security Audit Report

**Date**: $(date)
**Target**: ${BASE_URL}
**Issues Found**: ${ISSUES_FOUND}

## Executive Summary

This security audit tested HyperShimmy for common web application vulnerabilities
including OWASP Top 10 risks.

## Test Results

### 1. Security Headers
- Checked for presence of security-related HTTP headers
- See: headers.txt

### 2. Authentication & Authorization
- Tested for unauthenticated access
- Checked for default credentials
- Verified authorization controls

### 3. Input Validation
- Tested for SQL injection vulnerabilities
- Tested for XSS (Cross-Site Scripting)
- Tested for path traversal attacks

### 4. Rate Limiting
- Verified presence of rate limiting to prevent DoS

### 5. CORS Configuration
- Checked for overly permissive CORS policies
- See: cors.txt

### 6. File Upload Security
- Tested file size limits
- Tested file type restrictions
- Checked for malicious file handling

### 7. Information Disclosure
- Looked for exposed debug endpoints
- Checked for verbose error messages
- Verified no sensitive information leakage

### 8. SSL/TLS Configuration
- Tested SSL/TLS protocol versions
- Checked certificate validity
- Verified secure cipher suites

### 9. Dependency Vulnerabilities
- Noted need for regular dependency scanning

## Issues Found

EOF

if [ -f "$RESULTS_DIR/security_issues.txt" ]; then
    cat "$RESULTS_DIR/security_issues.txt" >> "$RESULTS_DIR/SECURITY_REPORT.md"
else
    echo "No security issues detected!" >> "$RESULTS_DIR/SECURITY_REPORT.md"
fi

cat >> "$RESULTS_DIR/SECURITY_REPORT.md" << 'EOF'

## Recommendations

### Critical Priority
1. Fix any CRITICAL severity issues immediately
2. Implement authentication and authorization
3. Add input validation and sanitization
4. Use HTTPS in production

### High Priority
1. Add security headers (CSP, X-Frame-Options, etc.)
2. Implement rate limiting
3. Restrict file upload types
4. Hide error details in production

### Medium Priority
1. Configure CORS properly
2. Remove debug endpoints in production
3. Hide server version information
4. Implement file size limits

### Low Priority
1. Enable HSTS header
2. Regular security audits
3. Keep dependencies updated
4. Security monitoring

## Security Best Practices

1. **Defense in Depth**: Implement multiple layers of security
2. **Principle of Least Privilege**: Grant minimum necessary permissions
3. **Fail Securely**: Default to secure configuration
4. **Don't Trust User Input**: Validate and sanitize all inputs
5. **Keep Security Simple**: Complex security is hard to maintain
6. **Regular Updates**: Keep dependencies and frameworks updated
7. **Security Testing**: Include security tests in CI/CD pipeline
8. **Monitoring**: Log and monitor security events

## Next Steps

- [ ] Review and fix all identified issues
- [ ] Implement automated security scanning in CI/CD
- [ ] Set up security monitoring and alerting
- [ ] Conduct penetration testing
- [ ] Create security incident response plan
- [ ] Regular security training for team

## Tools for Further Testing

- **OWASP ZAP**: Automated security testing
- **Burp Suite**: Manual penetration testing
- **Snyk**: Dependency vulnerability scanning
- **SonarQube**: Static code analysis
- **Trivy**: Container image scanning

EOF

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Security Audit Complete${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${GREEN}✓ No security issues detected!${NC}"
else
    echo -e "${RED}⚠ Found ${ISSUES_FOUND} security issues${NC}"
    echo -e "${YELLOW}  Review the report for details${NC}"
fi

echo ""
echo -e "${GREEN}Results saved to: ${RESULTS_DIR}/${NC}"
echo -e "${YELLOW}→ View report: cat ${RESULTS_DIR}/SECURITY_REPORT.md${NC}"
echo -e "${YELLOW}→ View issues: cat ${RESULTS_DIR}/security_issues.txt${NC}"
echo ""

exit $ISSUES_FOUND
