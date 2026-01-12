#!/bin/bash

echo "🔨 Building Zig + Mojo RAG Service"
echo "===================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Check for Zig
if ! command -v zig &> /dev/null; then
    echo -e "${RED}❌ Zig not found${NC}"
    echo "Install Zig: https://ziglang.org/download/"
    exit 1
fi

echo -e "${GREEN}✅ Zig found:${NC} $(zig version)"

# Check for Mojo
if ! command -v mojo &> /dev/null; then
    echo -e "${RED}❌ Mojo not found${NC}"
    echo "Install Mojo: https://docs.modular.com/mojo/manual/get-started/"
    exit 1
fi

echo -e "${GREEN}✅ Mojo found${NC}"
echo ""

# Build Zig HTTP library
echo -e "${BLUE}📦 Step 1: Building Zig HTTP library...${NC}"
zig build-lib zig_http.zig -dynamic -O ReleaseFast

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Zig HTTP library built: libzig_http.so${NC}"
else
    echo -e "${RED}❌ Zig HTTP build failed${NC}"
    exit 1
fi

echo ""

# Build Zig JSON library
echo -e "${BLUE}📦 Step 2: Building Zig JSON library...${NC}"
zig build-lib zig_json.zig -dynamic -O ReleaseFast

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Zig JSON library built: libzig_json.so${NC}"
else
    echo -e "${RED}❌ Zig JSON build failed${NC}"
    exit 1
fi

echo ""

# Build Zig Qdrant client
echo -e "${BLUE}📦 Step 3: Building Zig Qdrant client...${NC}"
zig build-lib zig_qdrant.zig -dynamic -O ReleaseFast

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Zig Qdrant client built: libzig_qdrant.so${NC}"
else
    echo -e "${RED}❌ Zig Qdrant build failed${NC}"
    exit 1
fi

echo ""

# Build Zig production HTTP server
echo -e "${BLUE}📦 Step 4: Building Zig production HTTP server...${NC}"
zig build-lib zig_http_production.zig -dynamic -O ReleaseFast

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Zig production HTTP built: libzig_http_production.so${NC}"
else
    echo -e "${RED}❌ Zig production HTTP build failed${NC}"
    exit 1
fi

echo ""

# Build Zig health & auth
echo -e "${BLUE}📦 Step 5: Building Zig health & auth...${NC}"
zig build-lib zig_health_auth.zig -dynamic -O ReleaseFast

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Zig health & auth built: libzig_health_auth.so${NC}"
else
    echo -e "${RED}❌ Zig health & auth build failed${NC}"
    exit 1
fi

echo ""

# Build load testing tool
echo -e "${BLUE}📦 Step 6: Building load test tool...${NC}"
zig build-exe load_test.zig -O ReleaseFast

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Load test tool built: load_test${NC}"
else
    echo -e "${RED}❌ Load test build failed${NC}"
    exit 1
fi

echo ""

# Build Mojo application
echo -e "${BLUE}📦 Step 7: Building Mojo application...${NC}"
mojo build main.mojo -o zig-mojo-rag

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Mojo application built: zig-mojo-rag${NC}"
else
    echo -e "${RED}❌ Mojo build failed${NC}"
    exit 1
fi

echo ""
echo "===================================="
echo -e "${GREEN}✅ Build complete!${NC}"
echo ""
echo "📦 Files created:"
echo "   Basic Libraries:"
echo "   • libzig_http.so              (Zig HTTP server)"
echo "   • libzig_json.so              (Zig JSON parser)"
echo "   • libzig_qdrant.so            (Zig Qdrant client)"
echo ""
echo "   Production Features (Phase 3):"
echo "   • libzig_http_production.so   (Multi-threaded HTTP)"
echo "   • libzig_health_auth.so       (Health & Auth)"
echo "   • load_test                   (Load testing tool)"
echo ""
echo "   Application:"
echo "   • zig-mojo-rag                (Mojo application)"
echo ""
echo "🚀 To run:"
echo "   ./zig-mojo-rag"
echo ""
echo "🌐 Server will start on:"
echo "   http://localhost:8009"
echo ""
echo "📝 Test with:"
echo "   curl http://localhost:8009/health"
echo ""
echo "🔥 Complete Production Stack (Phase 3):"
echo "   • Zig: HTTP, JSON, Qdrant, Auth, Metrics"
echo "   • Mojo: SIMD compute (10x faster)"
echo "   • Multi-threading, connection pooling"
echo "   • Health checks, rate limiting"
echo "   • Load testing included"
echo "   • Zero Python dependency!"
echo ""
echo "🧪 Run load test:"
echo "   ./load_test"
echo ""
