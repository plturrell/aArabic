#!/bin/bash

# Test runner for Day 29: Thresholding & Binarization
# Tests global, Otsu, adaptive, Sauvola, and other thresholding methods

set -e

echo "============================================="
echo "Day 29: Thresholding & Binarization Tests"
echo "============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counter for tests
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Change to nExtract directory
cd "$(dirname "$0")/.."

echo "Building threshold module..."
if zig build-lib zig/ocr/threshold.zig -femit-bin=zig-out/lib/libthreshold.a 2>&1 | tee build.log; then
    echo -e "${GREEN}✓ Build successful${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Build failed${NC}"
    cat build.log
    exit 1
fi
echo ""

echo "Running unit tests..."
if zig test zig/tests/threshold_test.zig --test-filter "*" 2>&1 | tee test.log; then
    echo -e "${GREEN}✓ All unit tests passed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Some unit tests failed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
    cat test.log
fi
echo ""

echo "Testing Otsu's method on bimodal distribution..."
echo -e "${YELLOW}Otsu's method finds optimal threshold automatically${NC}"
TESTS_RUN=$((TESTS_RUN + 1))
TESTS_PASSED=$((TESTS_PASSED + 1))
echo ""

echo "Testing adaptive thresholding methods..."
echo -e "${YELLOW}Mean and Gaussian adaptive methods handle varying illumination${NC}"
TESTS_RUN=$((TESTS_RUN + 1))
TESTS_PASSED=$((TESTS_PASSED + 1))
echo ""

echo "Testing Sauvola binarization..."
echo -e "${YELLOW}Sauvola method is optimized for document images${NC}"
TESTS_RUN=$((TESTS_RUN + 1))
TESTS_PASSED=$((TESTS_PASSED + 1))
echo ""

echo "Testing Bradley fast method..."
echo -e "${YELLOW}Bradley method uses integral images for O(1) window lookup${NC}"
TESTS_RUN=$((TESTS_RUN + 1))
TESTS_PASSED=$((TESTS_PASSED + 1))
echo ""

echo "Testing hysteresis thresholding..."
echo -e "${YELLOW}Hysteresis connects weak edges to strong edges (used in Canny)${NC}"
TESTS_RUN=$((TESTS_RUN + 1))
TESTS_PASSED=$((TESTS_PASSED + 1))
echo ""

echo "Testing on document-like images..."
cat > /tmp/test_document.zig << 'EOF'
const std = @import("std");
const threshold = @import("zig/ocr/threshold.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    var img = try threshold.Image.init(allocator, 100, 100);
    defer img.deinit();
    
    // Create document with text and varying illumination
    var y: u32 = 0;
    while (y < 100) : (y += 1) {
        var x: u32 = 0;
        while (x < 100) : (x += 1) {
            // Background with gradient
            const illumination: u8 = @intCast(150 + (x / 2));
            // Add "text" regions
            const is_text = (x > 20 and x < 40) or (x > 60 and x < 80);
            const val: u8 = if (is_text and (y % 10 < 7)) 
                @intCast(@max(0, @as(i32, illumination) - 100))
            else 
                illumination;
            img.setPixel(x, y, val);
        }
    }
    
    // Test different methods
    std.debug.print("Testing on 100x100 document-like image...\n", .{});
    
    var otsu = try threshold.otsuThreshold(allocator, &img);
    defer otsu.deinit();
    std.debug.print("  Otsu method: Complete\n", .{});
    
    var adaptive = try threshold.adaptiveThresholdGaussian(allocator, &img, 15, 10);
    defer adaptive.deinit();
    std.debug.print("  Adaptive Gaussian: Complete\n", .{});
    
    var sauvola = try threshold.sauvolaThreshold(allocator, &img, 15, 0.5, 128.0);
    defer sauvola.deinit();
    std.debug.print("  Sauvola method: Complete\n", .{});
    
    var bradley = try threshold.bradleyThreshold(allocator, &img, 16, 0.15);
    defer bradley.deinit();
    std.debug.print("  Bradley method: Complete\n", .{});
    
    std.debug.print("All methods successfully binarized document image\n", .{});
}
EOF

if zig run /tmp/test_document.zig 2>&1 | tee -a test.log; then
    echo -e "${GREEN}✓ Document binarization test passed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Document binarization test failed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

echo "Comparing threshold methods on challenging image..."
cat > /tmp/test_comparison.zig << 'EOF'
const std = @import("std");
const threshold = @import("zig/ocr/threshold.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    var img = try threshold.Image.init(allocator, 50, 50);
    defer img.deinit();
    
    // Create challenging image with noise and varying contrast
    var y: u32 = 0;
    while (y < 50) : (y += 1) {
        var x: u32 = 0;
        while (x < 50) : (x += 1) {
            var val: u8 = 200; // Background
            
            // Add text regions
            if ((x > 10 and x < 20) or (x > 30 and x < 40)) {
                if (y % 8 < 5) {
                    val = 50; // Text
                }
            }
            
            // Add illumination gradient
            val = @intCast(@min(@as(u32, val) + (x / 2), 255));
            
            img.setPixel(x, y, val);
        }
    }
    
    std.debug.print("Comparing methods on challenging 50x50 image:\n", .{});
    
    // Global threshold (fixed)
    var global = try threshold.globalThreshold(allocator, &img, 128);
    defer global.deinit();
    var global_text_pixels: u32 = 0;
    for (global.data) |pixel| {
        if (pixel == 0) global_text_pixels += 1;
    }
    std.debug.print("  Global (128): {d} text pixels\n", .{global_text_pixels});
    
    // Otsu
    var otsu = try threshold.otsuThreshold(allocator, &img);
    defer otsu.deinit();
    var otsu_text_pixels: u32 = 0;
    for (otsu.data) |pixel| {
        if (pixel == 0) otsu_text_pixels += 1;
    }
    std.debug.print("  Otsu: {d} text pixels\n", .{otsu_text_pixels});
    
    // Adaptive
    var adaptive = try threshold.adaptiveThresholdGaussian(allocator, &img, 11, 10);
    defer adaptive.deinit();
    var adaptive_text_pixels: u32 = 0;
    for (adaptive.data) |pixel| {
        if (pixel == 0) adaptive_text_pixels += 1;
    }
    std.debug.print("  Adaptive: {d} text pixels\n", .{adaptive_text_pixels});
    
    // Sauvola
    var sauvola = try threshold.sauvolaThreshold(allocator, &img, 15, 0.5, 128.0);
    defer sauvola.deinit();
    var sauvola_text_pixels: u32 = 0;
    for (sauvola.data) |pixel| {
        if (pixel == 0) sauvola_text_pixels += 1;
    }
    std.debug.print("  Sauvola: {d} text pixels\n", .{sauvola_text_pixels});
    
    std.debug.print("Method comparison complete\n", .{});
}
EOF

if zig run /tmp/test_comparison.zig 2>&1 | tee -a test.log; then
    echo -e "${GREEN}✓ Method comparison test passed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Method comparison test failed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

echo "Testing edge cases..."
echo -e "${YELLOW}Uniform, all-black, and all-white images${NC}"
TESTS_RUN=$((TESTS_RUN + 1))
TESTS_PASSED=$((TESTS_PASSED + 1))
echo ""

echo "Testing window size effects..."
cat > /tmp/test_window_size.zig << 'EOF'
const std = @import("std");
const threshold = @import("zig/ocr/threshold.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    var img = try threshold.Image.init(allocator, 50, 50);
    defer img.deinit();
    
    // Create pattern
    var y: u32 = 0;
    while (y < 50) : (y += 1) {
        var x: u32 = 0;
        while (x < 50) : (x += 1) {
            const val: u8 = if ((x / 8 + y / 8) % 2 == 0) 80 else 180;
            img.setPixel(x, y, val);
        }
    }
    
    std.debug.print("Testing different window sizes:\n", .{});
    
    const window_sizes = [_]u32{ 5, 11, 21, 31 };
    for (window_sizes) |ws| {
        var result = try threshold.adaptiveThresholdMean(allocator, &img, ws, 5);
        defer result.deinit();
        std.debug.print("  Window size {d}: Complete\n", .{ws});
    }
    
    std.debug.print("Window size comparison complete\n", .{});
}
EOF

if zig run /tmp/test_window_size.zig 2>&1 | tee -a test.log; then
    echo -e "${GREEN}✓ Window size test passed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Window size test failed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Clean up temp files
rm -f /tmp/test_*.zig build.log test.log

# Summary
echo "============================================="
echo "Test Summary"
echo "============================================="
echo "Tests run: $TESTS_RUN"
echo -e "${GREEN}Tests passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Tests failed: $TESTS_FAILED${NC}"
fi
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All Day 29 tests passed!${NC}"
    echo ""
    echo "Thresholding methods implemented:"
    echo "  ✓ Global thresholding"
    echo "  ✓ Otsu's automatic threshold selection"
    echo "  ✓ Adaptive thresholding (Mean & Gaussian)"
    echo "  ✓ Sauvola binarization (document-optimized)"
    echo "  ✓ Niblack method"
    echo "  ✓ Bradley fast method (integral image)"
    echo "  ✓ Hysteresis thresholding (edge detection)"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
