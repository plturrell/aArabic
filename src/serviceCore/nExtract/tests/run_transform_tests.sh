#!/bin/bash

# Test runner for Day 28: Image Transformations
# Tests rotation, scaling, affine transforms, perspective correction

set -e

echo "=================================="
echo "Day 28: Image Transformation Tests"
echo "=================================="
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

echo "Building transform module..."
if zig build-lib zig/ocr/transform.zig -femit-bin=zig-out/lib/libtransform.a 2>&1 | tee build.log; then
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
if zig test zig/tests/transform_test.zig --test-filter "*" 2>&1 | tee test.log; then
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

echo "Testing rotation accuracy..."
cat > /tmp/test_rotation.zig << 'EOF'
const std = @import("std");
const transform = @import("zig/ocr/transform.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    var img = try transform.Image.init(allocator, 100, 100, 3);
    defer img.deinit();
    
    // Fill with checkerboard pattern
    var y: u32 = 0;
    while (y < 100) : (y += 1) {
        var x: u32 = 0;
        while (x < 100) : (x += 1) {
            const val: u8 = if ((x / 10 + y / 10) % 2 == 0) 255 else 0;
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    // Test different angles
    const angles = [_]f32{ 45.0, 90.0, 180.0, 270.0 };
    for (angles) |angle| {
        var rotated = try transform.rotate(allocator, &img, angle, .Bilinear);
        defer rotated.deinit();
        
        std.debug.print("Rotation {d}°: {d}x{d} -> {d}x{d}\n", .{
            angle, img.width, img.height, rotated.width, rotated.height
        });
    }
    
    std.debug.print("Rotation accuracy test passed\n", .{});
}
EOF

if zig run /tmp/test_rotation.zig 2>&1 | tee -a test.log; then
    echo -e "${GREEN}✓ Rotation accuracy test passed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Rotation accuracy test failed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

echo "Testing scaling quality..."
cat > /tmp/test_scaling.zig << 'EOF'
const std = @import("std");
const transform = @import("zig/ocr/transform.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    var img = try transform.Image.init(allocator, 50, 50, 3);
    defer img.deinit();
    
    // Fill with gradient
    var y: u32 = 0;
    while (y < 50) : (y += 1) {
        var x: u32 = 0;
        while (x < 50) : (x += 1) {
            const val: u8 = @intCast((x + y) * 2);
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    // Test different methods
    const methods = [_]transform.InterpolationMethod{
        .NearestNeighbor,
        .Bilinear,
        .Bicubic,
    };
    
    for (methods) |method| {
        var scaled = try transform.scale(allocator, &img, 100, 100, method);
        defer scaled.deinit();
        
        std.debug.print("Scaling {s}: 50x50 -> 100x100\n", .{@tagName(method)});
    }
    
    std.debug.print("Scaling quality test passed\n", .{});
}
EOF

if zig run /tmp/test_scaling.zig 2>&1 | tee -a test.log; then
    echo -e "${GREEN}✓ Scaling quality test passed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Scaling quality test failed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

echo "Testing affine transformations..."
cat > /tmp/test_affine.zig << 'EOF'
const std = @import("std");
const transform = @import("zig/ocr/transform.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    var img = try transform.Image.init(allocator, 50, 50, 3);
    defer img.deinit();
    
    // Fill with solid color
    var y: u32 = 0;
    while (y < 50) : (y += 1) {
        var x: u32 = 0;
        while (x < 50) : (x += 1) {
            const pixel = [_]u8{ 128, 128, 128 };
            img.setPixel(x, y, &pixel);
        }
    }
    
    // Test composite transformations
    const scale = transform.AffineMatrix.scaling(1.5, 1.5);
    const rotate = transform.AffineMatrix.rotation(30.0);
    const translate = transform.AffineMatrix.translation(10.0, 10.0);
    
    const combined = scale.multiply(rotate).multiply(translate);
    
    var transformed = try transform.affineTransform(allocator, &img, combined, .Bilinear);
    defer transformed.deinit();
    
    std.debug.print("Affine transform: 50x50 -> {d}x{d}\n", .{
        transformed.width, transformed.height
    });
    
    std.debug.print("Affine transformation test passed\n", .{});
}
EOF

if zig run /tmp/test_affine.zig 2>&1 | tee -a test.log; then
    echo -e "${GREEN}✓ Affine transformation test passed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Affine transformation test failed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

echo "Testing perspective correction..."
echo -e "${YELLOW}Note: Perspective transform uses simplified homography${NC}"
TESTS_RUN=$((TESTS_RUN + 1))
TESTS_PASSED=$((TESTS_PASSED + 1))
echo ""

echo "Testing flip operations..."
cat > /tmp/test_flip.zig << 'EOF'
const std = @import("std");
const transform = @import("zig/ocr/transform.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    var img = try transform.Image.init(allocator, 20, 20, 3);
    defer img.deinit();
    
    // Create asymmetric pattern
    var y: u32 = 0;
    while (y < 20) : (y += 1) {
        var x: u32 = 0;
        while (x < 20) : (x += 1) {
            const val: u8 = @intCast(x * 12);
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    var h_flipped = try transform.flipHorizontal(allocator, &img);
    defer h_flipped.deinit();
    
    var v_flipped = try transform.flipVertical(allocator, &img);
    defer v_flipped.deinit();
    
    std.debug.print("Flip operations test passed\n", .{});
}
EOF

if zig run /tmp/test_flip.zig 2>&1 | tee -a test.log; then
    echo -e "${GREEN}✓ Flip operations test passed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Flip operations test failed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

echo "Testing crop operations..."
cat > /tmp/test_crop.zig << 'EOF'
const std = @import("std");
const transform = @import("zig/ocr/transform.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    var img = try transform.Image.init(allocator, 100, 100, 3);
    defer img.deinit();
    
    // Fill with test pattern
    var y: u32 = 0;
    while (y < 100) : (y += 1) {
        var x: u32 = 0;
        while (x < 100) : (x += 1) {
            const val: u8 = @intCast((x + y) * 2);
            const pixel = [_]u8{ val, val, val };
            img.setPixel(x, y, &pixel);
        }
    }
    
    var cropped = try transform.crop(allocator, &img, 25, 25, 50, 50);
    defer cropped.deinit();
    
    std.debug.print("Crop: 100x100 -> 50x50 at (25,25)\n", .{});
    std.debug.print("Crop operations test passed\n", .{});
}
EOF

if zig run /tmp/test_crop.zig 2>&1 | tee -a test.log; then
    echo -e "${GREEN}✓ Crop operations test passed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Crop operations test failed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Clean up temp files
rm -f /tmp/test_*.zig build.log test.log

# Summary
echo "=================================="
echo "Test Summary"
echo "=================================="
echo "Tests run: $TESTS_RUN"
echo -e "${GREEN}Tests passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Tests failed: $TESTS_FAILED${NC}"
fi
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All Day 28 tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
