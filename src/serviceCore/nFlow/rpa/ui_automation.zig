//! UI Automation for nWorkflow RPA
//! Provides window and screen manipulation capabilities

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Window information
pub const WindowInfo = struct {
    handle: u64,
    title: []const u8,
    class_name: []const u8,
    position: Position,
    size: Size,
    visible: bool = true,
    focused: bool = false,
    process_id: u32 = 0,

    pub const Position = struct {
        x: i32,
        y: i32,

        pub fn origin() Position {
            return Position{ .x = 0, .y = 0 };
        }
    };

    pub const Size = struct {
        width: u32,
        height: u32,

        pub fn zero() Size {
            return Size{ .width = 0, .height = 0 };
        }
    };

    pub fn getArea(self: *const WindowInfo) u64 {
        return @as(u64, self.size.width) * @as(u64, self.size.height);
    }

    pub fn contains(self: *const WindowInfo, x: i32, y: i32) bool {
        return x >= self.position.x and
            x < self.position.x + @as(i32, @intCast(self.size.width)) and
            y >= self.position.y and
            y < self.position.y + @as(i32, @intCast(self.size.height));
    }
};

/// Screen region for operations
pub const ScreenRegion = struct {
    x: i32,
    y: i32,
    width: u32,
    height: u32,

    pub fn fullScreen() ScreenRegion {
        return ScreenRegion{ .x = 0, .y = 0, .width = 1920, .height = 1080 };
    }

    pub fn fromWindow(window: *const WindowInfo) ScreenRegion {
        return ScreenRegion{
            .x = window.position.x,
            .y = window.position.y,
            .width = window.size.width,
            .height = window.size.height,
        };
    }

    pub fn contains(self: *const ScreenRegion, x: i32, y: i32) bool {
        return x >= self.x and x < self.x + @as(i32, @intCast(self.width)) and
            y >= self.y and y < self.y + @as(i32, @intCast(self.height));
    }

    pub fn getCenter(self: *const ScreenRegion) struct { x: i32, y: i32 } {
        return .{
            .x = self.x + @as(i32, @intCast(self.width / 2)),
            .y = self.y + @as(i32, @intCast(self.height / 2)),
        };
    }
};

/// UI Element properties
pub const ElementProperty = struct {
    name: []const u8,
    value: []const u8,
};

/// UI Element representation
pub const UIElement = struct {
    id: []const u8,
    element_type: ElementType,
    name: ?[]const u8 = null,
    class_name: ?[]const u8 = null,
    automation_id: ?[]const u8 = null,
    bounds: ScreenRegion,
    enabled: bool = true,
    visible: bool = true,
    focusable: bool = false,
    value: ?[]const u8 = null,
    properties: std.ArrayList(ElementProperty),
    children: std.ArrayList(*UIElement),
    parent: ?*UIElement = null,
    allocator: Allocator,

    pub const ElementType = enum {
        WINDOW,
        BUTTON,
        TEXT_BOX,
        LABEL,
        CHECKBOX,
        RADIO_BUTTON,
        COMBO_BOX,
        LIST_BOX,
        LIST_ITEM,
        MENU,
        MENU_ITEM,
        TAB,
        TAB_ITEM,
        TREE,
        TREE_ITEM,
        TABLE,
        TABLE_ROW,
        TABLE_CELL,
        IMAGE,
        LINK,
        PANE,
        GROUP,
        CUSTOM,

        pub fn toString(self: ElementType) []const u8 {
            return @tagName(self);
        }
    };

    pub fn init(allocator: Allocator, id: []const u8, element_type: ElementType, bounds: ScreenRegion) !UIElement {
        return UIElement{
            .id = try allocator.dupe(u8, id),
            .element_type = element_type,
            .bounds = bounds,
            .properties = std.ArrayList(ElementProperty){},
            .children = std.ArrayList(*UIElement){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *UIElement) void {
        self.properties.deinit(self.allocator);
        for (self.children.items) |child| {
            child.deinit();
            self.allocator.destroy(child);
        }
        self.children.deinit(self.allocator);
        self.allocator.free(self.id);
    }

    pub fn addProperty(self: *UIElement, name: []const u8, value: []const u8) !void {
        try self.properties.append(self.allocator, ElementProperty{ .name = name, .value = value });
    }

    pub fn getProperty(self: *const UIElement, name: []const u8) ?[]const u8 {
        for (self.properties.items) |prop| {
            if (std.mem.eql(u8, prop.name, name)) return prop.value;
        }
        return null;
    }

    pub fn addChild(self: *UIElement, child: *UIElement) !void {
        child.parent = self;
        try self.children.append(self.allocator, child);
    }

    pub fn getChildCount(self: *const UIElement) usize {
        return self.children.items.len;
    }
};

/// UI Automation engine for finding and interacting with elements
pub const UIAutomation = struct {
    root_element: ?*UIElement = null,
    timeout_ms: u32 = 5000,
    poll_interval_ms: u32 = 100,
    allocator: Allocator,

    pub fn init(allocator: Allocator) UIAutomation {
        return UIAutomation{ .allocator = allocator };
    }

    pub fn deinit(self: *UIAutomation) void {
        if (self.root_element) |root| {
            root.deinit();
            self.allocator.destroy(root);
        }
    }

    pub fn findElementById(self: *UIAutomation, id: []const u8) ?*UIElement {
        if (self.root_element) |root| {
            return self.findInTree(root, id);
        }
        return null;
    }

    fn findInTree(self: *UIAutomation, element: *UIElement, id: []const u8) ?*UIElement {
        if (std.mem.eql(u8, element.id, id)) return element;
        for (element.children.items) |child| {
            if (self.findInTree(child, id)) |found| return found;
        }
        return null;
    }

    pub fn findElementsByType(self: *UIAutomation, element_type: UIElement.ElementType) !std.ArrayList(*UIElement) {
        var results = std.ArrayList(*UIElement){};
        if (self.root_element) |root| {
            try self.collectByType(root, element_type, &results);
        }
        return results;
    }

    fn collectByType(self: *UIAutomation, element: *UIElement, target_type: UIElement.ElementType, results: *std.ArrayList(*UIElement)) !void {
        if (element.element_type == target_type) {
            try results.append(self.allocator, element);
        }
        for (element.children.items) |child| {
            try self.collectByType(child, target_type, results);
        }
    }

    pub fn getElementAtPoint(self: *UIAutomation, x: i32, y: i32) ?*UIElement {
        if (self.root_element) |root| {
            return self.findAtPoint(root, x, y);
        }
        return null;
    }

    fn findAtPoint(self: *UIAutomation, element: *UIElement, x: i32, y: i32) ?*UIElement {
        if (!element.bounds.contains(x, y)) return null;
        for (element.children.items) |child| {
            if (self.findAtPoint(child, x, y)) |found| return found;
        }
        return element;
    }
};

/// Window manager for manipulating windows
pub const WindowManager = struct {
    windows: std.ArrayList(WindowInfo),
    allocator: Allocator,

    pub fn init(allocator: Allocator) WindowManager {
        return WindowManager{
            .windows = std.ArrayList(WindowInfo){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *WindowManager) void {
        self.windows.deinit(self.allocator);
    }

    pub fn refreshWindows(self: *WindowManager) !void {
        self.windows.clearRetainingCapacity();
        // Simulated - would call native APIs
        try self.windows.append(self.allocator, WindowInfo{
            .handle = 1,
            .title = "Desktop",
            .class_name = "DesktopWindow",
            .position = WindowInfo.Position.origin(),
            .size = WindowInfo.Size{ .width = 1920, .height = 1080 },
        });
    }

    pub fn findWindowByTitle(self: *const WindowManager, title: []const u8) ?*const WindowInfo {
        for (self.windows.items) |*window| {
            if (std.mem.indexOf(u8, window.title, title) != null) return window;
        }
        return null;
    }

    pub fn getActiveWindow(self: *const WindowManager) ?*const WindowInfo {
        for (self.windows.items) |*window| {
            if (window.focused) return window;
        }
        return null;
    }

    pub fn bringToFront(self: *WindowManager, handle: u64) bool {
        for (self.windows.items) |*window| {
            window.focused = (window.handle == handle);
        }
        return true;
    }

    pub fn minimizeWindow(self: *WindowManager, handle: u64) bool {
        _ = self;
        _ = handle;
        return true;
    }

    pub fn maximizeWindow(self: *WindowManager, handle: u64) bool {
        _ = self;
        _ = handle;
        return true;
    }

    pub fn closeWindow(self: *WindowManager, handle: u64) bool {
        var index: ?usize = null;
        for (self.windows.items, 0..) |window, i| {
            if (window.handle == handle) {
                index = i;
                break;
            }
        }
        if (index) |i| {
            _ = self.windows.orderedRemove(i);
            return true;
        }
        return false;
    }

    pub fn moveWindow(self: *WindowManager, handle: u64, x: i32, y: i32) bool {
        for (self.windows.items) |*window| {
            if (window.handle == handle) {
                window.position = WindowInfo.Position{ .x = x, .y = y };
                return true;
            }
        }
        return false;
    }

    pub fn resizeWindow(self: *WindowManager, handle: u64, width: u32, height: u32) bool {
        for (self.windows.items) |*window| {
            if (window.handle == handle) {
                window.size = WindowInfo.Size{ .width = width, .height = height };
                return true;
            }
        }
        return false;
    }
};

/// Screen capture utilities
pub const ScreenCapture = struct {
    output_dir: []const u8,
    format: ImageFormat = .PNG,
    quality: u8 = 90,
    allocator: Allocator,

    pub const ImageFormat = enum {
        PNG,
        JPEG,
        BMP,

        pub fn toString(self: ImageFormat) []const u8 {
            return @tagName(self);
        }

        pub fn extension(self: ImageFormat) []const u8 {
            return switch (self) {
                .PNG => ".png",
                .JPEG => ".jpg",
                .BMP => ".bmp",
            };
        }
    };

    pub fn init(allocator: Allocator, output_dir: []const u8) ScreenCapture {
        return ScreenCapture{
            .output_dir = output_dir,
            .allocator = allocator,
        };
    }

    pub fn captureScreen(self: *const ScreenCapture) ![]const u8 {
        return self.captureRegion(ScreenRegion.fullScreen());
    }

    pub fn captureRegion(self: *const ScreenCapture, region: ScreenRegion) ![]const u8 {
        _ = region;
        const timestamp = std.time.timestamp();
        return std.fmt.allocPrint(self.allocator, "{s}/screenshot_{d}{s}", .{
            self.output_dir,
            timestamp,
            self.format.extension(),
        });
    }

    pub fn captureWindow(self: *const ScreenCapture, window: *const WindowInfo) ![]const u8 {
        return self.captureRegion(ScreenRegion.fromWindow(window));
    }

    pub fn captureElement(self: *const ScreenCapture, element: *const UIElement) ![]const u8 {
        return self.captureRegion(element.bounds);
    }
};

// Tests
test "WindowInfo creation" {
    const window = WindowInfo{
        .handle = 1,
        .title = "Test Window",
        .class_name = "TestClass",
        .position = WindowInfo.Position{ .x = 100, .y = 100 },
        .size = WindowInfo.Size{ .width = 800, .height = 600 },
    };
    try std.testing.expectEqual(@as(u64, 480000), window.getArea());
    try std.testing.expect(window.contains(200, 200));
    try std.testing.expect(!window.contains(50, 50));
}

test "ScreenRegion operations" {
    const region = ScreenRegion{ .x = 0, .y = 0, .width = 100, .height = 100 };
    try std.testing.expect(region.contains(50, 50));
    const center = region.getCenter();
    try std.testing.expectEqual(@as(i32, 50), center.x);
    try std.testing.expectEqual(@as(i32, 50), center.y);
}

test "UIElement creation" {
    const allocator = std.testing.allocator;
    var element = try UIElement.init(allocator, "btn-1", .BUTTON, ScreenRegion{ .x = 0, .y = 0, .width = 100, .height = 30 });
    defer element.deinit();
    try std.testing.expectEqualStrings("btn-1", element.id);
    try std.testing.expectEqual(UIElement.ElementType.BUTTON, element.element_type);
}

test "UIAutomation initialization" {
    const allocator = std.testing.allocator;
    var automation = UIAutomation.init(allocator);
    defer automation.deinit();
    try std.testing.expectEqual(@as(?*UIElement, null), automation.root_element);
}

test "WindowManager operations" {
    const allocator = std.testing.allocator;
    var manager = WindowManager.init(allocator);
    defer manager.deinit();
    try manager.refreshWindows();
    try std.testing.expectEqual(@as(usize, 1), manager.windows.items.len);
}

test "ScreenCapture initialization" {
    const allocator = std.testing.allocator;
    const capture = ScreenCapture.init(allocator, "/tmp/screenshots");
    try std.testing.expectEqualStrings("/tmp/screenshots", capture.output_dir);
    try std.testing.expectEqual(ScreenCapture.ImageFormat.PNG, capture.format);
}

test "ImageFormat extension" {
    try std.testing.expectEqualStrings(".png", ScreenCapture.ImageFormat.PNG.extension());
    try std.testing.expectEqualStrings(".jpg", ScreenCapture.ImageFormat.JPEG.extension());
}

