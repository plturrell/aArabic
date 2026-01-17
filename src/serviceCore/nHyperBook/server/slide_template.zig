const std = @import("std");

/// Slide template engine for generating HTML presentations
/// Day 46: Slide Template Engine

/// Slide layout types
pub const SlideLayout = enum {
    title,
    content,
    two_column,
    bullet_points,
    quote,
    image,
    conclusion,

    pub fn toString(self: SlideLayout) []const u8 {
        return switch (self) {
            .title => "title",
            .content => "content",
            .two_column => "two_column",
            .bullet_points => "bullet_points",
            .quote => "quote",
            .image => "image",
            .conclusion => "conclusion",
        };
    }
};

/// Slide theme/style
pub const SlideTheme = enum {
    professional,
    minimal,
    dark,
    academic,

    pub fn toString(self: SlideTheme) []const u8 {
        return switch (self) {
            .professional => "professional",
            .minimal => "minimal",
            .dark => "dark",
            .academic => "academic",
        };
    }
};

/// Individual slide structure
pub const Slide = struct {
    layout: SlideLayout,
    title: []const u8,
    content: []const u8,
    subtitle: ?[]const u8 = null,
    notes: ?[]const u8 = null,
    
    pub fn init(
        layout: SlideLayout,
        title: []const u8,
        content: []const u8,
    ) Slide {
        return Slide{
            .layout = layout,
            .title = title,
            .content = content,
        };
    }
};

/// Presentation structure
pub const Presentation = struct {
    title: []const u8,
    author: []const u8,
    theme: SlideTheme,
    slides: []const Slide,
    
    pub fn init(
        title: []const u8,
        author: []const u8,
        theme: SlideTheme,
        slides: []const Slide,
    ) Presentation {
        return Presentation{
            .title = title,
            .author = author,
            .theme = theme,
            .slides = slides,
        };
    }
};

/// HTML template renderer
pub const TemplateRenderer = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) TemplateRenderer {
        return TemplateRenderer{
            .allocator = allocator,
        };
    }
    
    /// Generate complete HTML presentation
    pub fn render(
        self: *TemplateRenderer,
        presentation: Presentation,
    ) ![]const u8 {
        var html = std.ArrayList(u8).init(self.allocator);
        defer html.deinit();
        
        const writer = html.writer();
        
        // HTML header
        try self.writeHeader(writer, presentation);
        
        // Slides
        try self.writeSlides(writer, presentation);
        
        // HTML footer
        try self.writeFooter(writer);
        
        return html.toOwnedSlice();
    }
    
    /// Write HTML header with theme styles
    fn writeHeader(
        self: *TemplateRenderer,
        writer: anytype,
        presentation: Presentation,
    ) !void {
        _ = self;
        
        try writer.writeAll(
            \\<!DOCTYPE html>
            \\<html lang="en">
            \\<head>
            \\    <meta charset="UTF-8">
            \\    <meta name="viewport" content="width=device-width, initial-scale=1.0">
            \\    <title>
        );
        
        try writer.writeAll(presentation.title);
        
        try writer.writeAll(
            \\</title>
            \\    <style>
            \\        * {
            \\            margin: 0;
            \\            padding: 0;
            \\            box-sizing: border-box;
            \\        }
            \\        
            \\        body {
            \\            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            \\            overflow: hidden;
            \\        }
            \\        
            \\        .presentation {
            \\            width: 100vw;
            \\            height: 100vh;
            \\            position: relative;
            \\        }
            \\        
            \\        .slide {
            \\            width: 100%;
            \\            height: 100%;
            \\            display: none;
            \\            padding: 60px 80px;
            \\            position: absolute;
            \\            top: 0;
            \\            left: 0;
            \\        }
            \\        
            \\        .slide.active {
            \\            display: flex;
            \\            flex-direction: column;
            \\            justify-content: center;
            \\        }
            \\        
            \\        .slide h1 {
            \\            font-size: 3.5rem;
            \\            margin-bottom: 1rem;
            \\        }
            \\        
            \\        .slide h2 {
            \\            font-size: 2.5rem;
            \\            margin-bottom: 2rem;
            \\        }
            \\        
            \\        .slide h3 {
            \\            font-size: 1.8rem;
            \\            margin-bottom: 1.5rem;
            \\        }
            \\        
            \\        .slide p {
            \\            font-size: 1.5rem;
            \\            line-height: 1.8;
            \\            margin-bottom: 1rem;
            \\        }
            \\        
            \\        .slide ul {
            \\            font-size: 1.5rem;
            \\            line-height: 2;
            \\            margin-left: 2rem;
            \\            list-style-type: disc;
            \\        }
            \\        
            \\        .slide ul li {
            \\            margin-bottom: 0.8rem;
            \\        }
            \\        
            \\        .slide-footer {
            \\            position: absolute;
            \\            bottom: 20px;
            \\            right: 40px;
            \\            font-size: 1rem;
            \\            color: #666;
            \\        }
            \\        
            \\        .navigation {
            \\            position: fixed;
            \\            bottom: 40px;
            \\            left: 50%;
            \\            transform: translateX(-50%);
            \\            display: flex;
            \\            gap: 20px;
            \\            z-index: 1000;
            \\        }
            \\        
            \\        .nav-button {
            \\            padding: 12px 24px;
            \\            font-size: 1.1rem;
            \\            background: #0070f2;
            \\            color: white;
            \\            border: none;
            \\            border-radius: 8px;
            \\            cursor: pointer;
            \\            transition: background 0.3s;
            \\        }
            \\        
            \\        .nav-button:hover {
            \\            background: #0056c0;
            \\        }
            \\        
            \\        .nav-button:disabled {
            \\            background: #ccc;
            \\            cursor: not-allowed;
            \\        }
            \\        
            \\        .slide-counter {
            \\            padding: 12px 24px;
            \\            background: #f0f0f0;
            \\            border-radius: 8px;
            \\            font-size: 1.1rem;
            \\            font-weight: 600;
            \\        }
            \\        
        );
        
        // Theme-specific styles
        try self.writeThemeStyles(writer, presentation.theme);
        
        try writer.writeAll(
            \\    </style>
            \\</head>
            \\<body>
            \\    <div class="presentation">
            \\
        );
    }
    
    /// Write theme-specific CSS
    fn writeThemeStyles(
        self: *TemplateRenderer,
        writer: anytype,
        theme: SlideTheme,
    ) !void {
        _ = self;
        
        switch (theme) {
            .professional => {
                try writer.writeAll(
                    \\        /* Professional Theme */
                    \\        .slide {
                    \\            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    \\            color: white;
                    \\        }
                    \\        
                    \\        .slide h1, .slide h2, .slide h3 {
                    \\            color: white;
                    \\        }
                    \\        
                );
            },
            .minimal => {
                try writer.writeAll(
                    \\        /* Minimal Theme */
                    \\        .slide {
                    \\            background: white;
                    \\            color: #333;
                    \\        }
                    \\        
                    \\        .slide h1, .slide h2, .slide h3 {
                    \\            color: #0070f2;
                    \\        }
                    \\        
                );
            },
            .dark => {
                try writer.writeAll(
                    \\        /* Dark Theme */
                    \\        .slide {
                    \\            background: #1a1a2e;
                    \\            color: #eee;
                    \\        }
                    \\        
                    \\        .slide h1, .slide h2, .slide h3 {
                    \\            color: #64ffda;
                    \\        }
                    \\        
                );
            },
            .academic => {
                try writer.writeAll(
                    \\        /* Academic Theme */
                    \\        .slide {
                    \\            background: #f5f5f5;
                    \\            color: #2c3e50;
                    \\        }
                    \\        
                    \\        .slide h1, .slide h2, .slide h3 {
                    \\            color: #34495e;
                    \\            border-bottom: 3px solid #3498db;
                    \\            padding-bottom: 0.5rem;
                    \\        }
                    \\        
                );
            },
        }
    }
    
    /// Write all slides
    fn writeSlides(
        self: *TemplateRenderer,
        writer: anytype,
        presentation: Presentation,
    ) !void {
        for (presentation.slides, 0..) |slide, index| {
            const is_first = index == 0;
            try self.writeSlide(writer, slide, index + 1, presentation.slides.len, is_first);
        }
    }
    
    /// Write individual slide based on layout
    fn writeSlide(
        self: *TemplateRenderer,
        writer: anytype,
        slide: Slide,
        slide_num: usize,
        total_slides: usize,
        is_active: bool,
    ) !void {
        _ = self;
        
        // Slide container
        if (is_active) {
            try writer.writeAll("        <div class=\"slide active\" data-slide=\"");
        } else {
            try writer.writeAll("        <div class=\"slide\" data-slide=\"");
        }
        try writer.print("{d}", .{slide_num});
        try writer.writeAll("\">\n");
        
        // Slide content based on layout
        switch (slide.layout) {
            .title => try self.writeTitleSlide(writer, slide),
            .content => try self.writeContentSlide(writer, slide),
            .two_column => try self.writeTwoColumnSlide(writer, slide),
            .bullet_points => try self.writeBulletPointsSlide(writer, slide),
            .quote => try self.writeQuoteSlide(writer, slide),
            .image => try self.writeImageSlide(writer, slide),
            .conclusion => try self.writeConclusionSlide(writer, slide),
        }
        
        // Slide footer
        try writer.writeAll("            <div class=\"slide-footer\">");
        try writer.print("{d} / {d}", .{ slide_num, total_slides });
        try writer.writeAll("</div>\n");
        
        try writer.writeAll("        </div>\n\n");
    }
    
    /// Title slide layout
    fn writeTitleSlide(self: *TemplateRenderer, writer: anytype, slide: Slide) !void {
        _ = self;
        try writer.writeAll("            <div style=\"text-align: center;\">\n");
        try writer.writeAll("                <h1>");
        try writer.writeAll(slide.title);
        try writer.writeAll("</h1>\n");
        
        if (slide.subtitle) |subtitle| {
            try writer.writeAll("                <h3 style=\"margin-top: 2rem; font-weight: 300;\">");
            try writer.writeAll(subtitle);
            try writer.writeAll("</h3>\n");
        }
        
        try writer.writeAll("                <p style=\"margin-top: 3rem; font-size: 1.3rem;\">");
        try writer.writeAll(slide.content);
        try writer.writeAll("</p>\n");
        try writer.writeAll("            </div>\n");
    }
    
    /// Content slide layout
    fn writeContentSlide(self: *TemplateRenderer, writer: anytype, slide: Slide) !void {
        _ = self;
        try writer.writeAll("            <h2>");
        try writer.writeAll(slide.title);
        try writer.writeAll("</h2>\n");
        try writer.writeAll("            <div style=\"margin-top: 2rem;\">\n");
        try writer.writeAll("                <p>");
        try writer.writeAll(slide.content);
        try writer.writeAll("</p>\n");
        try writer.writeAll("            </div>\n");
    }
    
    /// Two column slide layout
    fn writeTwoColumnSlide(self: *TemplateRenderer, writer: anytype, slide: Slide) !void {
        _ = self;
        try writer.writeAll("            <h2>");
        try writer.writeAll(slide.title);
        try writer.writeAll("</h2>\n");
        try writer.writeAll("            <div style=\"display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 2rem;\">\n");
        try writer.writeAll("                <div><p>");
        try writer.writeAll(slide.content);
        try writer.writeAll("</p></div>\n");
        try writer.writeAll("                <div><p>");
        if (slide.subtitle) |subtitle| {
            try writer.writeAll(subtitle);
        }
        try writer.writeAll("</p></div>\n");
        try writer.writeAll("            </div>\n");
    }
    
    /// Bullet points slide layout
    fn writeBulletPointsSlide(self: *TemplateRenderer, writer: anytype, slide: Slide) !void {
        _ = self;
        try writer.writeAll("            <h2>");
        try writer.writeAll(slide.title);
        try writer.writeAll("</h2>\n");
        try writer.writeAll("            <div style=\"margin-top: 2rem;\">\n");
        try writer.writeAll("                <ul>\n");
        
        // Parse content for bullet points (split by newlines)
        var lines = std.mem.split(u8, slide.content, "\n");
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len > 0) {
                try writer.writeAll("                    <li>");
                try writer.writeAll(trimmed);
                try writer.writeAll("</li>\n");
            }
        }
        
        try writer.writeAll("                </ul>\n");
        try writer.writeAll("            </div>\n");
    }
    
    /// Quote slide layout
    fn writeQuoteSlide(self: *TemplateRenderer, writer: anytype, slide: Slide) !void {
        _ = self;
        try writer.writeAll("            <div style=\"text-align: center; display: flex; flex-direction: column; justify-content: center; height: 100%;\">\n");
        try writer.writeAll("                <blockquote style=\"font-size: 2.5rem; font-style: italic; margin: 0;\">\n");
        try writer.writeAll("                    \"");
        try writer.writeAll(slide.content);
        try writer.writeAll("\"\n");
        try writer.writeAll("                </blockquote>\n");
        try writer.writeAll("                <p style=\"margin-top: 2rem; font-size: 1.5rem;\">— ");
        try writer.writeAll(slide.title);
        try writer.writeAll("</p>\n");
        try writer.writeAll("            </div>\n");
    }
    
    /// Image slide layout
    fn writeImageSlide(self: *TemplateRenderer, writer: anytype, slide: Slide) !void {
        _ = self;
        try writer.writeAll("            <h2>");
        try writer.writeAll(slide.title);
        try writer.writeAll("</h2>\n");
        try writer.writeAll("            <div style=\"text-align: center; margin-top: 2rem;\">\n");
        try writer.writeAll("                <div style=\"font-size: 4rem; margin-bottom: 2rem;\">📊</div>\n");
        try writer.writeAll("                <p>");
        try writer.writeAll(slide.content);
        try writer.writeAll("</p>\n");
        try writer.writeAll("            </div>\n");
    }
    
    /// Conclusion slide layout
    fn writeConclusionSlide(self: *TemplateRenderer, writer: anytype, slide: Slide) !void {
        _ = self;
        try writer.writeAll("            <div style=\"text-align: center;\">\n");
        try writer.writeAll("                <h1>");
        try writer.writeAll(slide.title);
        try writer.writeAll("</h1>\n");
        try writer.writeAll("                <div style=\"margin-top: 3rem;\">\n");
        try writer.writeAll("                    <p style=\"font-size: 1.8rem;\">");
        try writer.writeAll(slide.content);
        try writer.writeAll("</p>\n");
        try writer.writeAll("                </div>\n");
        try writer.writeAll("            </div>\n");
    }
    
    /// Write HTML footer with navigation JavaScript
    fn writeFooter(self: *TemplateRenderer, writer: anytype) !void {
        _ = self;
        try writer.writeAll(
            \\    </div>
            \\    
            \\    <div class="navigation">
            \\        <button class="nav-button" id="prevBtn" onclick="navigateSlide(-1)">← Previous</button>
            \\        <div class="slide-counter">
            \\            <span id="currentSlide">1</span> / <span id="totalSlides">1</span>
            \\        </div>
            \\        <button class="nav-button" id="nextBtn" onclick="navigateSlide(1)">Next →</button>
            \\    </div>
            \\    
            \\    <script>
            \\        let currentSlide = 1;
            \\        const slides = document.querySelectorAll('.slide');
            \\        const totalSlides = slides.length;
            \\        
            \\        document.getElementById('totalSlides').textContent = totalSlides;
            \\        
            \\        function updateSlide() {
            \\            slides.forEach((slide, index) => {
            \\                if (index + 1 === currentSlide) {
            \\                    slide.classList.add('active');
            \\                } else {
            \\                    slide.classList.remove('active');
            \\                }
            \\            });
            \\            
            \\            document.getElementById('currentSlide').textContent = currentSlide;
            \\            document.getElementById('prevBtn').disabled = currentSlide === 1;
            \\            document.getElementById('nextBtn').disabled = currentSlide === totalSlides;
            \\        }
            \\        
            \\        function navigateSlide(direction) {
            \\            const newSlide = currentSlide + direction;
            \\            if (newSlide >= 1 && newSlide <= totalSlides) {
            \\                currentSlide = newSlide;
            \\                updateSlide();
            \\            }
            \\        }
            \\        
            \\        // Keyboard navigation
            \\        document.addEventListener('keydown', (e) => {
            \\            if (e.key === 'ArrowLeft') {
            \\                navigateSlide(-1);
            \\            } else if (e.key === 'ArrowRight') {
            \\                navigateSlide(1);
            \\            }
            \\        });
            \\        
            \\        // Initialize
            \\        updateSlide();
            \\    </script>
            \\</body>
            \\</html>
            \\
        );
    }
};

// Tests
test "create presentation" {
    const allocator = std.testing.allocator;
    
    const slides = [_]Slide{
        Slide.init(.title, "Test Presentation", "By Test Author"),
        Slide.init(.content, "Introduction", "This is a test slide."),
    };
    
    const presentation = Presentation.init(
        "Test Presentation",
        "Test Author",
        .professional,
        &slides,
    );
    
    try std.testing.expectEqual(slides.len, presentation.slides.len);
    try std.testing.expectEqualStrings("Test Presentation", presentation.title);
}

test "render presentation" {
    const allocator = std.testing.allocator;
    
    const slides = [_]Slide{
        Slide.init(.title, "My Presentation", "By John Doe"),
    };
    
    const presentation = Presentation.init(
        "My Presentation",
        "John Doe",
        .minimal,
        &slides,
    );
    
    var renderer = TemplateRenderer.init(allocator);
    const html = try renderer.render(presentation);
    defer allocator.free(html);
    
    try std.testing.expect(html.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, html, "<!DOCTYPE html>") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "My Presentation") != null);
}
