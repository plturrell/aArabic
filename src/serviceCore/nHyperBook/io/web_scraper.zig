const std = @import("std");
const http = @import("http_client.zig");
const html = @import("html_parser.zig");

// ============================================================================
// HyperShimmy Web Scraper
// ============================================================================
//
// Day 13: Web scraper integration
//
// Combines HTTP client (Day 11) + HTML parser (Day 12) to:
// - Download web pages
// - Parse HTML content
// - Extract text, links, and metadata
// - Handle errors gracefully
// - Provide structured data for storage
// ============================================================================

/// Scraped content from a web page
pub const ScrapedContent = struct {
    url: []const u8,
    title: ?[]const u8,
    text: []const u8,
    links: []const []const u8,
    status_code: u16,
    error_message: ?[]const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ScrapedContent) void {
        self.allocator.free(self.url);
        if (self.title) |t| self.allocator.free(t);
        self.allocator.free(self.text);
        
        for (self.links) |link| {
            self.allocator.free(link);
        }
        self.allocator.free(self.links);
        
        if (self.error_message) |msg| self.allocator.free(msg);
    }
};

/// Web scraper configuration
pub const ScraperConfig = struct {
    follow_redirects: bool = true,
    max_redirects: u8 = 10,
    timeout_ms: u64 = 30000,
    user_agent: []const u8 = "HyperShimmy/1.0",
    max_content_length: usize = 10 * 1024 * 1024, // 10 MB default
};

/// Web scraper
pub const WebScraper = struct {
    allocator: std.mem.Allocator,
    http_client: http.HttpClient,
    html_parser: html.HtmlParser,
    config: ScraperConfig,

    pub fn init(allocator: std.mem.Allocator, config: ScraperConfig) WebScraper {
        return WebScraper{
            .allocator = allocator,
            .http_client = http.HttpClient.init(allocator),
            .html_parser = html.HtmlParser.init(allocator),
            .config = config,
        };
    }

    pub fn deinit(self: *WebScraper) void {
        self.http_client.deinit();
        self.html_parser.deinit();
    }

    /// Scrape a web page and extract content
    pub fn scrape(self: *WebScraper, url: []const u8) !ScrapedContent {
        // Make HTTP request
        var response = self.http_client.request(.{
            .method = .GET,
            .url = url,
            .follow_redirects = self.config.follow_redirects,
            .max_redirects = self.config.max_redirects,
            .timeout_ms = self.config.timeout_ms,
        }) catch |err| {
            // Return error information in ScrapedContent
            return ScrapedContent{
                .url = try self.allocator.dupe(u8, url),
                .title = null,
                .text = try self.allocator.dupe(u8, ""),
                .links = try self.allocator.alloc([]const u8, 0),
                .status_code = 0,
                .error_message = try std.fmt.allocPrint(
                    self.allocator,
                    "HTTP request failed: {s}",
                    .{@errorName(err)},
                ),
                .allocator = self.allocator,
            };
        };
        defer response.deinit();

        // Check if response is HTML
        const content_type = response.headers.get("content-type") orelse "";
        const is_html = std.mem.indexOf(u8, content_type, "text/html") != null or
            std.mem.indexOf(u8, content_type, "application/xhtml") != null;

        if (!is_html and response.status_code == 200) {
            return ScrapedContent{
                .url = try self.allocator.dupe(u8, url),
                .title = null,
                .text = try self.allocator.dupe(u8, ""),
                .links = try self.allocator.alloc([]const u8, 0),
                .status_code = response.status_code,
                .error_message = try std.fmt.allocPrint(
                    self.allocator,
                    "Content-Type is not HTML: {s}",
                    .{content_type},
                ),
                .allocator = self.allocator,
            };
        }

        // Check content length
        if (response.body.len > self.config.max_content_length) {
            return ScrapedContent{
                .url = try self.allocator.dupe(u8, url),
                .title = null,
                .text = try self.allocator.dupe(u8, ""),
                .links = try self.allocator.alloc([]const u8, 0),
                .status_code = response.status_code,
                .error_message = try std.fmt.allocPrint(
                    self.allocator,
                    "Content too large: {d} bytes (max: {d})",
                    .{ response.body.len, self.config.max_content_length },
                ),
                .allocator = self.allocator,
            };
        }

        // Parse HTML
        var document = self.html_parser.parse(response.body) catch |err| {
            return ScrapedContent{
                .url = try self.allocator.dupe(u8, url),
                .title = null,
                .text = try self.allocator.dupe(u8, ""),
                .links = try self.allocator.alloc([]const u8, 0),
                .status_code = response.status_code,
                .error_message = try std.fmt.allocPrint(
                    self.allocator,
                    "HTML parsing failed: {s}",
                    .{@errorName(err)},
                ),
                .allocator = self.allocator,
            };
        };
        defer document.deinit();

        // Extract text content
        var text_buffer = std.ArrayListUnmanaged(u8){};
        defer text_buffer.deinit(self.allocator);
        
        try document.getText(&text_buffer);
        const text = try self.allocator.dupe(u8, text_buffer.items);

        // Extract title
        const title = try document.getTitle();

        // Extract links
        var links_list = try document.getLinks();
        defer links_list.deinit(self.allocator);

        // Copy links to owned memory
        var links = try self.allocator.alloc([]const u8, links_list.items.len);
        for (links_list.items, 0..) |link, i| {
            links[i] = try self.allocator.dupe(u8, link);
        }

        return ScrapedContent{
            .url = try self.allocator.dupe(u8, url),
            .title = title,
            .text = text,
            .links = links,
            .status_code = response.status_code,
            .error_message = null,
            .allocator = self.allocator,
        };
    }

    /// Scrape multiple URLs concurrently (future enhancement)
    /// For now, this is a sequential implementation
    pub fn scrapeMultiple(self: *WebScraper, urls: []const []const u8) ![]ScrapedContent {
        var results = try self.allocator.alloc(ScrapedContent, urls.len);
        
        for (urls, 0..) |url, i| {
            results[i] = try self.scrape(url);
        }
        
        return results;
    }

    /// Validate URL before scraping
    pub fn validateUrl(url: []const u8) bool {
        // Check if URL starts with http:// or https://
        if (std.mem.startsWith(u8, url, "http://") or 
            std.mem.startsWith(u8, url, "https://")) {
            return true;
        }
        return false;
    }

    /// Extract domain from URL
    pub fn extractDomain(allocator: std.mem.Allocator, url: []const u8) ![]const u8 {
        var parsed_url = try http.Url.parse(allocator, url);
        defer parsed_url.deinit(allocator);
        
        return try allocator.dupe(u8, parsed_url.host);
    }

    /// Clean and normalize text content
    pub fn cleanText(allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
        var cleaned = std.ArrayListUnmanaged(u8){};
        defer cleaned.deinit(allocator);

        var last_was_space = true; // Start as true to trim leading whitespace
        
        for (text) |c| {
            if (std.ascii.isWhitespace(c)) {
                if (!last_was_space) {
                    try cleaned.append(allocator, ' ');
                    last_was_space = true;
                }
            } else {
                try cleaned.append(allocator, c);
                last_was_space = false;
            }
        }

        // Trim trailing whitespace
        while (cleaned.items.len > 0 and cleaned.items[cleaned.items.len - 1] == ' ') {
            _ = cleaned.pop();
        }

        return try allocator.dupe(u8, cleaned.items);
    }

    /// Resolve relative URLs to absolute URLs
    pub fn resolveUrl(allocator: std.mem.Allocator, base_url: []const u8, relative_url: []const u8) ![]const u8 {
        // If already absolute, return as-is
        if (std.mem.startsWith(u8, relative_url, "http://") or 
            std.mem.startsWith(u8, relative_url, "https://")) {
            return try allocator.dupe(u8, relative_url);
        }

        var base = try http.Url.parse(allocator, base_url);
        defer base.deinit(allocator);

        // Handle protocol-relative URLs (//example.com/path)
        if (std.mem.startsWith(u8, relative_url, "//")) {
            return try std.fmt.allocPrint(allocator, "{s}:{s}", .{ base.scheme, relative_url });
        }

        // Handle absolute paths (/path)
        if (std.mem.startsWith(u8, relative_url, "/")) {
            return try std.fmt.allocPrint(
                allocator,
                "{s}://{s}:{d}{s}",
                .{ base.scheme, base.host, base.port, relative_url },
            );
        }

        // Handle relative paths (path or ./path or ../path)
        // For simplicity, append to base path
        const base_path = base.path;
        const last_slash = std.mem.lastIndexOf(u8, base_path, "/") orelse 0;
        const dir_path = base_path[0..last_slash];

        return try std.fmt.allocPrint(
            allocator,
            "{s}://{s}:{d}{s}/{s}",
            .{ base.scheme, base.host, base.port, dir_path, relative_url },
        );
    }
};

// ============================================================================
// Tests
// ============================================================================

test "web scraper init and deinit" {
    const config = ScraperConfig{};
    var scraper = WebScraper.init(std.testing.allocator, config);
    defer scraper.deinit();

    try std.testing.expect(scraper.config.follow_redirects);
    try std.testing.expectEqual(@as(u8, 10), scraper.config.max_redirects);
}

test "validate URL" {
    try std.testing.expect(WebScraper.validateUrl("http://example.com"));
    try std.testing.expect(WebScraper.validateUrl("https://example.com"));
    try std.testing.expect(!WebScraper.validateUrl("ftp://example.com"));
    try std.testing.expect(!WebScraper.validateUrl("example.com"));
    try std.testing.expect(!WebScraper.validateUrl(""));
}

test "extract domain" {
    const domain1 = try WebScraper.extractDomain(std.testing.allocator, "http://example.com/path");
    defer std.testing.allocator.free(domain1);
    try std.testing.expectEqualStrings("example.com", domain1);

    const domain2 = try WebScraper.extractDomain(std.testing.allocator, "https://sub.example.com:8080/api");
    defer std.testing.allocator.free(domain2);
    try std.testing.expectEqualStrings("sub.example.com", domain2);
}

test "clean text" {
    const dirty = "  Hello   World  \n\n  Test  ";
    const clean = try WebScraper.cleanText(std.testing.allocator, dirty);
    defer std.testing.allocator.free(clean);
    
    try std.testing.expectEqualStrings("Hello World Test", clean);
}

test "clean text - empty" {
    const clean = try WebScraper.cleanText(std.testing.allocator, "");
    defer std.testing.allocator.free(clean);
    
    try std.testing.expectEqualStrings("", clean);
}

test "clean text - only whitespace" {
    const clean = try WebScraper.cleanText(std.testing.allocator, "   \n\n  \t  ");
    defer std.testing.allocator.free(clean);
    
    try std.testing.expectEqualStrings("", clean);
}

test "resolve URL - absolute URL" {
    const resolved = try WebScraper.resolveUrl(
        std.testing.allocator,
        "http://example.com/page",
        "https://other.com/path",
    );
    defer std.testing.allocator.free(resolved);
    
    try std.testing.expectEqualStrings("https://other.com/path", resolved);
}

test "resolve URL - absolute path" {
    const resolved = try WebScraper.resolveUrl(
        std.testing.allocator,
        "http://example.com/page",
        "/api/data",
    );
    defer std.testing.allocator.free(resolved);
    
    try std.testing.expectEqualStrings("http://example.com:80/api/data", resolved);
}

test "resolve URL - relative path" {
    const resolved = try WebScraper.resolveUrl(
        std.testing.allocator,
        "http://example.com/dir/page",
        "other.html",
    );
    defer std.testing.allocator.free(resolved);
    
    try std.testing.expectEqualStrings("http://example.com:80/dir/other.html", resolved);
}

test "resolve URL - protocol relative" {
    const resolved = try WebScraper.resolveUrl(
        std.testing.allocator,
        "https://example.com/page",
        "//cdn.example.com/image.png",
    );
    defer std.testing.allocator.free(resolved);
    
    try std.testing.expectEqualStrings("https://cdn.example.com/image.png", resolved);
}

test "scrape - mock test with simple HTML" {
    // Note: This test requires a mock server or real HTTP endpoint
    // For unit testing, we test the components individually
    // Integration tests should be done with a test server
    
    // Test that scraper can be initialized and configured
    const config = ScraperConfig{
        .timeout_ms = 5000,
        .max_content_length = 1024 * 1024,
    };
    
    var scraper = WebScraper.init(std.testing.allocator, config);
    defer scraper.deinit();
    
    try std.testing.expectEqual(@as(u64, 5000), scraper.config.timeout_ms);
    try std.testing.expectEqual(@as(usize, 1024 * 1024), scraper.config.max_content_length);
}

test "scraped content init and deinit" {
    var links = try std.testing.allocator.alloc([]const u8, 2);
    links[0] = try std.testing.allocator.dupe(u8, "http://example.com/1");
    links[1] = try std.testing.allocator.dupe(u8, "http://example.com/2");
    
    var content = ScrapedContent{
        .url = try std.testing.allocator.dupe(u8, "http://example.com"),
        .title = try std.testing.allocator.dupe(u8, "Example"),
        .text = try std.testing.allocator.dupe(u8, "Content"),
        .links = links,
        .status_code = 200,
        .error_message = null,
        .allocator = std.testing.allocator,
    };
    defer content.deinit();
    
    try std.testing.expectEqualStrings("http://example.com", content.url);
    try std.testing.expectEqualStrings("Example", content.title.?);
    try std.testing.expectEqual(@as(usize, 2), content.links.len);
}

// Note: Live scraping tests require network access and external services
// Example integration test (requires test server):
//
// test "scrape - real website" {
//     const config = ScraperConfig{};
//     var scraper = WebScraper.init(std.testing.allocator, config);
//     defer scraper.deinit();
//     
//     var content = try scraper.scrape("http://example.com");
//     defer content.deinit();
//     
//     try std.testing.expectEqual(@as(u16, 200), content.status_code);
//     try std.testing.expect(content.text.len > 0);
// }
