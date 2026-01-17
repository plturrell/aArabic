const std = @import("std");
const slide_template = @import("../server/slide_template.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Creating sample presentation...\n", .{});

    // Create sample slides
    const slides = [_]slide_template.Slide{
        // Title slide
        slide_template.Slide{
            .layout = .title,
            .title = "Research Findings",
            .content = "HyperShimmy Project",
            .subtitle = "Automated Presentation Generation",
            .notes = null,
        },
        
        // Content slide
        slide_template.Slide{
            .layout = .content,
            .title = "Project Overview",
            .content = "HyperShimmy is a research assistant that provides automated document analysis, summarization, and presentation generation. It uses local LLM inference for privacy and performance.",
            .subtitle = null,
            .notes = null,
        },
        
        // Bullet points slide
        slide_template.Slide{
            .layout = .bullet_points,
            .title = "Key Features",
            .content = 
                \\Document ingestion (PDF, URL, text)
                \\Semantic search with embeddings
                \\AI-powered chat interface
                \\Research summarization
                \\Knowledge graph generation
                \\Audio overview creation
                \\Automated slide generation
            ,
            .subtitle = null,
            .notes = null,
        },
        
        // Two column slide
        slide_template.Slide{
            .layout = .two_column,
            .title = "Technology Stack",
            .content = "Backend: Zig server with OData V4 protocol. Mojo for AI/ML operations. Local LLM inference with Shimmy integration.",
            .subtitle = "Frontend: SAPUI5 enterprise UI framework. Responsive 3-column layout. Real-time updates via OData.",
            .notes = null,
        },
        
        // Quote slide
        slide_template.Slide{
            .layout = .quote,
            .title = "Project Vision",
            .content = "Building the future of research assistance with privacy-first, local AI processing",
            .subtitle = null,
            .notes = null,
        },
        
        // Image slide
        slide_template.Slide{
            .layout = .image,
            .title = "Architecture Diagram",
            .content = "Clean separation: UI Layer → OData V4 → Backend (Zig) → AI Layer (Mojo) → Local LLM",
            .subtitle = null,
            .notes = null,
        },
        
        // Conclusion slide
        slide_template.Slide{
            .layout = .conclusion,
            .title = "Thank You",
            .content = "Questions? Visit the project repository for more information.",
            .subtitle = null,
            .notes = null,
        },
    };

    // Create presentation
    const presentation = slide_template.Presentation.init(
        "HyperShimmy Research Findings",
        "HyperShimmy Team",
        .professional,
        &slides,
    );

    std.debug.print("Rendering HTML presentation...\n", .{});

    // Render to HTML
    var renderer = slide_template.TemplateRenderer.init(allocator);
    const html = try renderer.render(presentation);
    defer allocator.free(html);

    // Write to file
    const file = try std.fs.cwd().createFile("test_output/presentation.html", .{});
    defer file.close();
    
    try file.writeAll(html);

    std.debug.print("✓ Presentation generated: test_output/presentation.html\n", .{});
    std.debug.print("✓ Total slides: {d}\n", .{slides.len});
    std.debug.print("✓ Theme: {s}\n", .{presentation.theme.toString()});
    std.debug.print("✓ HTML size: {d} bytes\n", .{html.len});
}
