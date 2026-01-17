const std = @import("std");
const slide_handler = @import("slide_handler.zig");

/// Slides OData endpoints
/// Provides Presentation entity CRUD and actions:
/// - GenerateSlides: Create new presentation
/// - ExportPresentation: Export with options

/// Presentation entity response
pub const PresentationEntity = struct {
    PresentationId: []const u8,
    SourceId: []const u8,
    Title: []const u8,
    Author: []const u8,
    Theme: []const u8,
    FilePath: []const u8,
    FileSize: u64,
    NumSlides: u32,
    TargetAudience: []const u8,
    DetailLevel: []const u8,
    GeneratedAt: i64,
    ProcessingTimeMs: ?u64,
    Status: []const u8,
    ErrorMessage: ?[]const u8,
    Version: u32,
    ExportFormat: []const u8,
};

/// Slide entity response
pub const SlideEntity = struct {
    SlideId: []const u8,
    PresentationId: []const u8,
    SlideNumber: u32,
    Layout: []const u8,
    Title: []const u8,
    Content: []const u8,
    Subtitle: ?[]const u8,
    Notes: ?[]const u8,
};

/// Request for GenerateSlides action
pub const GenerateSlidesRequest = struct {
    SourceId: []const u8,
    Title: ?[]const u8 = null,
    Theme: ?[]const u8 = null,
    TargetAudience: ?[]const u8 = null,
    DetailLevel: ?[]const u8 = null,
    NumSlides: ?u32 = null,
};

/// Response from GenerateSlides action
pub const GenerateSlidesResponse = struct {
    PresentationId: []const u8,
    Status: []const u8,
    FilePath: []const u8,
    NumSlides: u32,
    Message: []const u8,
};

/// Request for ExportPresentation action
pub const ExportPresentationRequest = struct {
    PresentationId: []const u8,
    Format: ?[]const u8 = null,
    IncludeNotes: ?bool = null,
    Standalone: ?bool = null,
    Compress: ?bool = null,
};

/// Response from ExportPresentation action
pub const ExportPresentationResponse = struct {
    PresentationId: []const u8,
    ExportPath: []const u8,
    Format: []const u8,
    FileSize: u64,
    Message: []const u8,
};

/// Handle GenerateSlides OData action
/// POST /odata/v4/research/GenerateSlides
pub fn handleGenerateSlides(
    allocator: std.mem.Allocator,
    request: GenerateSlidesRequest,
) !GenerateSlidesResponse {
    std.log.info("GenerateSlides action called", .{});
    std.log.info("  SourceId: {s}", .{request.SourceId});
    std.log.info("  Theme: {?s}", .{request.Theme});
    std.log.info("  NumSlides: {?d}", .{request.NumSlides});
    
    // Build generation options
    const options = slide_handler.GenerationOptions{
        .target_audience = request.TargetAudience orelse "general",
        .detail_level = request.DetailLevel orelse "medium",
        .num_slides = request.NumSlides orelse 7,
        .theme = request.Theme orelse "professional",
    };
    
    // Initialize handler
    var handler = slide_handler.SlideHandler.init(allocator);
    defer handler.deinit();
    
    // Generate slides
    const result = try handler.generatePresentation(
        request.SourceId,
        request.Title orelse "Research Presentation",
        options,
    );
    
    // Build response
    const message = try std.fmt.allocPrint(
        allocator,
        "Presentation generated successfully with {d} slides",
        .{result.num_slides},
    );
    
    return GenerateSlidesResponse{
        .PresentationId = result.presentation_id,
        .Status = "completed",
        .FilePath = result.file_path,
        .NumSlides = result.num_slides,
        .Message = message,
    };
}

/// Handle ExportPresentation OData action
/// POST /odata/v4/research/ExportPresentation
pub fn handleExportPresentation(
    allocator: std.mem.Allocator,
    request: ExportPresentationRequest,
) !ExportPresentationResponse {
    std.log.info("ExportPresentation action called", .{});
    std.log.info("  PresentationId: {s}", .{request.PresentationId});
    std.log.info("  Format: {?s}", .{request.Format});
    std.log.info("  IncludeNotes: {?}", .{request.IncludeNotes});
    
    // Build export options
    const export_options = slide_handler.ExportOptions{
        .format = request.Format orelse "html",
        .include_notes = request.IncludeNotes orelse false,
        .standalone = request.Standalone orelse true,
        .compress = request.Compress orelse false,
    };
    
    // Initialize handler
    var handler = slide_handler.SlideHandler.init(allocator);
    defer handler.deinit();
    
    // Export presentation
    const export_path = try handler.exportPresentation(
        request.PresentationId,
        export_options,
    );
    
    // Get file size
    const file = try std.fs.cwd().openFile(export_path, .{});
    defer file.close();
    const file_stat = try file.stat();
    const file_size = file_stat.size;
    
    // Build response
    const message = try std.fmt.allocPrint(
        allocator,
        "Presentation exported successfully in {s} format",
        .{export_options.format},
    );
    
    return ExportPresentationResponse{
        .PresentationId = request.PresentationId,
        .ExportPath = export_path,
        .Format = export_options.format,
        .FileSize = file_size,
        .Message = message,
    };
}

/// Handle GET /odata/v4/research/Presentation
/// Get all presentation entities, optionally filtered by SourceId
pub fn handleGetPresentationList(
    allocator: std.mem.Allocator,
    source_id: ?[]const u8,
) ![]PresentationEntity {
    std.log.info("GET Presentation collection", .{});
    if (source_id) |sid| {
        std.log.info("  Filtered by SourceId: {s}", .{sid});
    }
    
    // Initialize handler
    var handler = slide_handler.SlideHandler.init(allocator);
    defer handler.deinit();
    
    // Get presentations
    const presentations = if (source_id) |sid|
        try handler.listPresentations(sid)
    else
        try handler.listAllPresentations();
    
    // Convert to entities
    var entities = try allocator.alloc(PresentationEntity, presentations.len);
    for (presentations, 0..) |pres, i| {
        entities[i] = PresentationEntity{
            .PresentationId = pres.presentation_id,
            .SourceId = pres.source_id,
            .Title = pres.title,
            .Author = pres.author,
            .Theme = pres.theme,
            .FilePath = pres.file_path,
            .FileSize = pres.file_size,
            .NumSlides = pres.num_slides,
            .TargetAudience = pres.target_audience,
            .DetailLevel = pres.detail_level,
            .GeneratedAt = pres.generated_at,
            .ProcessingTimeMs = pres.processing_time_ms,
            .Status = pres.status,
            .ErrorMessage = pres.error_message,
            .Version = pres.version,
            .ExportFormat = pres.export_format,
        };
    }
    
    return entities;
}

/// Handle GET /odata/v4/research/Presentation('{id}')
/// Get single presentation entity by ID
pub fn handleGetPresentation(
    allocator: std.mem.Allocator,
    presentation_id: []const u8,
) !?PresentationEntity {
    std.log.info("GET Presentation by ID: {s}", .{presentation_id});
    
    // Initialize handler
    var handler = slide_handler.SlideHandler.init(allocator);
    defer handler.deinit();
    
    // Get presentation
    const pres_opt = try handler.getPresentation(presentation_id);
    if (pres_opt == null) {
        return null;
    }
    
    const pres = pres_opt.?;
    
    return PresentationEntity{
        .PresentationId = pres.presentation_id,
        .SourceId = pres.source_id,
        .Title = pres.title,
        .Author = pres.author,
        .Theme = pres.theme,
        .FilePath = pres.file_path,
        .FileSize = pres.file_size,
        .NumSlides = pres.num_slides,
        .TargetAudience = pres.target_audience,
        .DetailLevel = pres.detail_level,
        .GeneratedAt = pres.generated_at,
        .ProcessingTimeMs = pres.processing_time_ms,
        .Status = pres.status,
        .ErrorMessage = pres.error_message,
        .Version = pres.version,
        .ExportFormat = pres.export_format,
    };
}

/// Handle GET /odata/v4/research/Presentation('{id}')/Slides
/// Get all slides for a presentation
pub fn handleGetSlides(
    allocator: std.mem.Allocator,
    presentation_id: []const u8,
) ![]SlideEntity {
    std.log.info("GET Slides for Presentation: {s}", .{presentation_id});
    
    // Initialize handler
    var handler = slide_handler.SlideHandler.init(allocator);
    defer handler.deinit();
    
    // Get slides
    const slides = try handler.getSlides(presentation_id);
    
    // Convert to entities
    var entities = try allocator.alloc(SlideEntity, slides.len);
    for (slides, 0..) |slide, i| {
        entities[i] = SlideEntity{
            .SlideId = slide.slide_id,
            .PresentationId = slide.presentation_id,
            .SlideNumber = slide.slide_number,
            .Layout = slide.layout,
            .Title = slide.title,
            .Content = slide.content,
            .Subtitle = slide.subtitle,
            .Notes = slide.notes,
        };
    }
    
    return entities;
}

/// Handle DELETE /odata/v4/research/Presentation('{id}')
/// Delete presentation entity (and all slides)
pub fn handleDeletePresentation(
    allocator: std.mem.Allocator,
    presentation_id: []const u8,
) !void {
    std.log.info("DELETE Presentation: {s}", .{presentation_id});
    
    // Initialize handler
    var handler = slide_handler.SlideHandler.init(allocator);
    defer handler.deinit();
    
    // Delete presentation (cascades to slides and deletes file)
    try handler.deletePresentation(presentation_id);
    
    std.log.info("Presentation deleted successfully: {s}", .{presentation_id});
}

/// Serialize PresentationEntity to JSON
pub fn serializePresentationEntity(
    allocator: std.mem.Allocator,
    entity: PresentationEntity,
) ![]u8 {
    const error_msg = if (entity.ErrorMessage) |msg|
        try std.fmt.allocPrint(allocator, "\"{s}\"", .{msg})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(error_msg);
    
    const processing_time = if (entity.ProcessingTimeMs) |pt|
        try std.fmt.allocPrint(allocator, "{d}", .{pt})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(processing_time);
    
    return try std.fmt.allocPrint(
        allocator,
        \\{{
        \\  "PresentationId": "{s}",
        \\  "SourceId": "{s}",
        \\  "Title": "{s}",
        \\  "Author": "{s}",
        \\  "Theme": "{s}",
        \\  "FilePath": "{s}",
        \\  "FileSize": {d},
        \\  "NumSlides": {d},
        \\  "TargetAudience": "{s}",
        \\  "DetailLevel": "{s}",
        \\  "GeneratedAt": {d},
        \\  "ProcessingTimeMs": {s},
        \\  "Status": "{s}",
        \\  "ErrorMessage": {s},
        \\  "Version": {d},
        \\  "ExportFormat": "{s}"
        \\}}
        ,
        .{
            entity.PresentationId,
            entity.SourceId,
            entity.Title,
            entity.Author,
            entity.Theme,
            entity.FilePath,
            entity.FileSize,
            entity.NumSlides,
            entity.TargetAudience,
            entity.DetailLevel,
            entity.GeneratedAt,
            processing_time,
            entity.Status,
            error_msg,
            entity.Version,
            entity.ExportFormat,
        },
    );
}

/// Serialize SlideEntity to JSON
pub fn serializeSlideEntity(
    allocator: std.mem.Allocator,
    entity: SlideEntity,
) ![]u8 {
    const subtitle = if (entity.Subtitle) |sub|
        try std.fmt.allocPrint(allocator, "\"{s}\"", .{sub})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(subtitle);
    
    const notes = if (entity.Notes) |n|
        try std.fmt.allocPrint(allocator, "\"{s}\"", .{n})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(notes);
    
    return try std.fmt.allocPrint(
        allocator,
        \\{{
        \\  "SlideId": "{s}",
        \\  "PresentationId": "{s}",
        \\  "SlideNumber": {d},
        \\  "Layout": "{s}",
        \\  "Title": "{s}",
        \\  "Content": "{s}",
        \\  "Subtitle": {s},
        \\  "Notes": {s}
        \\}}
        ,
        .{
            entity.SlideId,
            entity.PresentationId,
            entity.SlideNumber,
            entity.Layout,
            entity.Title,
            entity.Content,
            subtitle,
            notes,
        },
    );
}

/// Serialize GenerateSlidesResponse to JSON
pub fn serializeGenerateSlidesResponse(
    allocator: std.mem.Allocator,
    response: GenerateSlidesResponse,
) ![]u8 {
    return try std.fmt.allocPrint(
        allocator,
        \\{{
        \\  "PresentationId": "{s}",
        \\  "Status": "{s}",
        \\  "FilePath": "{s}",
        \\  "NumSlides": {d},
        \\  "Message": "{s}"
        \\}}
        ,
        .{
            response.PresentationId,
            response.Status,
            response.FilePath,
            response.NumSlides,
            response.Message,
        },
    );
}

/// Serialize ExportPresentationResponse to JSON
pub fn serializeExportPresentationResponse(
    allocator: std.mem.Allocator,
    response: ExportPresentationResponse,
) ![]u8 {
    return try std.fmt.allocPrint(
        allocator,
        \\{{
        \\  "PresentationId": "{s}",
        \\  "ExportPath": "{s}",
        \\  "Format": "{s}",
        \\  "FileSize": {d},
        \\  "Message": "{s}"
        \\}}
        ,
        .{
            response.PresentationId,
            response.ExportPath,
            response.Format,
            response.FileSize,
            response.Message,
        },
    );
}

test "generate slides request parsing" {
    const allocator = std.testing.allocator;
    
    const request = GenerateSlidesRequest{
        .SourceId = "source_123",
        .Title = "AI Research Overview",
        .Theme = "professional",
        .TargetAudience = "technical",
        .DetailLevel = "high",
        .NumSlides = 10,
    };
    
    try std.testing.expectEqualStrings("source_123", request.SourceId);
    try std.testing.expectEqualStrings("professional", request.Theme.?);
    try std.testing.expect(request.NumSlides.? == 10);
    
    _ = allocator;
}

test "export presentation request parsing" {
    const allocator = std.testing.allocator;
    
    const request = ExportPresentationRequest{
        .PresentationId = "pres_123",
        .Format = "html",
        .IncludeNotes = true,
        .Standalone = true,
        .Compress = false,
    };
    
    try std.testing.expectEqualStrings("pres_123", request.PresentationId);
    try std.testing.expectEqualStrings("html", request.Format.?);
    try std.testing.expect(request.IncludeNotes.? == true);
    
    _ = allocator;
}
