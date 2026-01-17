const std = @import("std");
const audio_handler = @import("audio_handler.zig");

/// Audio OData endpoints
/// Provides Audio entity CRUD and GenerateAudio action

/// Audio entity response
pub const AudioEntity = struct {
    AudioId: []const u8,
    SourceId: []const u8,
    Title: []const u8,
    FilePath: []const u8,
    FileSize: u64,
    DurationSeconds: f32,
    SampleRate: u32,
    BitDepth: u8,
    Channels: u8,
    Provider: []const u8,
    Voice: []const u8,
    GeneratedAt: i64,
    ProcessingTimeMs: ?u64,
    Status: []const u8,
    ErrorMessage: ?[]const u8,
};

/// Request for GenerateAudio action
pub const GenerateAudioRequest = struct {
    SourceId: []const u8,
    Text: []const u8,
    Voice: ?[]const u8 = null,
    Format: ?[]const u8 = null,
};

/// Response from GenerateAudio action
pub const GenerateAudioResponse = struct {
    AudioId: []const u8,
    Status: []const u8,
    FilePath: []const u8,
    Message: []const u8,
};

/// Handle GenerateAudio OData action
/// POST /odata/v4/research/GenerateAudio
pub fn handleGenerateAudio(
    allocator: std.mem.Allocator,
    request: GenerateAudioRequest,
) !GenerateAudioResponse {
    std.log.info("GenerateAudio action called", .{});
    std.log.info("  SourceId: {s}", .{request.SourceId});
    std.log.info("  Text length: {} chars", .{request.Text.len});
    
    // Build audio request
    const audio_request = audio_handler.AudioRequest{
        .source_id = request.SourceId,
        .text = request.Text,
        .voice = request.Voice orelse "default",
        .format = request.Format orelse "mp3",
    };
    
    // Generate audio (currently stub)
    const metadata = try audio_handler.generateAudio(allocator, audio_request);
    
    // Build response
    const message = try std.fmt.allocPrint(
        allocator,
        "Audio generation initiated. AudioLabShimmy integration pending.",
        .{}
    );
    
    return GenerateAudioResponse{
        .AudioId = metadata.audio_id,
        .Status = metadata.status,
        .FilePath = metadata.file_path,
        .Message = message,
    };
}

/// Handle GET /odata/v4/research/Audio
/// Get all audio entities
pub fn handleGetAudioList(
    allocator: std.mem.Allocator,
) ![]AudioEntity {
    std.log.info("GET Audio collection", .{});
    
    // TODO: Query from SQLite database
    // For now, return empty array
    _ = allocator;
    
    const empty_list = try allocator.alloc(AudioEntity, 0);
    return empty_list;
}

/// Handle GET /odata/v4/research/Audio('{id}')
/// Get single audio entity by ID
pub fn handleGetAudio(
    allocator: std.mem.Allocator,
    audio_id: []const u8,
) !?AudioEntity {
    std.log.info("GET Audio by ID: {s}", .{audio_id});
    
    // TODO: Query from SQLite database
    // For now, return null (not found)
    _ = allocator;
    
    return null;
}

/// Handle DELETE /odata/v4/research/Audio('{id}')
/// Delete audio entity
pub fn handleDeleteAudio(
    allocator: std.mem.Allocator,
    audio_id: []const u8,
) !void {
    std.log.info("DELETE Audio: {s}", .{audio_id});
    
    // TODO: 
    // 1. Delete from SQLite database
    // 2. Delete audio file from filesystem
    _ = allocator;
    
    std.log.info("Audio deleted (stub): {s}", .{audio_id});
}

/// Serialize AudioEntity to JSON
pub fn serializeAudioEntity(
    allocator: std.mem.Allocator,
    entity: AudioEntity,
) ![]u8 {
    return try std.fmt.allocPrint(
        allocator,
        \\{{
        \\  "AudioId": "{s}",
        \\  "SourceId": "{s}",
        \\  "Title": "{s}",
        \\  "FilePath": "{s}",
        \\  "FileSize": {d},
        \\  "DurationSeconds": {d:.2},
        \\  "SampleRate": {d},
        \\  "BitDepth": {d},
        \\  "Channels": {d},
        \\  "Provider": "{s}",
        \\  "Voice": "{s}",
        \\  "GeneratedAt": {d},
        \\  "ProcessingTimeMs": {?d},
        \\  "Status": "{s}",
        \\  "ErrorMessage": {?s}
        \\}}
        ,
        .{
            entity.AudioId,
            entity.SourceId,
            entity.Title,
            entity.FilePath,
            entity.FileSize,
            entity.DurationSeconds,
            entity.SampleRate,
            entity.BitDepth,
            entity.Channels,
            entity.Provider,
            entity.Voice,
            entity.GeneratedAt,
            entity.ProcessingTimeMs,
            entity.Status,
            entity.ErrorMessage,
        },
    );
}

/// Serialize GenerateAudioResponse to JSON
pub fn serializeGenerateAudioResponse(
    allocator: std.mem.Allocator,
    response: GenerateAudioResponse,
) ![]u8 {
    return try std.fmt.allocPrint(
        allocator,
        \\{{
        \\  "AudioId": "{s}",
        \\  "Status": "{s}",
        \\  "FilePath": "{s}",
        \\  "Message": "{s}"
        \\}}
        ,
        .{
            response.AudioId,
            response.Status,
            response.FilePath,
            response.Message,
        },
    );
}

test "generate audio request parsing" {
    const allocator = std.testing.allocator;
    
    const request = GenerateAudioRequest{
        .SourceId = "source_123",
        .Text = "This is a test audio generation.",
        .Voice = "echo",
        .Format = "mp3",
    };
    
    try std.testing.expectEqualStrings("source_123", request.SourceId);
    try std.testing.expectEqualStrings("echo", request.Voice.?);
    
    _ = allocator;
}
