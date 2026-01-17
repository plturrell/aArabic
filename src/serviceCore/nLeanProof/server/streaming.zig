const std = @import("std");

pub const StreamWriter = struct {
    writer: std.net.Stream,

    pub fn init(stream: std.net.Stream) StreamWriter {
        return .{ .writer = stream };
    }

    pub fn sendHeaders(self: *StreamWriter) !void {
        const header =
            "HTTP/1.1 200 OK\r\n" ++
            "Content-Type: text/event-stream\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
            "Cache-Control: no-cache\r\n" ++
            "Connection: keep-alive\r\n" ++
            "\r\n";
        _ = try self.writer.writeAll(header);
    }

    pub fn sendEvent(self: *StreamWriter, data: []const u8) !void {
        _ = try self.writer.writeAll("data: ");
        _ = try self.writer.writeAll(data);
        _ = try self.writer.writeAll("\n\n");
    }

    pub fn sendDone(self: *StreamWriter) !void {
        _ = try self.writer.writeAll("data: [DONE]\n\n");
    }
};
