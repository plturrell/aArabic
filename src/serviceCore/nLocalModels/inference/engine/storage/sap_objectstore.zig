///! SAP BTP Object Store Storage Backend (S3-Compatible)
///! Implements StorageBackend interface with AWS Signature V4 auth

const std = @import("std");
const storage_backend = @import("storage_backend.zig");
const StorageBackend = storage_backend.StorageBackend;
const StorageError = storage_backend.StorageError;

/// Configuration for SAP Object Store connection
pub const SAPObjectStoreConfig = struct {
    endpoint: []const u8, // S3-compatible endpoint URL
    access_key: []const u8, // Access key ID
    secret_key: []const u8, // Secret access key
    bucket: []const u8, // Bucket name
    region: []const u8, // AWS region
    prefix: []const u8 = "", // Optional key prefix

    /// Parse configuration from BTP service binding JSON
    pub fn fromServiceKey(allocator: std.mem.Allocator, json_str: []const u8) !SAPObjectStoreConfig {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_str, .{}) catch return StorageError.InvalidData;
        defer parsed.deinit();
        const root = parsed.value;
        if (root != .object) return StorageError.InvalidData;
        const credentials = root.object.get("credentials") orelse return StorageError.InvalidData;
        if (credentials != .object) return StorageError.InvalidData;
        const c = credentials.object;
        const getStr = struct {
            fn f(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
                const v = obj.get(key) orelse return null;
                return if (v == .string) v.string else null;
            }
        }.f;
        return .{
            .endpoint = try allocator.dupe(u8, getStr(c, "endpoint") orelse getStr(c, "uri") orelse return StorageError.InvalidData),
            .access_key = try allocator.dupe(u8, getStr(c, "access_key_id") orelse return StorageError.InvalidData),
            .secret_key = try allocator.dupe(u8, getStr(c, "secret_access_key") orelse return StorageError.InvalidData),
            .bucket = try allocator.dupe(u8, getStr(c, "bucket") orelse return StorageError.InvalidData),
            .region = try allocator.dupe(u8, getStr(c, "region") orelse "eu-central-1"),
            .prefix = try allocator.dupe(u8, getStr(c, "prefix") orelse ""),
        };
    }

    pub fn deinit(self: *SAPObjectStoreConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.endpoint);
        allocator.free(self.access_key);
        allocator.free(self.secret_key);
        allocator.free(self.bucket);
        allocator.free(self.region);
        if (self.prefix.len > 0) allocator.free(self.prefix);
    }
};

/// SAP Object Store storage implementation
pub const SAPObjectStore = struct {
    config: SAPObjectStoreConfig,
    allocator: std.mem.Allocator,
    http_client: std.http.Client,
    const Self = @This();

    pub const vtable = StorageBackend.VTable{ .read = read, .write = write, .exists = exists, .list = list, .delete = deleteObject };

    pub fn init(allocator: std.mem.Allocator, config: SAPObjectStoreConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);
        self.* = .{ .config = config, .allocator = allocator, .http_client = std.http.Client{ .allocator = allocator } };
        std.log.info("[sap-objectstore] Initialized: endpoint={s} bucket={s}", .{ config.endpoint, config.bucket });
        return self;
    }

    pub fn fromServiceKey(allocator: std.mem.Allocator, json_str: []const u8) !*Self {
        return init(allocator, try SAPObjectStoreConfig.fromServiceKey(allocator, json_str));
    }

    pub fn backend(self: *Self) StorageBackend {
        return .{ .vtable = &vtable, .ctx = @ptrCast(self) };
    }

    pub fn deinit(self: *Self) void {
        self.http_client.deinit();
        var cfg = self.config;
        cfg.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    fn buildObjectKey(self: *const Self, alloc: std.mem.Allocator, path: []const u8) ![]u8 {
        return if (self.config.prefix.len == 0) alloc.dupe(u8, path) else std.fmt.allocPrint(alloc, "{s}/{s}", .{ self.config.prefix, path });
    }

    fn buildObjectUrl(self: *const Self, alloc: std.mem.Allocator, key: []const u8) ![]u8 {
        return std.fmt.allocPrint(alloc, "{s}/{s}/{s}", .{ self.config.endpoint, self.config.bucket, key });
    }

    fn getTimestamp() [16]u8 {
        const epoch = std.time.epoch.EpochSeconds{ .secs = @intCast(std.time.timestamp()) };
        const yd = epoch.getEpochDay().calculateYearDay();
        const md = yd.calculateMonthDay();
        const ds = epoch.getDaySeconds();
        var buf: [16]u8 = undefined;
        _ = std.fmt.bufPrint(&buf, "{d:0>4}{d:0>2}{d:0>2}T{d:0>2}{d:0>2}{d:0>2}Z", .{ yd.year, md.month.numeric(), md.day_index + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute() }) catch unreachable;
        return buf;
    }

    fn hmacSha256(key: []const u8, data: []const u8) [32]u8 {
        var mac: [32]u8 = undefined;
        std.crypto.auth.hmac.sha2.HmacSha256.create(&mac, data, key);
        return mac;
    }

    fn generateAuthHeader(self: *const Self, alloc: std.mem.Allocator, method: []const u8, path: []const u8, ts: []const u8) ![]u8 {
        const date = ts[0..8];
        var ch = std.crypto.hash.sha2.Sha256.init(.{});
        ch.update(method);
        ch.update("\n");
        ch.update(path);
        ch.update("\n\nhost:");
        ch.update(self.config.endpoint);
        ch.update("\nx-amz-date:");
        ch.update(ts);
        ch.update("\n\nhost;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        const scope = try std.fmt.allocPrint(alloc, "{s}/{s}/s3/aws4_request", .{ date, self.config.region });
        defer alloc.free(scope);
        var sh = std.crypto.hash.sha2.Sha256.init(.{});
        sh.update("AWS4-HMAC-SHA256\n");
        sh.update(ts);
        sh.update("\n");
        sh.update(scope);
        sh.update("\n");
        sh.update(&std.fmt.bytesToHex(ch.finalResult(), .lower));
        const ks = try std.fmt.allocPrint(alloc, "AWS4{s}", .{self.config.secret_key});
        defer alloc.free(ks);
        const sig = hmacSha256(&hmacSha256(&hmacSha256(&hmacSha256(ks, date), self.config.region), "s3"), "aws4_request");
        const final_sig = hmacSha256(&sig, &std.fmt.bytesToHex(sh.finalResult(), .lower));
        return std.fmt.allocPrint(alloc, "AWS4-HMAC-SHA256 Credential={s}/{s},SignedHeaders=host;x-amz-date,Signature={s}", .{ self.config.access_key, scope, std.fmt.bytesToHex(final_sig, .lower) });
    }

    fn read(ctx: *anyopaque, path: []const u8, alloc: std.mem.Allocator) anyerror![]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const key = try self.buildObjectKey(alloc, path);
        defer alloc.free(key);
        const url = try self.buildObjectUrl(alloc, key);
        defer alloc.free(url);
        const uri = std.Uri.parse(url) catch return StorageError.InvalidPath;
        const ts = getTimestamp();
        const auth = try self.generateAuthHeader(alloc, "GET", key, &ts);
        defer alloc.free(auth);
        var buf = std.ArrayList(u8){};
        errdefer buf.deinit();
        const res = self.http_client.fetch(.{ .location = .{ .uri = uri }, .method = .GET, .extra_headers = &.{ .{ .name = "Authorization", .value = auth }, .{ .name = "x-amz-date", .value = &ts }, .{ .name = "x-amz-content-sha256", .value = "UNSIGNED-PAYLOAD" } }, .response_storage = .{ .dynamic = &buf } });
        if (res) |r| {
            if (r.status == .ok) return buf.toOwnedSlice();
            buf.deinit();
            return if (r.status == .not_found) StorageError.PathNotFound else if (r.status == .forbidden) StorageError.PermissionDenied else StorageError.ReadFailed;
        } else |_| { buf.deinit(); return StorageError.ConnectionLost; }
    }

    fn write(ctx: *anyopaque, path: []const u8, data: []const u8) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const key = try self.buildObjectKey(self.allocator, path);
        defer self.allocator.free(key);
        const url = try self.buildObjectUrl(self.allocator, key);
        defer self.allocator.free(url);
        const uri = std.Uri.parse(url) catch return StorageError.InvalidPath;
        const ts = getTimestamp();
        const auth = try self.generateAuthHeader(self.allocator, "PUT", key, &ts);
        defer self.allocator.free(auth);
        var ch = std.crypto.hash.sha2.Sha256.init(.{});
        ch.update(data);
        const hash_hex = std.fmt.bytesToHex(ch.finalResult(), .lower);
        const res = self.http_client.fetch(.{ .location = .{ .uri = uri }, .method = .PUT, .extra_headers = &.{ .{ .name = "Authorization", .value = auth }, .{ .name = "x-amz-date", .value = &ts }, .{ .name = "x-amz-content-sha256", .value = &hash_hex }, .{ .name = "Content-Type", .value = "application/octet-stream" } }, .payload = data });
        if (res) |r| {
            if (r.status == .ok or r.status == .created or r.status == .no_content) return;
            return if (r.status == .forbidden) StorageError.PermissionDenied else StorageError.WriteFailed;
        } else |_| return StorageError.ConnectionLost;
    }

    fn exists(ctx: *anyopaque, path: []const u8) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const key = self.buildObjectKey(self.allocator, path) catch return false;
        defer self.allocator.free(key);
        const url = self.buildObjectUrl(self.allocator, key) catch return false;
        defer self.allocator.free(url);
        const uri = std.Uri.parse(url) catch return false;
        const ts = getTimestamp();
        const auth = self.generateAuthHeader(self.allocator, "HEAD", key, &ts) catch return false;
        defer self.allocator.free(auth);
        const res = self.http_client.fetch(.{ .location = .{ .uri = uri }, .method = .HEAD, .extra_headers = &.{ .{ .name = "Authorization", .value = auth }, .{ .name = "x-amz-date", .value = &ts }, .{ .name = "x-amz-content-sha256", .value = "UNSIGNED-PAYLOAD" } } });
        return if (res) |r| r.status == .ok else false;
    }

    fn list(ctx: *anyopaque, prefix: []const u8, alloc: std.mem.Allocator) anyerror![][]const u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const fp = try self.buildObjectKey(alloc, prefix);
        defer alloc.free(fp);
        const url = try std.fmt.allocPrint(alloc, "{s}/{s}?list-type=2&prefix={s}", .{ self.config.endpoint, self.config.bucket, fp });
        defer alloc.free(url);
        const uri = std.Uri.parse(url) catch return StorageError.InvalidPath;
        const ts = getTimestamp();
        const auth = try self.generateAuthHeader(alloc, "GET", "", &ts);
        defer alloc.free(auth);
        var buf = std.ArrayList(u8){};
        defer buf.deinit();
        const res = self.http_client.fetch(.{ .location = .{ .uri = uri }, .method = .GET, .extra_headers = &.{ .{ .name = "Authorization", .value = auth }, .{ .name = "x-amz-date", .value = &ts }, .{ .name = "x-amz-content-sha256", .value = "UNSIGNED-PAYLOAD" } }, .response_storage = .{ .dynamic = &buf } });
        if (res) |r| { if (r.status != .ok) return StorageError.ListFailed; } else |_| return StorageError.ConnectionLost;
        return parseListXml(alloc, buf.items, self.config.prefix);
    }

    fn deleteObject(ctx: *anyopaque, path: []const u8) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const key = try self.buildObjectKey(self.allocator, path);
        defer self.allocator.free(key);
        const url = try self.buildObjectUrl(self.allocator, key);
        defer self.allocator.free(url);
        const uri = std.Uri.parse(url) catch return StorageError.InvalidPath;
        const ts = getTimestamp();
        const auth = try self.generateAuthHeader(self.allocator, "DELETE", key, &ts);
        defer self.allocator.free(auth);
        const res = self.http_client.fetch(.{ .location = .{ .uri = uri }, .method = .DELETE, .extra_headers = &.{ .{ .name = "Authorization", .value = auth }, .{ .name = "x-amz-date", .value = &ts }, .{ .name = "x-amz-content-sha256", .value = "UNSIGNED-PAYLOAD" } } });
        if (res) |r| {
            if (r.status == .ok or r.status == .no_content or r.status == .not_found) return;
            return if (r.status == .forbidden) StorageError.PermissionDenied else StorageError.DeleteFailed;
        } else |_| return StorageError.ConnectionLost;
    }
};

fn parseListXml(alloc: std.mem.Allocator, xml: []const u8, strip_prefix: []const u8) ![][]const u8 {
    var result: std.ArrayListUnmanaged([]const u8) = .empty;
    errdefer { for (result.items) |i| alloc.free(i); result.deinit(alloc); }
    var pos: usize = 0;
    while (pos < xml.len) {
        const ks = std.mem.indexOf(u8, xml[pos..], "<Key>") orelse break;
        const cs = pos + ks + 5;
        const ke = std.mem.indexOf(u8, xml[cs..], "</Key>") orelse break;
        const key = xml[cs .. cs + ke];
        const stripped = if (strip_prefix.len > 0 and std.mem.startsWith(u8, key, strip_prefix)) key[strip_prefix.len + 1 ..] else key;
        if (stripped.len > 0) try result.append(alloc, try alloc.dupe(u8, stripped));
        pos = cs + ke + 6;
    }
    return result.toOwnedSlice(alloc);
}

test "SAPObjectStoreConfig fromServiceKey" {
    const a = std.testing.allocator;
    var cfg = try SAPObjectStoreConfig.fromServiceKey(a, \\{"credentials":{"endpoint":"https://s3.eu-central-1.amazonaws.com","access_key_id":"AKIA","secret_access_key":"s","bucket":"b","region":"eu-central-1"}});
    defer cfg.deinit(a);
    try std.testing.expectEqualStrings("https://s3.eu-central-1.amazonaws.com", cfg.endpoint);
}

test "parseListXml" {
    const a = std.testing.allocator;
    const keys = try parseListXml(a, "<Contents><Key>p/f1.txt</Key></Contents><Contents><Key>p/f2.txt</Key></Contents>", "p");
    defer { for (keys) |k| a.free(k); a.free(keys); }
    try std.testing.expectEqual(@as(usize, 2), keys.len);
}

test "SAPObjectStore init/deinit" {
    const a = std.testing.allocator;
    const s = try SAPObjectStore.init(a, .{ .endpoint = "https://s3.example.com", .access_key = "k", .secret_key = "s", .bucket = "b", .region = "r", .prefix = "" });
    _ = s.backend();
    s.deinit();
}
