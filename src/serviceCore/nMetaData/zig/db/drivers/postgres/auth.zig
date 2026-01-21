const std = @import("std");
const crypto = std.crypto;

/// MD5 password authentication
/// Format: "md5" + md5(md5(password + user) + salt)
pub fn computeMd5Password(
    allocator: std.mem.Allocator,
    password: []const u8,
    user: []const u8,
    salt: [4]u8,
) ![]const u8 {
    // Step 1: md5(password + user)
    var hash1: [16]u8 = undefined;
    var hasher1 = crypto.hash.Md5.init(.{});
    hasher1.update(password);
    hasher1.update(user);
    hasher1.final(&hash1);

    // Convert to hex
    var hex1: [32]u8 = undefined;
    _ = std.fmt.bufPrint(&hex1, "{x:0>32}", .{std.fmt.fmtSliceHexLower(&hash1)}) catch unreachable;

    // Step 2: md5(hex1 + salt)
    var hash2: [16]u8 = undefined;
    var hasher2 = crypto.hash.Md5.init(.{});
    hasher2.update(&hex1);
    hasher2.update(&salt);
    hasher2.final(&hash2);

    // Convert to hex and prepend "md5"
    const result = try allocator.alloc(u8, 35); // "md5" + 32 hex chars
    @memcpy(result[0..3], "md5");
    _ = std.fmt.bufPrint(result[3..], "{x:0>32}", .{std.fmt.fmtSliceHexLower(&hash2)}) catch unreachable;

    return result;
}

/// SCRAM-SHA-256 authentication
pub const ScramSha256 = struct {
    allocator: std.mem.Allocator,
    client_nonce: []const u8,
    server_nonce: []const u8,
    salt: []const u8,
    iterations: u32,
    client_first_bare: []const u8,
    server_first: []const u8,

    pub fn init(allocator: std.mem.Allocator) !ScramSha256 {
        // Generate client nonce (24 random bytes, base64 encoded)
        var nonce_bytes: [24]u8 = undefined;
        crypto.random.bytes(&nonce_bytes);
        const nonce = try base64Encode(allocator, &nonce_bytes);

        return ScramSha256{
            .allocator = allocator,
            .client_nonce = nonce,
            .server_nonce = &.{},
            .salt = &.{},
            .iterations = 0,
            .client_first_bare = &.{},
            .server_first = &.{},
        };
    }

    pub fn deinit(self: *ScramSha256) void {
        self.allocator.free(self.client_nonce);
        if (self.server_nonce.len > 0) self.allocator.free(self.server_nonce);
        if (self.salt.len > 0) self.allocator.free(self.salt);
        if (self.client_first_bare.len > 0) self.allocator.free(self.client_first_bare);
        if (self.server_first.len > 0) self.allocator.free(self.server_first);
    }

    /// Generate client-first message
    pub fn clientFirstMessage(self: *ScramSha256) ![]const u8 {
        // Format: n,,n=<user>,r=<client-nonce>
        // For now, we use * as username (server will use auth user)
        const bare = try std.fmt.allocPrint(
            self.allocator,
            "n=*,r={s}",
            .{self.client_nonce},
        );
        self.client_first_bare = bare;

        const full = try std.fmt.allocPrint(
            self.allocator,
            "n,,{s}",
            .{bare},
        );

        return full;
    }

    /// Parse server-first message
    pub fn parseServerFirst(self: *ScramSha256, message: []const u8) !void {
        self.server_first = try self.allocator.dupe(u8, message);

        var it = std.mem.split(u8, message, ",");
        
        // Parse r=<server-nonce>
        if (it.next()) |r_part| {
            if (!std.mem.startsWith(u8, r_part, "r=")) return error.InvalidServerFirst;
            self.server_nonce = try self.allocator.dupe(u8, r_part[2..]);
        } else return error.InvalidServerFirst;

        // Parse s=<salt>
        if (it.next()) |s_part| {
            if (!std.mem.startsWith(u8, s_part, "s=")) return error.InvalidServerFirst;
            self.salt = try base64Decode(self.allocator, s_part[2..]);
        } else return error.InvalidServerFirst;

        // Parse i=<iterations>
        if (it.next()) |i_part| {
            if (!std.mem.startsWith(u8, i_part, "i=")) return error.InvalidServerFirst;
            self.iterations = try std.fmt.parseInt(u32, i_part[2..], 10);
        } else return error.InvalidServerFirst;
    }

    /// Generate client-final message
    pub fn clientFinalMessage(self: *ScramSha256, password: []const u8) ![]const u8 {
        // Compute SaltedPassword = Hi(password, salt, iterations)
        const salted_password = try pbkdf2Sha256(
            self.allocator,
            password,
            self.salt,
            self.iterations,
        );
        defer self.allocator.free(salted_password);

        // Compute ClientKey = HMAC(SaltedPassword, "Client Key")
        var client_key: [32]u8 = undefined;
        crypto.auth.hmac.sha2.HmacSha256.create(&client_key, "Client Key", salted_password);

        // Compute StoredKey = SHA256(ClientKey)
        var stored_key: [32]u8 = undefined;
        crypto.hash.sha2.Sha256.hash(&client_key, &stored_key, .{});

        // Build auth message
        const channel_binding = "n,,"; // No channel binding
        const client_final_without_proof = try std.fmt.allocPrint(
            self.allocator,
            "c={s},r={s}",
            .{ try base64EncodeStr(self.allocator, channel_binding), self.server_nonce },
        );
        defer self.allocator.free(client_final_without_proof);

        const auth_message = try std.fmt.allocPrint(
            self.allocator,
            "{s},{s},{s}",
            .{ self.client_first_bare, self.server_first, client_final_without_proof },
        );
        defer self.allocator.free(auth_message);

        // Compute ClientSignature = HMAC(StoredKey, AuthMessage)
        var client_signature: [32]u8 = undefined;
        crypto.auth.hmac.sha2.HmacSha256.create(&client_signature, auth_message, &stored_key);

        // Compute ClientProof = ClientKey XOR ClientSignature
        var client_proof: [32]u8 = undefined;
        for (0..32) |i| {
            client_proof[i] = client_key[i] ^ client_signature[i];
        }

        // Encode proof
        const proof_b64 = try base64Encode(self.allocator, &client_proof);
        defer self.allocator.free(proof_b64);

        // Build final message
        return try std.fmt.allocPrint(
            self.allocator,
            "{s},p={s}",
            .{ client_final_without_proof, proof_b64 },
        );
    }
};

/// PBKDF2-HMAC-SHA256
fn pbkdf2Sha256(
    allocator: std.mem.Allocator,
    password: []const u8,
    salt: []const u8,
    iterations: u32,
) ![]u8 {
    const dklen = 32; // SHA256 output length
    var result = try allocator.alloc(u8, dklen);

    // PBKDF2 with single block (dklen <= hash length)
    var u: [32]u8 = undefined;
    var temp: [32]u8 = undefined;

    // U1 = HMAC(password, salt || INT(1))
    var salt_with_i = try allocator.alloc(u8, salt.len + 4);
    defer allocator.free(salt_with_i);
    @memcpy(salt_with_i[0..salt.len], salt);
    std.mem.writeInt(u32, salt_with_i[salt.len..][0..4], 1, .big);

    crypto.auth.hmac.sha2.HmacSha256.create(&u, salt_with_i, password);
    @memcpy(&temp, &u);

    // Ui = HMAC(password, Ui-1) for iterations
    var i: u32 = 1;
    while (i < iterations) : (i += 1) {
        crypto.auth.hmac.sha2.HmacSha256.create(&u, &u, password);
        for (0..32) |j| {
            temp[j] ^= u[j];
        }
    }

    @memcpy(result, &temp);
    return result;
}

/// Base64 encode bytes
fn base64Encode(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    const encoder = std.base64.standard.Encoder;
    const encoded_len = encoder.calcSize(data.len);
    const result = try allocator.alloc(u8, encoded_len);
    _ = encoder.encode(result, data);
    return result;
}

/// Base64 encode string
fn base64EncodeStr(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    return base64Encode(allocator, data);
}

/// Base64 decode
fn base64Decode(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    const decoder = std.base64.standard.Decoder;
    const decoded_len = try decoder.calcSizeForSlice(data);
    const result = try allocator.alloc(u8, decoded_len);
    try decoder.decode(result, data);
    return result;
}

// ============================================================================
// Unit Tests
// ============================================================================

test "MD5 password - basic" {
    const allocator = std.testing.allocator;

    const password = "secret";
    const user = "postgres";
    var salt = [_]u8{ 0x12, 0x34, 0x56, 0x78 };

    const result = try computeMd5Password(allocator, password, user, salt);
    defer allocator.free(result);

    // Should start with "md5"
    try std.testing.expect(std.mem.startsWith(u8, result, "md5"));
    // Should be 35 characters total
    try std.testing.expectEqual(@as(usize, 35), result.len);
}

test "MD5 password - empty password" {
    const allocator = std.testing.allocator;

    const password = "";
    const user = "postgres";
    var salt = [_]u8{ 0, 0, 0, 0 };

    const result = try computeMd5Password(allocator, password, user, salt);
    defer allocator.free(result);

    try std.testing.expect(std.mem.startsWith(u8, result, "md5"));
}

test "ScramSha256 - init and deinit" {
    const allocator = std.testing.allocator;

    var scram = try ScramSha256.init(allocator);
    defer scram.deinit();

    try std.testing.expect(scram.client_nonce.len > 0);
    try std.testing.expectEqual(@as(u32, 0), scram.iterations);
}

test "ScramSha256 - client first message" {
    const allocator = std.testing.allocator;

    var scram = try ScramSha256.init(allocator);
    defer scram.deinit();

    const msg = try scram.clientFirstMessage();
    defer allocator.free(msg);

    // Should start with "n,,"
    try std.testing.expect(std.mem.startsWith(u8, msg, "n,,"));
    // Should contain "n=*,r="
    try std.testing.expect(std.mem.indexOf(u8, msg, "n=*,r=") != null);
}

test "ScramSha256 - parse server first" {
    const allocator = std.testing.allocator;

    var scram = try ScramSha256.init(allocator);
    defer scram.deinit();

    // Typical server-first message
    const server_msg = "r=clientnonce123servernonce456,s=c2FsdA==,i=4096";
    try scram.parseServerFirst(server_msg);

    try std.testing.expect(scram.server_nonce.len > 0);
    try std.testing.expect(scram.salt.len > 0);
    try std.testing.expectEqual(@as(u32, 4096), scram.iterations);
}

test "base64 encode/decode" {
    const allocator = std.testing.allocator;

    const original = "Hello, World!";
    const encoded = try base64Encode(allocator, original);
    defer allocator.free(encoded);

    const decoded = try base64Decode(allocator, encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(original, decoded);
}

test "pbkdf2Sha256 - basic" {
    const allocator = std.testing.allocator;

    const password = "password";
    const salt = "salt";
    const iterations = 1;

    const result = try pbkdf2Sha256(allocator, password, salt, iterations);
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 32), result.len);
}

test "pbkdf2Sha256 - multiple iterations" {
    const allocator = std.testing.allocator;

    const password = "password";
    const salt = "salt";
    const iterations = 100;

    const result = try pbkdf2Sha256(allocator, password, salt, iterations);
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 32), result.len);
}
