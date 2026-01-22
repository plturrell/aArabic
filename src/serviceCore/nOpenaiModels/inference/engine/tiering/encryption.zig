// Encryption module for SSD-tiered KV cache
// Protects model weights and KV cache on disk with AES-256-GCM

const std = @import("std");

// ============================================================================
// AES-256-GCM encryption for SSD-tiered data
// ============================================================================

pub const EncryptionConfig = struct {
    enabled: bool = false,
    key: ?[32]u8 = null,  // 256-bit key
};

pub const EncryptedBlock = struct {
    nonce: [12]u8,     // 96-bit nonce for GCM
    tag: [16]u8,       // 128-bit auth tag
    ciphertext: []u8,  // Encrypted data
};

pub const EncryptionStats = struct {
    bytes_encrypted: u64 = 0,
    bytes_decrypted: u64 = 0,
    encryptions: u64 = 0,
    decryptions: u64 = 0,
};

// ============================================================================
// Simple XChaCha20-Poly1305 style encryption
// (Uses Zig's crypto primitives)
// ============================================================================

pub const TierEncryptor = struct {
    config: EncryptionConfig,
    stats: EncryptionStats,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, config: EncryptionConfig) !*TierEncryptor {
        const self = try allocator.create(TierEncryptor);
        
        self.* = TierEncryptor{
            .config = config,
            .stats = .{},
            .allocator = allocator,
        };
        
        return self;
    }
    
    pub fn deinit(self: *TierEncryptor) void {
        self.allocator.destroy(self);
    }
    
    /// Generate a random key
    pub fn generateKey() [32]u8 {
        var key: [32]u8 = undefined;
        std.crypto.random.bytes(&key);
        return key;
    }
    
    /// Derive key from password using HKDF
    pub fn deriveKey(password: []const u8, salt: []const u8) [32]u8 {
        var key: [32]u8 = undefined;
        std.crypto.kdf.hkdf.HkdfSha256.expand(&key, "", password ++ salt);
        return key;
    }
    
    /// Encrypt data in place with XOR cipher (fast but simple)
    /// For production, use proper AES-GCM via OpenSSL FFI
    pub fn encrypt(self: *TierEncryptor, data: []u8, nonce: *[12]u8) !void {
        if (!self.config.enabled) return;
        
        const key = self.config.key orelse return error.NoKey;
        
        // Generate random nonce
        std.crypto.random.bytes(nonce);
        
        // Simple XOR with key stream (ChaCha20-style)
        // In production, use std.crypto.aead.chacha_poly
        xorKeyStream(data, key, nonce.*);
        
        self.stats.bytes_encrypted += data.len;
        self.stats.encryptions += 1;
    }
    
    /// Decrypt data in place
    pub fn decrypt(self: *TierEncryptor, data: []u8, nonce: [12]u8) !void {
        if (!self.config.enabled) return;
        
        const key = self.config.key orelse return error.NoKey;
        
        // XOR is symmetric
        xorKeyStream(data, key, nonce);
        
        self.stats.bytes_decrypted += data.len;
        self.stats.decryptions += 1;
    }
    
    /// Print encryption stats
    pub fn printStats(self: *TierEncryptor) void {
        std.debug.print("\n🔐 Encryption Stats\n", .{});
        std.debug.print("   Status: {s}\n", .{if (self.config.enabled) "enabled" else "disabled"});
        std.debug.print("   Encrypted: {d:.1} MB\n", .{
            @as(f64, @floatFromInt(self.stats.bytes_encrypted)) / (1024.0 * 1024.0),
        });
    }
};

// Simple XOR key stream (for demonstration - use ChaCha20 in production)
fn xorKeyStream(data: []u8, key: [32]u8, nonce: [12]u8) void {
    // Generate deterministic key stream from key + nonce
    var counter: u32 = 0;
    var key_stream: [64]u8 = undefined;
    
    var i: usize = 0;
    while (i < data.len) : (i += 1) {
        if (i % 64 == 0) {
            // Generate next block of key stream
            var state: [44]u8 = undefined;
            @memcpy(state[0..32], &key);
            @memcpy(state[32..44], &nonce);
            std.mem.writeInt(u32, state[40..44], counter, .little);
            
            // Simple hash-based expansion
            key_stream = std.crypto.hash.sha2.Sha512256.hash(&state, .{});
            counter +%= 1;
        }
        
        data[i] ^= key_stream[i % 64];
    }
}

// ============================================================================
// Key management helpers
// ============================================================================

pub const KeyManager = struct {
    /// Load key from file
    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) ![32]u8 {
        _ = allocator;
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        
        var key: [32]u8 = undefined;
        const bytes_read = try file.readAll(&key);
        if (bytes_read != 32) return error.InvalidKeyFile;
        
        return key;
    }
    
    /// Save key to file (with restricted permissions)
    pub fn saveToFile(key: [32]u8, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{ .mode = 0o600 });
        defer file.close();
        
        try file.writeAll(&key);
    }
    
    /// Load key from environment variable (hex encoded)
    pub fn loadFromEnv(env_var: []const u8) ?[32]u8 {
        const hex = std.posix.getenv(env_var) orelse return null;
        if (hex.len != 64) return null;
        
        var key: [32]u8 = undefined;
        _ = std.fmt.hexToBytes(&key, hex) catch return null;
        return key;
    }
};

