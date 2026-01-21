// Petri Net Engine - Core Implementation
// Part of serviceCore nWorkflow
// Days 1-3: Foundation
//
// A Petri Net is a mathematical modeling tool for distributed systems consisting of:
// - Places: Hold tokens (state)
// - Transitions: Process tokens (actions)
// - Arcs: Connect places and transitions (flow)
// - Tokens: Represent data/state

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Token represents data flowing through the Petri Net
/// Tokens can carry typed payloads for workflow data
pub const Token = struct {
    id: u64,
    data: []const u8, // JSON-encoded payload
    timestamp: i64,
    
    pub fn init(id: u64, data: []const u8) Token {
        return Token{
            .id = id,
            .data = data,
            .timestamp = std.time.timestamp(),
        };
    }
    
    pub fn deinit(self: *Token, allocator: Allocator) void {
        allocator.free(self.data);
    }
    
    pub fn clone(self: *const Token, allocator: Allocator) !Token {
        const data_copy = try allocator.dupe(u8, self.data);
        return Token{
            .id = self.id,
            .data = data_copy,
            .timestamp = self.timestamp,
        };
    }
};

/// Place holds tokens and represents state in the Petri Net
pub const Place = struct {
    id: []const u8,
    name: []const u8,
    tokens: std.ArrayList(Token),
    capacity: ?usize, // None = unlimited
    
    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, capacity: ?usize) !Place {
        return Place{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .tokens = .{},
            .capacity = capacity,
        };
    }
    
    pub fn deinit(self: *Place, allocator: Allocator) void {
        for (self.tokens.items) |*token| {
            token.deinit(allocator);
        }
        self.tokens.deinit(allocator);
        allocator.free(self.id);
        allocator.free(self.name);
    }
    
    pub fn addToken(self: *Place, allocator: Allocator, token: Token) !void {
        if (self.capacity) |cap| {
            if (self.tokens.items.len >= cap) {
                return error.PlaceCapacityExceeded;
            }
        }
        try self.tokens.append(allocator, token);
    }
    
    pub fn removeToken(self: *Place) ?Token {
        if (self.tokens.items.len == 0) return null;
        return self.tokens.orderedRemove(0);
    }
    
    pub fn tokenCount(self: *const Place) usize {
        return self.tokens.items.len;
    }
    
    pub fn hasTokens(self: *const Place) bool {
        return self.tokens.items.len > 0;
    }
};

/// ArcType defines the direction of token flow
pub const ArcType = enum {
    input,  // Place -> Transition
    output, // Transition -> Place
    inhibitor, // Place -o Transition (prevents firing if tokens present)
};

/// Arc connects places and transitions
pub const Arc = struct {
    id: []const u8,
    arc_type: ArcType,
    weight: usize, // Number of tokens required/produced
    source_id: []const u8,
    target_id: []const u8,
    
    pub fn init(allocator: Allocator, id: []const u8, arc_type: ArcType, weight: usize, source_id: []const u8, target_id: []const u8) !Arc {
        return Arc{
            .id = try allocator.dupe(u8, id),
            .arc_type = arc_type,
            .weight = weight,
            .source_id = try allocator.dupe(u8, source_id),
            .target_id = try allocator.dupe(u8, target_id),
        };
    }
    
    pub fn deinit(self: *Arc, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.source_id);
        allocator.free(self.target_id);
    }
};

/// TransitionGuard is a condition that must be true for transition to fire
pub const TransitionGuard = struct {
    expression: []const u8, // Simple boolean expression
    
    pub fn init(allocator: Allocator, expression: []const u8) !TransitionGuard {
        return TransitionGuard{
            .expression = try allocator.dupe(u8, expression),
        };
    }
    
    pub fn deinit(self: *TransitionGuard, allocator: Allocator) void {
        allocator.free(self.expression);
    }
    
    pub fn evaluate(self: *const TransitionGuard, tokens: []const Token) bool {
        // Simple guard evaluation - can be extended
        // For now, just check if expression is "true" or token count > 0
        if (std.mem.eql(u8, self.expression, "true")) return true;
        if (std.mem.eql(u8, self.expression, "has_tokens")) return tokens.len > 0;
        return false;
    }
};

/// Transition processes tokens (workflow actions)
pub const Transition = struct {
    id: []const u8,
    name: []const u8,
    guard: ?TransitionGuard,
    priority: i32, // Higher priority transitions fire first
    enabled: bool,
    
    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, priority: i32) !Transition {
        return Transition{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .guard = null,
            .priority = priority,
            .enabled = true,
        };
    }
    
    pub fn deinit(self: *Transition, allocator: Allocator) void {
        if (self.guard) |*g| {
            g.deinit(allocator);
        }
        allocator.free(self.id);
        allocator.free(self.name);
    }
    
    pub fn setGuard(self: *Transition, guard: TransitionGuard) void {
        self.guard = guard;
    }
};

/// Marking represents the current state of the Petri Net (token distribution)
pub const Marking = struct {
    place_tokens: std.StringHashMap(usize),
    
    pub fn init(allocator: Allocator) Marking {
        return Marking{
            .place_tokens = std.StringHashMap(usize).init(allocator),
        };
    }
    
    pub fn deinit(self: *Marking) void {
        self.place_tokens.deinit();
    }
    
    pub fn set(self: *Marking, place_id: []const u8, count: usize) !void {
        try self.place_tokens.put(place_id, count);
    }
    
    pub fn get(self: *const Marking, place_id: []const u8) usize {
        return self.place_tokens.get(place_id) orelse 0;
    }
    
    pub fn clone(self: *const Marking, allocator: Allocator) !Marking {
        var new_marking = Marking.init(allocator);
        var it = self.place_tokens.iterator();
        while (it.next()) |entry| {
            try new_marking.place_tokens.put(entry.key_ptr.*, entry.value_ptr.*);
        }
        return new_marking;
    }
    
    pub fn equals(self: *const Marking, other: *const Marking) bool {
        if (self.place_tokens.count() != other.place_tokens.count()) return false;
        
        var it = self.place_tokens.iterator();
        while (it.next()) |entry| {
            const other_count = other.place_tokens.get(entry.key_ptr.*) orelse return false;
            if (entry.value_ptr.* != other_count) return false;
        }
        return true;
    }
};

/// PetriNet is the main structure containing places, transitions, and arcs
pub const PetriNet = struct {
    allocator: Allocator,
    name: []const u8,
    places: std.StringHashMap(*Place),
    transitions: std.StringHashMap(*Transition),
    arcs: std.ArrayList(*Arc),
    next_token_id: u64,
    
    pub fn init(allocator: Allocator, name: []const u8) !PetriNet {
        return PetriNet{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .places = std.StringHashMap(*Place).init(allocator),
            .transitions = std.StringHashMap(*Transition).init(allocator),
            .arcs = .{},
            .next_token_id = 1,
        };
    }
    
    pub fn deinit(self: *PetriNet) void {
        // Free places
        var place_it = self.places.valueIterator();
        while (place_it.next()) |place_ptr| {
            place_ptr.*.deinit(self.allocator);
            self.allocator.destroy(place_ptr.*);
        }
        self.places.deinit();
        
        // Free transitions
        var trans_it = self.transitions.valueIterator();
        while (trans_it.next()) |trans_ptr| {
            trans_ptr.*.deinit(self.allocator);
            self.allocator.destroy(trans_ptr.*);
        }
        self.transitions.deinit();
        
        // Free arcs
        for (self.arcs.items) |arc| {
            arc.deinit(self.allocator);
            self.allocator.destroy(arc);
        }
        self.arcs.deinit(self.allocator);
        
        self.allocator.free(self.name);
    }
    
    /// Add a place to the Petri Net
    pub fn addPlace(self: *PetriNet, id: []const u8, name: []const u8, capacity: ?usize) !*Place {
        const place = try self.allocator.create(Place);
        errdefer self.allocator.destroy(place);
        place.* = try Place.init(self.allocator, id, name, capacity);
        // Use the duped id from Place as key (place.id points to allocated memory)
        try self.places.put(place.id, place);
        return place;
    }

    /// Add a transition to the Petri Net
    pub fn addTransition(self: *PetriNet, id: []const u8, name: []const u8, priority: i32) !*Transition {
        const transition = try self.allocator.create(Transition);
        errdefer self.allocator.destroy(transition);
        transition.* = try Transition.init(self.allocator, id, name, priority);
        // Use the duped id from Transition as key (transition.id points to allocated memory)
        try self.transitions.put(transition.id, transition);
        return transition;
    }
    
    /// Add an arc connecting a place and transition
    pub fn addArc(self: *PetriNet, id: []const u8, arc_type: ArcType, weight: usize, source_id: []const u8, target_id: []const u8) !*Arc {
        const arc = try self.allocator.create(Arc);
        arc.* = try Arc.init(self.allocator, id, arc_type, weight, source_id, target_id);
        try self.arcs.append(self.allocator, arc);
        return arc;
    }
    
    /// Add a token to a place
    pub fn addTokenToPlace(self: *PetriNet, place_id: []const u8, data: []const u8) !void {
        const place = self.places.get(place_id) orelse return error.PlaceNotFound;
        const data_copy = try self.allocator.dupe(u8, data);
        const token = Token.init(self.next_token_id, data_copy);
        self.next_token_id += 1;
        try place.addToken(self.allocator, token);
    }
    
    /// Get the current marking (state) of the Petri Net
    pub fn getCurrentMarking(self: *const PetriNet) !Marking {
        var marking = Marking.init(self.allocator);
        var it = self.places.iterator();
        while (it.next()) |entry| {
            try marking.set(entry.key_ptr.*, entry.value_ptr.*.tokenCount());
        }
        return marking;
    }
    
    /// Check if a transition is enabled (can fire)
    pub fn isTransitionEnabled(self: *const PetriNet, transition_id: []const u8) bool {
        const transition = self.transitions.get(transition_id) orelse return false;
        if (!transition.enabled) return false;
        
        // Check all input arcs
        for (self.arcs.items) |arc| {
            if (!std.mem.eql(u8, arc.target_id, transition_id)) continue;
            
            if (arc.arc_type == .input) {
                const place = self.places.get(arc.source_id) orelse return false;
                if (place.tokenCount() < arc.weight) return false;
            } else if (arc.arc_type == .inhibitor) {
                const place = self.places.get(arc.source_id) orelse return false;
                if (place.tokenCount() >= arc.weight) return false;
            }
        }
        
        // Check guard if present
        if (transition.guard) |guard| {
            // Collect input tokens for guard evaluation
            const TokenList = std.ArrayList(Token);
            var input_tokens = TokenList{};
            defer input_tokens.deinit(self.allocator);
            
            for (self.arcs.items) |arc| {
                if (!std.mem.eql(u8, arc.target_id, transition_id)) continue;
                if (arc.arc_type == .input) {
                    const place = self.places.get(arc.source_id) orelse continue;
                    for (place.tokens.items) |token| {
                        input_tokens.append(self.allocator, token) catch continue;
                    }
                }
            }
            
            if (!guard.evaluate(input_tokens.items)) return false;
        }
        
        return true;
    }
    
    /// Get all enabled transitions
    pub fn getEnabledTransitions(self: *const PetriNet) !std.ArrayList([]const u8) {
        const StringList = std.ArrayList([]const u8);
        var enabled = StringList{};
        
        var it = self.transitions.iterator();
        while (it.next()) |entry| {
            if (self.isTransitionEnabled(entry.key_ptr.*)) {
                try enabled.append(self.allocator, entry.key_ptr.*);
            }
        }
        
        return enabled;
    }
    
    /// Fire a transition (consume input tokens, produce output tokens)
    pub fn fireTransition(self: *PetriNet, transition_id: []const u8) !void {
        if (!self.isTransitionEnabled(transition_id)) {
            return error.TransitionNotEnabled;
        }
        
        // Collect tokens to remove and add
        const TokenDataList = std.ArrayList(struct { place_id: []const u8, token: Token });
        var tokens_to_add = TokenDataList{};
        defer {
            for (tokens_to_add.items) |item| {
                self.allocator.free(item.token.data);
            }
            tokens_to_add.deinit(self.allocator);
        }
        
        // Remove tokens from input places
        for (self.arcs.items) |arc| {
            if (!std.mem.eql(u8, arc.target_id, transition_id)) continue;
            if (arc.arc_type != .input) continue;
            
            const place = self.places.get(arc.source_id) orelse continue;
            var i: usize = 0;
            while (i < arc.weight) : (i += 1) {
                if (place.removeToken()) |token| {
                    // Store token for potential output
                    try tokens_to_add.append(self.allocator, .{ .place_id = arc.source_id, .token = token });
                }
            }
        }
        
        // Add tokens to output places
        for (self.arcs.items) |arc| {
            if (!std.mem.eql(u8, arc.source_id, transition_id)) continue;
            if (arc.arc_type != .output) continue;
            
            const place = self.places.get(arc.target_id) orelse continue;
            
            var i: usize = 0;
            while (i < arc.weight) : (i += 1) {
                // Create new token or clone from input
                const token_data = if (tokens_to_add.items.len > 0)
                    try self.allocator.dupe(u8, tokens_to_add.items[0].token.data)
                else
                    try self.allocator.dupe(u8, "{}");
                
                const token = Token.init(self.next_token_id, token_data);
                self.next_token_id += 1;
                try place.addToken(self.allocator, token);
            }
        }
    }
    
    /// Check if the Petri Net is in a deadlock state (no enabled transitions)
    pub fn isDeadlocked(self: *const PetriNet) bool {
        var it = self.transitions.iterator();
        while (it.next()) |entry| {
            if (self.isTransitionEnabled(entry.key_ptr.*)) return false;
        }
        return true;
    }
    
    /// Get statistics about the Petri Net
    pub fn getStats(self: *const PetriNet) PetriNetStats {
        var total_tokens: usize = 0;
        var place_it = self.places.valueIterator();
        while (place_it.next()) |place| {
            total_tokens += place.*.tokenCount();
        }
        
        return PetriNetStats{
            .place_count = self.places.count(),
            .transition_count = self.transitions.count(),
            .arc_count = self.arcs.items.len,
            .total_tokens = total_tokens,
        };
    }
};

pub const PetriNetStats = struct {
    place_count: usize,
    transition_count: usize,
    arc_count: usize,
    total_tokens: usize,
};

// ============================================================================
// TESTS
// ============================================================================

test "Token creation and cloning" {
    const allocator = std.testing.allocator;
    
    const data = try allocator.dupe(u8, "{\"key\": \"value\"}");
    var token = Token.init(1, data);
    defer token.deinit(allocator);
    
    try std.testing.expectEqual(@as(u64, 1), token.id);
    try std.testing.expectEqualStrings("{\"key\": \"value\"}", token.data);
    
    var cloned = try token.clone(allocator);
    defer cloned.deinit(allocator);
    try std.testing.expectEqual(token.id, cloned.id);
    try std.testing.expectEqualStrings(token.data, cloned.data);
}

test "Place token management" {
    const allocator = std.testing.allocator;
    
    var place = try Place.init(allocator, "p1", "Place 1", 3);
    defer place.deinit(allocator);
    
    // Add tokens
    const data1 = try allocator.dupe(u8, "token1");
    try place.addToken(allocator, Token.init(1, data1));
    try std.testing.expectEqual(@as(usize, 1), place.tokenCount());
    
    const data2 = try allocator.dupe(u8, "token2");
    try place.addToken(allocator, Token.init(2, data2));
    try std.testing.expectEqual(@as(usize, 2), place.tokenCount());
    
    // Test capacity
    const data3 = try allocator.dupe(u8, "token3");
    try place.addToken(allocator, Token.init(3, data3));
    
    const data4 = try allocator.dupe(u8, "token4");
    const result = place.addToken(allocator, Token.init(4, data4));
    try std.testing.expectError(error.PlaceCapacityExceeded, result);
    allocator.free(data4);
    
    // Remove token
    if (place.removeToken()) |token| {
        try std.testing.expectEqual(@as(u64, 1), token.id);
        var t = token;
        t.deinit(allocator);
    }
    try std.testing.expectEqual(@as(usize, 2), place.tokenCount());
}

test "Petri Net basic operations" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Test Net");
    defer net.deinit();
    
    // Add places
    _ = try net.addPlace("p1", "Input Place", null);
    _ = try net.addPlace("p2", "Output Place", null);
    
    // Add transition
    _ = try net.addTransition("t1", "Process", 0);
    
    // Add arcs
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    
    // Add tokens
    try net.addTokenToPlace("p1", "{}");
    
    // Check marking
    var marking = try net.getCurrentMarking();
    defer marking.deinit();
    try std.testing.expectEqual(@as(usize, 1), marking.get("p1"));
    try std.testing.expectEqual(@as(usize, 0), marking.get("p2"));
    
    // Check transition enabled
    try std.testing.expect(net.isTransitionEnabled("t1"));
    
    // Fire transition
    try net.fireTransition("t1");
    
    // Check new marking
    var marking2 = try net.getCurrentMarking();
    defer marking2.deinit();
    try std.testing.expectEqual(@as(usize, 0), marking2.get("p1"));
    try std.testing.expectEqual(@as(usize, 1), marking2.get("p2"));
}

test "Transition guard evaluation" {
    const allocator = std.testing.allocator;
    
    var guard = try TransitionGuard.init(allocator, "has_tokens");
    defer guard.deinit(allocator);
    
    const data = try allocator.dupe(u8, "test");
    var token = Token.init(1, data);
    defer token.deinit(allocator);
    
    const tokens = [_]Token{token};
    try std.testing.expect(guard.evaluate(&tokens));
}

test "Petri Net enabled transitions" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Test Net");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Place 1", null);
    _ = try net.addPlace("p2", "Place 2", null);
    _ = try net.addTransition("t1", "Trans 1", 0);
    _ = try net.addTransition("t2", "Trans 2", 0);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .input, 1, "p2", "t2");
    
    try net.addTokenToPlace("p1", "{}");
    
    var enabled = try net.getEnabledTransitions();
    defer enabled.deinit(allocator);
    
    try std.testing.expectEqual(@as(usize, 1), enabled.items.len);
    try std.testing.expectEqualStrings("t1", enabled.items[0]);
}

test "Petri Net deadlock detection" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Deadlock Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Place 1", null);
    _ = try net.addTransition("t1", "Trans 1", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    
    // No tokens, should be deadlocked
    try std.testing.expect(net.isDeadlocked());
    
    // Add token, should not be deadlocked
    try net.addTokenToPlace("p1", "{}");
    try std.testing.expect(!net.isDeadlocked());
}

test "Petri Net statistics" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Stats Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Place 1", null);
    _ = try net.addPlace("p2", "Place 2", null);
    _ = try net.addTransition("t1", "Trans 1", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    
    try net.addTokenToPlace("p1", "{}");
    try net.addTokenToPlace("p1", "{}");
    
    const stats = net.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.place_count);
    try std.testing.expectEqual(@as(usize, 1), stats.transition_count);
    try std.testing.expectEqual(@as(usize, 2), stats.arc_count);
    try std.testing.expectEqual(@as(usize, 2), stats.total_tokens);
}

test "Marking equality" {
    const allocator = std.testing.allocator;
    
    var m1 = Marking.init(allocator);
    defer m1.deinit();
    var m2 = Marking.init(allocator);
    defer m2.deinit();
    
    try m1.set("p1", 2);
    try m1.set("p2", 1);
    try m2.set("p1", 2);
    try m2.set("p2", 1);
    
    try std.testing.expect(m1.equals(&m2));
    
    try m2.set("p2", 2);
    try std.testing.expect(!m1.equals(&m2));
}

test "Inhibitor arc" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Inhibitor Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Input", null);
    _ = try net.addPlace("p2", "Inhibitor", null);
    _ = try net.addPlace("p3", "Output", null);
    _ = try net.addTransition("t1", "Trans", 0);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .inhibitor, 1, "p2", "t1");
    _ = try net.addArc("a3", .output, 1, "t1", "p3");
    
    // Add token to p1, transition should be enabled
    try net.addTokenToPlace("p1", "{}");
    try std.testing.expect(net.isTransitionEnabled("t1"));
    
    // Add token to inhibitor place, transition should be disabled
    try net.addTokenToPlace("p2", "{}");
    try std.testing.expect(!net.isTransitionEnabled("t1"));
}
