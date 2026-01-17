// Lock-Free Queues for Mojo Runtime
// Implements Chase-Lev work-stealing deque and MPSC queue

const std = @import("std");
const builtin = @import("builtin");
const Atomic = std.atomic.Value;

// ============================================================================
// Lock-Free Chase-Lev Deque (Work-Stealing)
// Single-Producer Multi-Consumer
// ============================================================================

pub fn WorkStealingDeque(comptime T: type) type {
    return struct {
        const Self = @This();
        const INITIAL_CAPACITY = 32;
        
        // Circular buffer array
        buffer: [*]Atomic(?T),
        capacity: usize,
        mask: usize,
        
        // Indices
        top: Atomic(isize),
        bottom: Atomic(isize),
        
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);
            const buffer = try allocator.alloc(Atomic(?T), INITIAL_CAPACITY);
            
            for (buffer) |*slot| {
                slot.* = Atomic(?T).init(null);
            }
            
            self.* = .{
                .buffer = buffer.ptr,
                .capacity = INITIAL_CAPACITY,
                .mask = INITIAL_CAPACITY - 1,
                .top = Atomic(isize).init(0),
                .bottom = Atomic(isize).init(0),
                .allocator = allocator,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            const slice = self.buffer[0..self.capacity];
            self.allocator.free(slice);
            self.allocator.destroy(self);
        }

        /// Push to the bottom (local thread only)
        pub fn push(self: *Self, item: T) !void {
            const b = self.bottom.load(.monotonic);
            const t = self.top.load(.acquire);
            
            const size = b - t;
            if (size >= self.capacity - 1) {
                // Resize would happen here in a full implementation
                // For now, just return error or blocking behavior logic could be added
                // Or simply expand
                try self.resize();
            }
            
            // Re-read after potential resize
            const buffer = self.buffer;
            const mask = self.mask;
            
            buffer[@as(usize, @intCast(b)) & mask].store(item, .monotonic);
            
            self.bottom.store(b + 1, .release);
        }

        /// Pop from the bottom (local thread only)
        pub fn pop(self: *Self) ?T {
            const b = self.bottom.load(.monotonic) - 1;
            self.bottom.store(b, .seq_cst);
            
            const t = self.top.load(.monotonic);
            
            if (t <= b) {
                // Non-empty queue
                const item = self.buffer[@as(usize, @intCast(b)) & self.mask].load(.monotonic);
                
                if (t == b) {
                    // Last item, need to handle potential race with steal
                    if (self.top.cmpxchgStrong(t, t + 1, .seq_cst, .monotonic)) |curr_top| {
                        // Failed race
                        _ = curr_top;
                        self.bottom.store(b + 1, .monotonic);
                        return null;
                    }
                    self.bottom.store(b + 1, .monotonic);
                    return item;
                }
                
                return item;
            } else {
                // Empty queue
                self.bottom.store(b + 1, .monotonic);
                return null;
            }
        }

        /// Steal from the top (other threads)
        pub fn steal(self: *Self) ?T {
            const t = self.top.load(.seq_cst);
            const b = self.bottom.load(.acquire);
            
            if (t >= b) return null;
            
            const item = self.buffer[@as(usize, @intCast(t)) & self.mask].load(.monotonic);
            
            if (self.top.cmpxchgStrong(t, t + 1, .seq_cst, .monotonic)) |curr_top| {
                // Failed race
                _ = curr_top;
                return null; // Lost race
            }
            
            return item;
        }
        
        fn resize(self: *Self) !void {
            // Simplified resize: double capacity
            const new_cap = self.capacity * 2;
            const new_buffer = try self.allocator.alloc(Atomic(?T), new_cap);
            
            for (new_buffer) |*slot| {
                slot.* = Atomic(?T).init(null);
            }
            
            const t = self.top.load(.monotonic);
            const b = self.bottom.load(.monotonic);
            
            for (0..@intCast(b - t)) |i| {
                const idx = @as(usize, @intCast(t)) + i;
                const item = self.buffer[idx & self.mask].load(.monotonic);
                new_buffer[idx & (new_cap - 1)].store(item, .monotonic);
            }
            
            const old_slice = self.buffer[0..self.capacity];
            self.allocator.free(old_slice);
            
            self.buffer = new_buffer.ptr;
            self.capacity = new_cap;
            self.mask = new_cap - 1;
        }
    };
}

// ============================================================================
// Lock-Free MPSC Queue (Global Injector)
// Multi-Producer Single-Consumer
// ============================================================================

pub fn MPSCQueue(comptime T: type) type {
    return struct {
        const Self = @This();
        
        const Node = struct {
            next: Atomic(?*Node),
            value: T,
        };
        
        head: Atomic(*Node),
        tail: *Node, // Only accessed by consumer
        stub: *Node, // Stub node
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);
            const stub = try allocator.create(Node);
            stub.next = Atomic(?*Node).init(null);
            // Value in stub is undefined/unused
            
            self.* = .{
                .head = Atomic(*Node).init(stub),
                .tail = stub,
                .stub = stub,
                .allocator = allocator,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            var current = self.tail;
            while (current.next.load(.monotonic)) |next| {
                self.allocator.destroy(current);
                current = next;
            }
            self.allocator.destroy(current); // Destroy last node
            self.allocator.destroy(self);
        }

        pub fn push(self: *Self, value: T) !void {
            const node = try self.allocator.create(Node);
            node.value = value;
            node.next = Atomic(?*Node).init(null);
            
            const prev = self.head.swap(node, .acq_rel);
            prev.next.store(node, .release);
        }

        pub fn pop(self: *Self) ?T {
            const tail = self.tail;
            const next = tail.next.load(.acquire);
            
            if (next) |next_node| {
                const val = next_node.value;
                self.tail = next_node;
                self.allocator.destroy(tail);
                return val;
            }
            
            return null;
        }
        
        pub fn isEmpty(self: *Self) bool {
            return self.tail.next.load(.monotonic) == null;
        }
    };
}
