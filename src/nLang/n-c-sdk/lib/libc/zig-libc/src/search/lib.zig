// search module - Phase 1.22
// Real implementations of hash table, binary tree, and linear search
const std = @import("std");

pub const ENTRY = extern struct {
    key: ?[*:0]u8,
    data: ?*anyopaque,
};

pub const ACTION = enum(c_uint) {
    FIND = 0,
    ENTER = 1,
};

pub const VISIT = enum(c_uint) {
    preorder = 0,
    postorder = 1,
    endorder = 2,
    leaf = 3,
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// ============ Hash Table Implementation ============

const HashEntry = struct {
    entry: ENTRY,
    used: bool,
};

var hash_table: ?[]HashEntry = null;
var hash_size: usize = 0;

fn hashString(key: [*:0]const u8) usize {
    const str = std.mem.span(key);
    var hash: usize = 5381;
    for (str) |c| {
        hash = ((hash << 5) +% hash) +% c;
    }
    return hash;
}

/// Create hash table with at least nel entries
pub export fn hcreate(nel: usize) c_int {
    if (hash_table != null) return 0; // Already created

    // Allocate table (use next power of 2 for efficient modulo)
    const size = std.math.ceilPowerOfTwo(usize, nel) catch nel;
    hash_table = allocator.alloc(HashEntry, size) catch return 0;

    for (hash_table.?) |*entry| {
        entry.used = false;
        entry.entry.key = null;
        entry.entry.data = null;
    }
    hash_size = size;
    return 1;
}

/// Destroy hash table
pub export fn hdestroy() void {
    if (hash_table) |table| {
        allocator.free(table);
        hash_table = null;
        hash_size = 0;
    }
}

/// Search or insert into hash table
pub export fn hsearch(item: ENTRY, action: ACTION) ?*ENTRY {
    if (hash_table == null or hash_size == 0) return null;
    if (item.key == null) return null;

    const table = hash_table.?;
    const hash = hashString(item.key.?) % hash_size;
    var idx = hash;

    // Linear probing
    var attempts: usize = 0;
    while (attempts < hash_size) : (attempts += 1) {
        const entry = &table[idx];

        if (!entry.used) {
            // Empty slot
            if (action == .ENTER) {
                entry.entry = item;
                entry.used = true;
                return &entry.entry;
            }
            return null; // FIND and not found
        }

        // Check if keys match
        if (entry.entry.key) |existing_key| {
            const key_str = std.mem.span(item.key.?);
            const existing_str = std.mem.span(existing_key);
            if (std.mem.eql(u8, key_str, existing_str)) {
                return &entry.entry;
            }
        }

        idx = (idx + 1) % hash_size;
    }

    return null; // Table full
}

// ============ Queue Implementation ============

// Queue element structure (matches POSIX)
const QueueElem = extern struct {
    q_forw: ?*QueueElem,
    q_back: ?*QueueElem,
};

/// Insert element into queue after prev
pub export fn insque(elem: ?*anyopaque, prev: ?*anyopaque) void {
    if (elem == null) return;

    const e: *QueueElem = @ptrCast(@alignCast(elem));

    if (prev == null) {
        // Insert at head (circular list with self)
        e.q_forw = e;
        e.q_back = e;
    } else {
        const p: *QueueElem = @ptrCast(@alignCast(prev));
        e.q_forw = p.q_forw;
        e.q_back = p;
        if (p.q_forw) |next| {
            next.q_back = e;
        }
        p.q_forw = e;
    }
}

/// Remove element from queue
pub export fn remque(elem: ?*anyopaque) void {
    if (elem == null) return;

    const e: *QueueElem = @ptrCast(@alignCast(elem));

    if (e.q_back) |back| {
        back.q_forw = e.q_forw;
    }
    if (e.q_forw) |forw| {
        forw.q_back = e.q_back;
    }
}

// ============ Linear Search Implementation ============

/// Linear search, add if not found
pub export fn lsearch(key: ?*const anyopaque, base: ?*anyopaque, nelp: *usize, width: usize, compar: *const fn (?*const anyopaque, ?*const anyopaque) callconv(.C) c_int) ?*anyopaque {
    // First try to find
    if (lfind(key, base, nelp, width, compar)) |found| {
        return found;
    }

    // Not found - add at end
    if (base == null or width == 0) return null;

    const base_bytes: [*]u8 = @ptrCast(base.?);
    const new_pos = nelp.* * width;
    nelp.* += 1;

    // Copy key to new position
    if (key) |k| {
        const key_bytes: [*]const u8 = @ptrCast(k);
        @memcpy(base_bytes[new_pos..][0..width], key_bytes[0..width]);
    }

    return @ptrCast(base_bytes + new_pos);
}

/// Linear search (find only)
pub export fn lfind(key: ?*const anyopaque, base: ?*const anyopaque, nelp: *usize, width: usize, compar: *const fn (?*const anyopaque, ?*const anyopaque) callconv(.C) c_int) ?*anyopaque {
    if (key == null or base == null or width == 0) return null;

    const base_bytes: [*]const u8 = @ptrCast(base.?);
    const nel = nelp.*;

    var i: usize = 0;
    while (i < nel) : (i += 1) {
        const elem_ptr: *const anyopaque = @ptrCast(base_bytes + i * width);
        if (compar(key, elem_ptr) == 0) {
            return @constCast(@ptrCast(elem_ptr));
        }
    }

    return null;
}

// ============ Binary Tree Implementation ============

const TreeNode = struct {
    key: ?*const anyopaque,
    left: ?*TreeNode,
    right: ?*TreeNode,
};

/// Search binary tree, insert if not found
pub export fn tsearch(key: ?*const anyopaque, rootp: ?*?*anyopaque, compar: *const fn (?*const anyopaque, ?*const anyopaque) callconv(.C) c_int) ?*anyopaque {
    if (rootp == null) return null;

    const root_ptr: *?*TreeNode = @ptrCast(@alignCast(rootp.?));

    if (root_ptr.* == null) {
        // Empty tree, create root
        const node = allocator.create(TreeNode) catch return null;
        node.key = key;
        node.left = null;
        node.right = null;
        root_ptr.* = node;
        return @ptrCast(&node.key);
    }

    // Search tree
    var current = root_ptr.*;
    while (current) |node| {
        const cmp = compar(key, node.key);
        if (cmp == 0) {
            return @ptrCast(&node.key);
        } else if (cmp < 0) {
            if (node.left == null) {
                const new_node = allocator.create(TreeNode) catch return null;
                new_node.key = key;
                new_node.left = null;
                new_node.right = null;
                node.left = new_node;
                return @ptrCast(&new_node.key);
            }
            current = node.left;
        } else {
            if (node.right == null) {
                const new_node = allocator.create(TreeNode) catch return null;
                new_node.key = key;
                new_node.left = null;
                new_node.right = null;
                node.right = new_node;
                return @ptrCast(&new_node.key);
            }
            current = node.right;
        }
    }

    return null;
}

/// Find in binary tree (no insert)
pub export fn tfind(key: ?*const anyopaque, rootp: ?*const ?*anyopaque, compar: *const fn (?*const anyopaque, ?*const anyopaque) callconv(.C) c_int) ?*anyopaque {
    if (rootp == null) return null;

    const root_ptr: *const ?*TreeNode = @ptrCast(@alignCast(rootp.?));
    var current = root_ptr.*;

    while (current) |node| {
        const cmp = compar(key, node.key);
        if (cmp == 0) {
            return @ptrCast(@constCast(&node.key));
        } else if (cmp < 0) {
            current = node.left;
        } else {
            current = node.right;
        }
    }

    return null;
}

/// Delete from binary tree
pub export fn tdelete(key: ?*const anyopaque, rootp: ?*?*anyopaque, compar: *const fn (?*const anyopaque, ?*const anyopaque) callconv(.C) c_int) ?*anyopaque {
    if (rootp == null) return null;

    const root_ptr: *?*TreeNode = @ptrCast(@alignCast(rootp.?));
    return tdeleteNode(key, root_ptr, compar);
}

fn tdeleteNode(key: ?*const anyopaque, nodep: *?*TreeNode, compar: *const fn (?*const anyopaque, ?*const anyopaque) callconv(.C) c_int) ?*anyopaque {
    if (nodep.* == null) return null;

    const node = nodep.*.?;
    const cmp = compar(key, node.key);

    if (cmp < 0) {
        return tdeleteNode(key, &node.left, compar);
    } else if (cmp > 0) {
        return tdeleteNode(key, &node.right, compar);
    } else {
        // Found node to delete
        const parent_key = node.key;

        if (node.left == null) {
            nodep.* = node.right;
            allocator.destroy(node);
        } else if (node.right == null) {
            nodep.* = node.left;
            allocator.destroy(node);
        } else {
            // Two children - find inorder successor
            var successor_parent = node;
            var successor = node.right.?;
            while (successor.left) |left| {
                successor_parent = successor;
                successor = left;
            }
            node.key = successor.key;
            if (successor_parent == node) {
                node.right = successor.right;
            } else {
                successor_parent.left = successor.right;
            }
            allocator.destroy(successor);
        }

        return @ptrCast(@constCast(&parent_key));
    }
}

/// Walk binary tree in order
pub export fn twalk(root: ?*const anyopaque, action: *const fn (?*const anyopaque, VISIT, c_int) callconv(.C) void) void {
    if (root == null) return;

    const node: *const TreeNode = @ptrCast(@alignCast(root.?));
    twalkNode(node, action, 0);
}

fn twalkNode(node: *const TreeNode, action: *const fn (?*const anyopaque, VISIT, c_int) callconv(.C) void, level: c_int) void {
    if (node.left == null and node.right == null) {
        action(@ptrCast(&node.key), .leaf, level);
        return;
    }

    action(@ptrCast(&node.key), .preorder, level);

    if (node.left) |left| {
        twalkNode(left, action, level + 1);
    }

    action(@ptrCast(&node.key), .postorder, level);

    if (node.right) |right| {
        twalkNode(right, action, level + 1);
    }

    action(@ptrCast(&node.key), .endorder, level);
}

/// Destroy entire tree
pub export fn tdestroy(root: ?*anyopaque, free_node: *const fn (?*anyopaque) callconv(.C) void) void {
    if (root == null) return;

    const node: *TreeNode = @ptrCast(@alignCast(root.?));
    if (node.left) |left| {
        tdestroy(@ptrCast(left), free_node);
    }
    if (node.right) |right| {
        tdestroy(@ptrCast(right), free_node);
    }
    free_node(@ptrCast(@constCast(node.key)));
    allocator.destroy(node);
}
