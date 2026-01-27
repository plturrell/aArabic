// Process Execution - Phase 1.4 Process Management
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

/// Execute program (path)
pub export fn execv(path: [*:0]const u8, argv: [*:null]const ?[*:0]const u8) c_int {
    const path_slice = std.mem.span(path);
    
    // Convert argv
    var args = std.ArrayList([]const u8).init(std.heap.page_allocator);
    defer args.deinit();
    
    var i: usize = 0;
    while (argv[i]) |arg| : (i += 1) {
        args.append(std.mem.span(arg)) catch {
            setErrno(.NOMEM);
            return -1;
        };
    }
    
    std.posix.execve(path_slice, args.items, &[_][]const u8{}) catch |err| {
        setErrno(switch (err) {
            error.FileNotFound => .NOENT,
            error.AccessDenied => .ACCES,
            error.IsDir => .ACCES,
            error.NotDir => .NOTDIR,
            error.NameTooLong => .NAMETOOLONG,
            else => .INVAL,
        });
        return -1;
    };
    
    unreachable;
}

/// Execute program (path + environment)
pub export fn execve(path: [*:0]const u8, argv: [*:null]const ?[*:0]const u8, envp: [*:null]const ?[*:0]const u8) c_int {
    const path_slice = std.mem.span(path);
    
    // Convert argv
    var args = std.ArrayList([]const u8).init(std.heap.page_allocator);
    defer args.deinit();
    
    var i: usize = 0;
    while (argv[i]) |arg| : (i += 1) {
        args.append(std.mem.span(arg)) catch {
            setErrno(.NOMEM);
            return -1;
        };
    }
    
    // Convert envp
    var env = std.ArrayList([]const u8).init(std.heap.page_allocator);
    defer env.deinit();
    
    i = 0;
    while (envp[i]) |e| : (i += 1) {
        env.append(std.mem.span(e)) catch {
            setErrno(.NOMEM);
            return -1;
        };
    }
    
    std.posix.execve(path_slice, args.items, env.items) catch |err| {
        setErrno(switch (err) {
            error.FileNotFound => .NOENT,
            error.AccessDenied => .ACCES,
            error.IsDir => .ACCES,
            else => .INVAL,
        });
        return -1;
    };
    
    unreachable;
}

/// Execute program (filename searched in PATH)
pub export fn execvp(file: [*:0]const u8, argv: [*:null]const ?[*:0]const u8) c_int {
    const file_slice = std.mem.span(file);
    
    // If contains /, use as path
    if (std.mem.indexOf(u8, file_slice, "/")) |_| {
        return execv(file, argv);
    }
    
    // Search in PATH
    const path_env = std.posix.getenv("PATH") orelse "/usr/local/bin:/usr/bin:/bin";
    
    var it = std.mem.split(u8, path_env, ":");
    while (it.next()) |dir| {
        var path_buf: [std.fs.MAX_PATH_BYTES]u8 = undefined;
        const full_path = std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, file_slice }) catch continue;
        
        // Try to exec
        const result = execv(@ptrCast(full_path.ptr), argv);
        if (result == 0) return result;
    }
    
    setErrno(.NOENT);
    return -1;
}

/// Execute program with environment (filename in PATH)
pub export fn execvpe(file: [*:0]const u8, argv: [*:null]const ?[*:0]const u8, envp: [*:null]const ?[*:0]const u8) c_int {
    const file_slice = std.mem.span(file);
    
    if (std.mem.indexOf(u8, file_slice, "/")) |_| {
        return execve(file, argv, envp);
    }
    
    const path_env = std.posix.getenv("PATH") orelse "/usr/local/bin:/usr/bin:/bin";
    
    var it = std.mem.split(u8, path_env, ":");
    while (it.next()) |dir| {
        var path_buf: [std.fs.MAX_PATH_BYTES]u8 = undefined;
        const full_path = std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, file_slice }) catch continue;
        
        const result = execve(@ptrCast(full_path.ptr), argv, envp);
        if (result == 0) return result;
    }
    
    setErrno(.NOENT);
    return -1;
}

/// Execute program (varargs)
pub export fn execl(path: [*:0]const u8, arg0: [*:0]const u8, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    
    // Build argv array
    var argv = std.ArrayList(?[*:0]const u8).init(std.heap.page_allocator);
    defer argv.deinit();
    
    // First arg provided
    argv.append(arg0) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    // Parse remaining varargs until null
    while (true) {
        const arg = @cVaArg(&args, ?[*:0]const u8);
        argv.append(arg) catch {
            setErrno(.NOMEM);
            return -1;
        };
        if (arg == null) break;
    }
    
    return execv(path, @ptrCast(argv.items.ptr));
}

/// Execute program with environment (varargs)
pub export fn execle(path: [*:0]const u8, arg0: [*:0]const u8, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    
    // Build argv array
    var argv = std.ArrayList(?[*:0]const u8).init(std.heap.page_allocator);
    defer argv.deinit();
    
    argv.append(arg0) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    // Parse args until null
    var envp: [*:null]const ?[*:0]const u8 = undefined;
    while (true) {
        const arg = @cVaArg(&args, ?[*:0]const u8);
        if (arg == null) {
            // Next vararg is envp
            envp = @cVaArg(&args, [*:null]const ?[*:0]const u8);
            break;
        }
        argv.append(arg) catch {
            setErrno(.NOMEM);
            return -1;
        };
    }
    argv.append(null) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    return execve(path, @ptrCast(argv.items.ptr), envp);
}

/// Execute program in PATH (varargs)
pub export fn execlp(file: [*:0]const u8, arg0: [*:0]const u8, ...) c_int {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    
    // Build argv array
    var argv = std.ArrayList(?[*:0]const u8).init(std.heap.page_allocator);
    defer argv.deinit();
    
    argv.append(arg0) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    // Parse remaining varargs until null
    while (true) {
        const arg = @cVaArg(&args, ?[*:0]const u8);
        argv.append(arg) catch {
            setErrno(.NOMEM);
            return -1;
        };
        if (arg == null) break;
    }
    
    return execvp(file, @ptrCast(argv.items.ptr));
}

/// Execute file descriptor
pub export fn fexecve(fd: c_int, argv: [*:null]const ?[*:0]const u8, envp: [*:null]const ?[*:0]const u8) c_int {
    // Get path from fd
    var path_buf: [std.fs.MAX_PATH_BYTES]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/proc/self/fd/{d}", .{fd}) catch {
        setErrno(.BADF);
        return -1;
    };
    
    return execve(@ptrCast(path.ptr), argv, envp);
}

/// System - execute shell command
pub export fn system(command: ?[*:0]const u8) c_int {
    if (command == null) {
        // Check if shell is available
        return 1;
    }
    
    const cmd = std.mem.span(command.?);
    
    // Fork and exec
    const pid = std.posix.fork() catch {
        setErrno(.AGAIN);
        return -1;
    };
    
    if (pid == 0) {
        // Child: exec shell
        var argv = [_]?[*:0]const u8{ "/bin/sh", "-c", command, null };
        _ = execv("/bin/sh", &argv);
        std.posix.exit(127);
    }
    
    // Parent: wait for child
    var status: c_int = 0;
    _ = std.posix.waitpid(pid, 0) catch {
        return -1;
    };
    
    return status;
}

/// Posix spawn (simplified)
pub export fn posix_spawn(pid: *c_int, path: [*:0]const u8, file_actions: ?*const anyopaque, attrp: ?*const anyopaque, argv: [*:null]const ?[*:0]const u8, envp: [*:null]const ?[*:0]const u8) c_int {
    _ = file_actions;
    _ = attrp;
    
    const child_pid = std.posix.fork() catch {
        setErrno(.AGAIN);
        return -1;
    };
    
    if (child_pid == 0) {
        _ = execve(path, argv, envp);
        std.posix.exit(127);
    }
    
    pid.* = @intCast(child_pid);
    return 0;
}

/// Posix spawnp (search PATH)
pub export fn posix_spawnp(pid: *c_int, file: [*:0]const u8, file_actions: ?*const anyopaque, attrp: ?*const anyopaque, argv: [*:null]const ?[*:0]const u8, envp: [*:null]const ?[*:0]const u8) c_int {
    _ = file_actions;
    _ = attrp;
    
    const child_pid = std.posix.fork() catch {
        setErrno(.AGAIN);
        return -1;
    };
    
    if (child_pid == 0) {
        _ = execvpe(file, argv, envp);
        std.posix.exit(127);
    }
    
    pid.* = @intCast(child_pid);
    return 0;
}
