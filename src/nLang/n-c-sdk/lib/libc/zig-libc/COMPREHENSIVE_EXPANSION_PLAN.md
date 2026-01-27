# Comprehensive Module Expansion Plan for n-c-sdk zig-libc

## Overview
This document outlines the complete expansion plan for all zig-libc modules to achieve full POSIX compliance and C standard library compatibility.

## Current Status Summary

### ✅ Phase 1.1: Foundation (COMPLETE)
- Core modules: errno, assert, limits, string, ctype
- **60 functions implemented**

### ✅ Phase 1.2: Stdlib Expansion (COMPLETE)  
- New modules: string_util, environment, random, arithmetic
- **46 functions added**

### ⏳ Phase 1.3: System Interfaces (IN PROGRESS)
- unistd.h: ✅ Expanded (+20 functions)
- fcntl.h: ✅ Complete rewrite (+11 functions)
- sys/stat.h: ⚠️ Stubs need real implementations
- dirent.h: ⚠️ Stubs need real implementations  
- sys/mman.h: ⚠️ Stubs need real implementations

---

## Module-by-Module Expansion Plan

### Priority 1: Core System Interfaces (Phase 1.3 Completion)

#### 1. dirent.h - Directory Operations
**Status**: Stubbed (13 functions)
**Needs**: Real DIR structure with fd tracking, proper readdir implementation

**Functions to implement**:
- `opendir` - Open directory stream with real fd
- `fdopendir` - Convert fd to DIR*
- `closedir` - Close and free DIR structure
- `readdir` - Read next directory entry (use getdents64 syscall)
- `readdir_r` - Thread-safe readdir
- `rewinddir` - Reset directory stream
- `seekdir` / `telldir` - Position management
- `dirfd` - Extract fd from DIR*
- `scandir` - Scan directory with filter
- `alphasort` / `versionsort` - Sort functions

**Implementation approach**: Create internal DIR structure wrapping file descriptor, use std.posix.system.getdents64

#### 2. sys/mman.h - Memory Management
**Status**: Stubbed (~15 functions)
**Needs**: Real mmap/munmap implementations

**Functions to implement**:
- `mmap` - Map files/devices into memory
- `munmap` - Unmap memory
- `mprotect` - Set protection on memory region
- `msync` - Synchronize mapped memory
- `mlock` / `munlock` - Lock/unlock memory
- `mlockall` / `munlockall` - Lock/unlock all memory
- `madvise` - Give advice about memory usage
- `mincore` - Determine if pages are resident
- `mremap` - Remap virtual memory (Linux-specific)
- `mmap2` - mmap with page-aligned offset

**Implementation approach**: Use std.posix.system.mmap, mprotect, etc.

#### 3. sys/stat.h - File Status (Real Implementations)
**Status**: Stubbed (17 functions)
**Needs**: Replace stubs with real syscalls

**Functions to fix**:
- `stat` / `fstat` / `lstat` / `fstatat` - Use real syscalls
- `mkdir` / `mkdirat` - Real directory creation
- `mknod` / `mknodat` - Real special file creation
- `mkfifo` / `mkfifoat` - Real FIFO creation
- `chmod` / `fchmod` / `fchmodat` - Real permission changes
- `umask` - Real mask management
- `futimens` / `utimensat` - Real timestamp updates

**Implementation approach**: Use std.posix.system.fstatat, mkdirat, etc.

---

### Priority 2: Standard I/O (Phase 1.4)

#### 4. stdio.h - Standard I/O
**Status**: Partially implemented (~80 functions total)
**Needs**: Complete FILE* operations, formatted I/O

**Categories**:

**A. File Operations** (15 functions):
- `fopen` / `freopen` / `fdopen` - Open streams
- `fclose` / `fflush` - Close/flush
- `setbuf` / `setvbuf` - Buffering control
- `fwide` - Set stream orientation
- `freopen64`, `fopen64` - Large file support

**B. Character I/O** (12 functions):
- `fgetc` / `getc` / `getchar` - Read character
- `fgets` / `gets` - Read string  
- `fputc` / `putc` / `putchar` - Write character
- `fputs` / `puts` - Write string
- `ungetc` - Push back character

**C. Formatted I/O** (20 functions):
- `printf` family: `printf`, `fprintf`, `sprintf`, `snprintf`, `dprintf`
- `vprintf` family: `vprintf`, `vfprintf`, `vsprintf`, `vsnprintf`, `vdprintf`
- `scanf` family: `scanf`, `fscanf`, `sscanf`
- `vscanf` family: `vscanf`, `vfscanf`, `vsscanf`

**D. Binary I/O** (4 functions):
- `fread` / `fwrite` - Binary read/write
- `feof` / `ferror` / `clearerr` - Status checks

**E. Positioning** (8 functions):
- `fseek` / `ftell` / `rewind` - Position operations
- `fgetpos` / `fsetpos` - Position save/restore
- `fseeko` / `ftello` - Large file positioning

**F. File Management** (8 functions):
- `remove` / `rename` / `renameat` - File operations
- `tmpfile` / `tmpnam` / `tempnam` - Temporary files
- `mkstemp` / `mkdtemp` - Secure temporary files

**Implementation approach**: Create internal FILE structure with buffering, use std.fmt for formatting

---

### Priority 3: Math Library (Phase 1.5)

#### 5. math.h - Mathematics
**Status**: Stubbed (~100+ functions)
**Needs**: Real implementations for all math functions

**Categories**:

**A. Trigonometric** (18 functions):
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Float/long double variants: `sinf`, `sinl`, etc.

**B. Exponential/Logarithmic** (18 functions):
- `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`
- `pow`, `sqrt`, `cbrt`, `hypot`
- Float/long double variants

**C. Rounding/Remainder** (15 functions):
- `ceil`, `floor`, `trunc`, `round`, `nearbyint`
- `fmod`, `remainder`, `remquo`
- Float/long double variants

**D. Special Functions** (12 functions):
- `erf`, `erfc`, `gamma`, `lgamma`, `tgamma`
- `j0`, `j1`, `jn`, `y0`, `y1`, `yn` (Bessel functions)

**E. Classification** (10 functions):
- `fpclassify`, `isfinite`, `isinf`, `isnan`, `isnormal`
- `signbit`, `isgreater`, `isless`, etc.

**F. Manipulation** (8 functions):
- `frexp`, `ldexp`, `modf`, `scalbn`, `logb`
- `nextafter`, `copysign`, `nan`

**Implementation approach**: Use Zig's std.math where available, implement missing functions

---

### Priority 4: Threading & Synchronization (Phase 1.6)

#### 6. pthread.h - POSIX Threads
**Status**: Stubbed (~60 functions)
**Needs**: Real thread management

**Categories**:

**A. Thread Management** (8 functions):
- `pthread_create` / `pthread_exit` / `pthread_join` / `pthread_detach`
- `pthread_self` / `pthread_equal`
- `pthread_cancel` / `pthread_setcancelstate`

**B. Mutexes** (12 functions):
- `pthread_mutex_init` / `pthread_mutex_destroy`
- `pthread_mutex_lock` / `pthread_mutex_trylock` / `pthread_mutex_unlock`
- `pthread_mutex_timedlock`
- Mutex attributes: `pthread_mutexattr_*`

**C. Condition Variables** (10 functions):
- `pthread_cond_init` / `pthread_cond_destroy`
- `pthread_cond_wait` / `pthread_cond_timedwait`
- `pthread_cond_signal` / `pthread_cond_broadcast`
- Cond attributes: `pthread_condattr_*`

**D. Read-Write Locks** (10 functions):
- `pthread_rwlock_*` family

**E. Barriers** (5 functions):
- `pthread_barrier_*` family

**F. Thread-Specific Data** (5 functions):
- `pthread_key_create` / `pthread_key_delete`
- `pthread_setspecific` / `pthread_getspecific`

**G. Thread Attributes** (10 functions):
- `pthread_attr_*` family

**Implementation approach**: Use std.Thread, std.Thread.Mutex, std.Thread.Condition

---

### Priority 5: Networking (Phase 1.7)

#### 7. sys/socket.h - Sockets
**Status**: Stubbed (~30 functions)

**Functions**:
- `socket` / `bind` / `listen` / `accept` / `connect`
- `send` / `recv` / `sendto` / `recvfrom`
- `sendmsg` / `recvmsg` / `sendmmsg` / `recvmmsg`
- `getsockopt` / `setsockopt`
- `shutdown` / `socketpair`

#### 8. netinet/in.h & arpa/inet.h - Internet Protocols
**Status**: Stubbed (~15 functions)

**Functions**:
- `inet_aton` / `inet_ntoa` / `inet_pton` / `inet_ntop`
- `htons` / `htonl` / `ntohs` / `ntohl`

#### 9. netdb.h - Network Database
**Status**: Stubbed (~20 functions)

**Functions**:
- `gethostbyname` / `gethostbyaddr`
- `getaddrinfo` / `freeaddrinfo` / `getnameinfo`
- `getservbyname` / `getservbyport`
- `getprotobyname` / `getprotobynumber`

**Implementation approach**: Use std.net, std.posix.system socket functions

---

### Priority 6: Signal Handling (Phase 1.8)

#### 10. signal.h - Signals
**Status**: Stubbed (~25 functions)

**Functions**:
- `signal` / `sigaction` / `kill` / `raise`
- `sigprocmask` / `sigsuspend` / `sigpending`
- `sigaltstack` / `sigqueue`
- `sigset_t` operations: `sigemptyset`, `sigfillset`, `sigaddset`, `sigdelset`

**Implementation approach**: Use std.posix.system signal functions

---

### Priority 7: Time & Date (Phase 1.9)

#### 11. time.h - Time Operations
**Status**: Partially implemented (~30 functions)

**Functions**:
- `time` / `difftime` / `mktime`
- `localtime` / `gmtime` / `ctime` / `asctime`
- `strftime` / `strptime`
- `clock` / `clock_gettime` / `clock_settime`
- `nanosleep` / `timer_create` / `timer_settime`

#### 12. sys/time.h - Time Structures
**Status**: Stubbed (~10 functions)

**Functions**:
- `gettimeofday` / `settimeofday`
- `getitimer` / `setitimer`
- `timeval` / `timezone` operations

**Implementation approach**: Use std.time, std.posix.system time functions

---

### Priority 8: Advanced Features (Phase 1.10)

#### 13. regex.h - Regular Expressions
**Status**: Stubbed (~8 functions)

**Functions**:
- `regcomp` / `regexec` / `regerror` / `regfree`

#### 14. locale.h - Localization
**Status**: Stubbed (~10 functions)

**Functions**:
- `setlocale` / `localeconv`
- `nl_langinfo` / `newlocale` / `uselocale`

#### 15. wchar.h - Wide Characters
**Status**: Stubbed (~50 functions)

**Functions**:
- Wide character I/O
- Wide string operations
- Multibyte conversions

---

### Priority 9: System-Specific (Phase 1.11)

#### 16. sys/wait.h - Process Wait
**Status**: Stubbed (~8 functions)

#### 17. sys/resource.h - Resource Limits
**Status**: Stubbed (~10 functions)

#### 18. poll.h - I/O Multiplexing
**Status**: Stubbed (~5 functions)

#### 19. sys/select.h - Select
**Status**: Stubbed (~5 functions)

#### 20. termios.h - Terminal I/O
**Status**: Stubbed (~20 functions)

---

## Implementation Strategy

### Phase Approach
1. **Phase 1.3**: Complete system interfaces (dirent, mman, stat fixes)
2. **Phase 1.4**: Implement stdio.h completely
3. **Phase 1.5**: Implement math.h
4. **Phase 1.6**: Implement pthread.h
5. **Phase 1.7**: Implement networking
6. **Phase 1.8**: Implement signals
7. **Phase 1.9**: Implement time/date
8. **Phase 1.10**: Implement advanced features
9. **Phase 1.11**: Implement remaining system-specific modules

### Testing Strategy
- Unit tests for each module
- Integration tests for complex interactions
- POSIX compliance test suite
- Cross-platform validation (Linux, macOS, BSD)

### Documentation Strategy
- API reference for each function
- Implementation notes
- Platform-specific behavior
- Migration guides

---

## Estimated Scope

**Total Functions to Implement**: ~600-800
**Current Progress**: ~154 functions (Phase 1.1 + 1.2 + 1.3 partial)
**Remaining Work**: ~450-650 functions

**By Priority**:
- Priority 1 (System Interfaces): ~45 functions
- Priority 2 (stdio): ~80 functions
- Priority 3 (math): ~100 functions
- Priority 4 (pthread): ~60 functions
- Priority 5 (networking): ~65 functions
- Priority 6 (signals): ~25 functions
- Priority 7 (time): ~40 functions
- Priority 8 (advanced): ~70 functions
- Priority 9 (system-specific): ~70 functions

---

## Next Steps

1. Complete Priority 1 (dirent, mman, stat real implementations)
2. Move to Priority 2 (stdio complete implementation)
3. Continue sequentially through priorities
4. Test and validate each priority before moving to next
5. Update documentation as we go

This is a multi-month effort requiring systematic implementation of hundreds of functions.
