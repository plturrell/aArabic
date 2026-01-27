// stdlib module for zig-libc
// Phase 1.2 - Week 25
// Standard library functions

const std = @import("std");
const config = @import("config");

// Re-export submodules
pub const memory = @import("memory.zig");
pub const conversion = @import("conversion.zig");
pub const math = @import("math.zig");
const random_mod = @import("random.zig");
pub const sort = @import("sort.zig");
pub const environment = @import("environment.zig");
pub const process = @import("process.zig");
pub const time = @import("time.zig");
pub const string_util = @import("string_util.zig");
pub const numeric = @import("numeric.zig");
pub const utilities = @import("utilities.zig");
pub const arithmetic = @import("arithmetic.zig");

// Direct exports for C compatibility
// Memory allocation
pub const malloc = memory.malloc;
pub const free = memory.free;
pub const calloc = memory.calloc;
pub const realloc = memory.realloc;
pub const malloc_usable_size = memory.malloc_usable_size;
pub const memalign = memory.memalign;
pub const valloc = memory.valloc;
pub const cfree = memory.cfree;
pub const reallocf = memory.reallocf;

// String conversion
pub const atoi = conversion.atoi;
pub const atol = conversion.atol;
pub const atoll = conversion.atoll;
pub const atof = conversion.atof;
pub const strtol = conversion.strtol;
pub const strtoll = conversion.strtoll;
pub const strtoul = conversion.strtoul;
pub const strtod = conversion.strtod;
pub const strtof = conversion.strtof;

// Math functions
pub const abs = math.abs;
pub const labs = math.labs;
pub const llabs = math.llabs;
pub const div = math.div;
pub const ldiv = math.ldiv;
pub const lldiv = math.lldiv;
pub const div_t = math.div_t;
pub const ldiv_t = math.ldiv_t;
pub const lldiv_t = math.lldiv_t;

// Random number generation
pub const rand = random_mod.rand;
pub const srand = random_mod.srand;
pub const random = random_mod.random;
pub const srandom = random_mod.srandom;
pub const arc4random = random_mod.arc4random;
pub const arc4random_uniform = random_mod.arc4random_uniform;
pub const arc4random_buf = random_mod.arc4random_buf;
pub const rand_u64 = random_mod.rand_u64;
pub const rand_u32 = random_mod.rand_u32;
pub const rand_f64 = random_mod.rand_f64;
pub const rand_f32 = random_mod.rand_f32;
pub const uniform_u64 = random_mod.uniform_u64;
pub const uniform_f64 = random_mod.uniform_f64;
pub const normal = random_mod.normal;
pub const normal_params = random_mod.normal_params;
pub const lognormal = random_mod.lognormal;
pub const exponential = random_mod.exponential;
pub const gamma = random_mod.gamma;
pub const poisson = random_mod.poisson;

// Sorting and searching
pub const qsort = sort.qsort;
pub const bsearch = sort.bsearch;
pub const CompareFn = sort.CompareFn;

// Environment variables
pub const getenv = environment.getenv;
pub const setenv = environment.setenv;
pub const unsetenv = environment.unsetenv;
pub const putenv = environment.putenv;

// Process control
pub const exit = process.exit;
pub const atexit = process.atexit;
pub const abort = process.abort;
pub const _Exit = process._Exit;
pub const ExitHandlerFn = process.ExitHandlerFn;

// Time functions
pub const time_fn = time.time;
pub const difftime = time.difftime;
pub const gmtime = time.gmtime;
pub const localtime = time.localtime;
pub const mktime = time.mktime;
pub const ctime = time.ctime;
pub const tm = time.tm;

// String utilities
pub const strdup = string_util.strdup;
pub const strndup = string_util.strndup;
pub const strerror = string_util.strerror;
pub const strlcpy = string_util.strlcpy;
pub const strlcat = string_util.strlcat;
pub const strstr = string_util.strstr;
pub const strcspn = string_util.strcspn;
pub const strspn = string_util.strspn;
pub const strpbrk = string_util.strpbrk;
pub const strtok = string_util.strtok;
pub const strtok_r = string_util.strtok_r;
pub const strchr = string_util.strchr;
pub const strrchr = string_util.strrchr;

// Memory functions
pub const memmove = string_util.memmove;
pub const memchr = string_util.memchr;
pub const memrchr = string_util.memrchr;
pub const memset = string_util.memset;

// Numeric
pub const strtoull = numeric.strtoull;
pub const strtold = numeric.strtold;

// Utilities
pub const aligned_alloc = utilities.aligned_alloc;
pub const posix_memalign = utilities.posix_memalign;
pub const basename = utilities.basename;
pub const dirname = utilities.dirname;
pub const system = utilities.system;
pub const tmpnam = utilities.tmpnam;
pub const strtoimax = utilities.strtoimax;
pub const strtoumax = utilities.strtoumax;
pub const lsearch = utilities.lsearch;
pub const reallocarray = utilities.reallocarray;
pub const clearenv = utilities.clearenv;
pub const getloadavg = utilities.getloadavg;
pub const realpath = utilities.realpath;
pub const mkstemp = utilities.mkstemp;
pub const popen = utilities.popen;
pub const pclose = utilities.pclose;
pub const FILE = utilities.FILE;

// Arithmetic functions
pub const abs_arith = arithmetic.abs;
pub const labs_arith = arithmetic.labs;
pub const llabs_arith = arithmetic.llabs;
pub const div_arith = arithmetic.div;
pub const ldiv_arith = arithmetic.ldiv;
pub const lldiv_arith = arithmetic.lldiv;
pub const div_t_arith = arithmetic.div_t;
pub const ldiv_t_arith = arithmetic.ldiv_t;
pub const lldiv_t_arith = arithmetic.lldiv_t;
pub const qsort_arith = arithmetic.qsort;
pub const bsearch_arith = arithmetic.bsearch;
pub const CompareFn_arith = arithmetic.CompareFn;
pub const mblen = arithmetic.mblen;
pub const mbtowc = arithmetic.mbtowc;
pub const wctomb = arithmetic.wctomb;
pub const mbstowcs = arithmetic.mbstowcs;
pub const wcstombs = arithmetic.wcstombs;
