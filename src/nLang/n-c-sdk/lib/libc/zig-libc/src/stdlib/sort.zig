// Sorting and searching functions for stdlib
// Phase 1.2 - Week 27
// Implements qsort, bsearch

const std = @import("std");

/// Comparison function type for qsort and bsearch
/// Returns: <0 if a<b, 0 if a==b, >0 if a>b
pub const CompareFn = *const fn (?*const anyopaque, ?*const anyopaque) callconv(.C) c_int;

/// Sort array using quicksort algorithm
/// C signature: void qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
pub export fn qsort(base: ?*anyopaque, nmemb: usize, size: usize, compar: CompareFn) void {
    if (base == null or nmemb <= 1 or size == 0) return;
    
    const bytes: [*]u8 = @ptrCast(base);
    quicksort(bytes, 0, nmemb - 1, size, compar);
}

fn quicksort(arr: [*]u8, low: usize, high: usize, size: usize, compar: CompareFn) void {
    if (low >= high) return;
    
    const pivot_idx = partition(arr, low, high, size, compar);
    
    if (pivot_idx > low) {
        quicksort(arr, low, pivot_idx - 1, size, compar);
    }
    if (pivot_idx < high) {
        quicksort(arr, pivot_idx + 1, high, size, compar);
    }
}

fn partition(arr: [*]u8, low: usize, high: usize, size: usize, compar: CompareFn) usize {
    const pivot = &arr[high * size];
    var i = low;
    
    var j = low;
    while (j < high) : (j += 1) {
        const elem = &arr[j * size];
        if (compar(elem, pivot) < 0) {
            swap(arr, i, j, size);
            i += 1;
        }
    }
    
    swap(arr, i, high, size);
    return i;
}

fn swap(arr: [*]u8, i: usize, j: usize, size: usize) void {
    if (i == j) return;
    
    const ai = &arr[i * size];
    const aj = &arr[j * size];
    
    var k: usize = 0;
    while (k < size) : (k += 1) {
        const temp = ai[k];
        ai[k] = aj[k];
        aj[k] = temp;
    }
}

/// Binary search in sorted array
/// C signature: void *bsearch(const void *key, const void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
pub export fn bsearch(key: ?*const anyopaque, base: ?*const anyopaque, nmemb: usize, size: usize, compar: CompareFn) ?*anyopaque {
    if (key == null or base == null or nmemb == 0 or size == 0) return null;
    
    const bytes: [*]const u8 = @ptrCast(base);
    var low: usize = 0;
    var high: usize = nmemb;
    
    while (low < high) {
        const mid = low + (high - low) / 2;
        const elem = &bytes[mid * size];
        const cmp = compar(key, elem);
        
        if (cmp < 0) {
            high = mid;
        } else if (cmp > 0) {
            low = mid + 1;
        } else {
            return @constCast(@ptrCast(elem));
        }
    }
    
    return null;
}
