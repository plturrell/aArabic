# SIMD/Vector - SIMD Operations Wrapper
# Days 40-41: Platform-aware vectorization and SIMD operations

from builtin import Int, Float64, Float32, Bool
from collections.list import List
from math import sqrt, sin, cos, exp, ln


# SIMD vector types

struct Vec4f:
    """4-wide float32 SIMD vector.
    
    Represents 4 float32 values that can be processed in parallel.
    
    Examples:
        ```mojo
        var v1 = Vec4f(1.0, 2.0, 3.0, 4.0)
        var v2 = Vec4f(5.0, 6.0, 7.0, 8.0)
        var sum = v1 + v2  # SIMD addition
        ```
    """
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32
    
    fn __init__(inout self, x: Float32, y: Float32, z: Float32, w: Float32):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    
    fn __init__(inout self, value: Float32):
        """Broadcast scalar to all lanes."""
        self.x = value
        self.y = value
        self.z = value
        self.w = value
    
    fn __add__(self, other: Vec4f) -> Vec4f:
        """SIMD addition."""
        return Vec4f(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w
        )
    
    fn __sub__(self, other: Vec4f) -> Vec4f:
        """SIMD subtraction."""
        return Vec4f(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w
        )
    
    fn __mul__(self, other: Vec4f) -> Vec4f:
        """SIMD multiplication."""
        return Vec4f(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
            self.w * other.w
        )
    
    fn __truediv__(self, other: Vec4f) -> Vec4f:
        """SIMD division."""
        return Vec4f(
            self.x / other.x,
            self.y / other.y,
            self.z / other.z,
            self.w / other.w
        )
    
    fn dot(self, other: Vec4f) -> Float32:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    
    fn length(self) -> Float32:
        """Vector length."""
        return sqrt(self.dot(self))
    
    fn normalize(self) -> Vec4f:
        """Normalize to unit length."""
        let len = self.length()
        return Vec4f(
            self.x / len,
            self.y / len,
            self.z / len,
            self.w / len
        )
    
    fn sum(self) -> Float32:
        """Sum of all lanes."""
        return self.x + self.y + self.z + self.w
    
    fn min(self) -> Float32:
        """Minimum value across lanes."""
        var result = self.x
        if self.y < result:
            result = self.y
        if self.z < result:
            result = self.z
        if self.w < result:
            result = self.w
        return result
    
    fn max(self) -> Float32:
        """Maximum value across lanes."""
        var result = self.x
        if self.y > result:
            result = self.y
        if self.z > result:
            result = self.z
        if self.w > result:
            result = self.w
        return result


struct Vec4i:
    """4-wide int32 SIMD vector."""
    var x: Int
    var y: Int
    var z: Int
    var w: Int
    
    fn __init__(inout self, x: Int, y: Int, z: Int, w: Int):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    
    fn __init__(inout self, value: Int):
        """Broadcast scalar to all lanes."""
        self.x = value
        self.y = value
        self.z = value
        self.w = value
    
    fn __add__(self, other: Vec4i) -> Vec4i:
        return Vec4i(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w
        )
    
    fn __sub__(self, other: Vec4i) -> Vec4i:
        return Vec4i(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w
        )
    
    fn __mul__(self, other: Vec4i) -> Vec4i:
        return Vec4i(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
            self.w * other.w
        )
    
    fn sum(self) -> Int:
        """Sum of all lanes."""
        return self.x + self.y + self.z + self.w


# Vectorized operations on arrays

fn vadd_f32(a: List[Float32], b: List[Float32]) -> List[Float32]:
    """Vectorized addition of float32 arrays.
    
    Args:
        a: First array
        b: Second array
    
    Returns:
        Element-wise sum using SIMD
    """
    let n = min(len(a), len(b))
    var result = List[Float32]()
    
    # Process 4 elements at a time
    var i = 0
    while i + 3 < n:
        let va = Vec4f(a[i], a[i+1], a[i+2], a[i+3])
        let vb = Vec4f(b[i], b[i+1], b[i+2], b[i+3])
        let vc = va + vb
        
        result.append(vc.x)
        result.append(vc.y)
        result.append(vc.z)
        result.append(vc.w)
        i += 4
    
    # Handle remaining elements
    while i < n:
        result.append(a[i] + b[i])
        i += 1
    
    return result


fn vmul_f32(a: List[Float32], b: List[Float32]) -> List[Float32]:
    """Vectorized multiplication of float32 arrays."""
    let n = min(len(a), len(b))
    var result = List[Float32]()
    
    var i = 0
    while i + 3 < n:
        let va = Vec4f(a[i], a[i+1], a[i+2], a[i+3])
        let vb = Vec4f(b[i], b[i+1], b[i+2], b[i+3])
        let vc = va * vb
        
        result.append(vc.x)
        result.append(vc.y)
        result.append(vc.z)
        result.append(vc.w)
        i += 4
    
    while i < n:
        result.append(a[i] * b[i])
        i += 1
    
    return result


fn vdot_f32(a: List[Float32], b: List[Float32]) -> Float32:
    """Vectorized dot product.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Dot product computed with SIMD
    """
    let n = min(len(a), len(b))
    var sum = 0.0
    
    var i = 0
    while i + 3 < n:
        let va = Vec4f(a[i], a[i+1], a[i+2], a[i+3])
        let vb = Vec4f(b[i], b[i+1], b[i+2], b[i+3])
        sum += va.dot(vb)
        i += 4
    
    while i < n:
        sum += a[i] * b[i]
        i += 1
    
    return sum


fn vsum_f32(a: List[Float32]) -> Float32:
    """Vectorized sum of array.
    
    Args:
        a: Array to sum
    
    Returns:
        Sum of all elements using SIMD
    """
    let n = len(a)
    var sum = 0.0
    
    var i = 0
    while i + 3 < n:
        let v = Vec4f(a[i], a[i+1], a[i+2], a[i+3])
        sum += v.sum()
        i += 4
    
    while i < n:
        sum += a[i]
        i += 1
    
    return sum


fn vmin_f32(a: List[Float32]) -> Float32:
    """Find minimum using SIMD.
    
    Args:
        a: Array
    
    Returns:
        Minimum value
    """
    if len(a) == 0:
        return 0.0
    
    var result = a[0]
    var i = 0
    
    while i + 3 < len(a):
        let v = Vec4f(a[i], a[i+1], a[i+2], a[i+3])
        let vmin = v.min()
        if vmin < result:
            result = vmin
        i += 4
    
    while i < len(a):
        if a[i] < result:
            result = a[i]
        i += 1
    
    return result


fn vmax_f32(a: List[Float32]) -> Float32:
    """Find maximum using SIMD."""
    if len(a) == 0:
        return 0.0
    
    var result = a[0]
    var i = 0
    
    while i + 3 < len(a):
        let v = Vec4f(a[i], a[i+1], a[i+2], a[i+3])
        let vmax = v.max()
        if vmax > result:
            result = vmax
        i += 4
    
    while i < len(a):
        if a[i] > result:
            result = a[i]
        i += 1
    
    return result


# Matrix operations with SIMD

fn matrix_multiply_4x4(a: List[Float32], b: List[Float32]) -> List[Float32]:
    """4x4 matrix multiplication using SIMD.
    
    Args:
        a: First matrix (16 elements, row-major)
        b: Second matrix (16 elements, row-major)
    
    Returns:
        Result matrix (16 elements)
    """
    var result = List[Float32]()
    
    for i in range(4):
        for j in range(4):
            # Compute dot product of row i from a with column j from b
            let row = Vec4f(
                a[i*4 + 0],
                a[i*4 + 1],
                a[i*4 + 2],
                a[i*4 + 3]
            )
            let col = Vec4f(
                b[0*4 + j],
                b[1*4 + j],
                b[2*4 + j],
                b[3*4 + j]
            )
            result.append(row.dot(col))
    
    return result


# Special vectorized math functions

fn vexp_f32(a: List[Float32]) -> List[Float32]:
    """Vectorized exponential function."""
    var result = List[Float32]()
    for i in range(len(a)):
        result.append(exp(a[i]))
    return result


fn vln_f32(a: List[Float32]) -> List[Float32]:
    """Vectorized natural logarithm."""
    var result = List[Float32]()
    for i in range(len(a)):
        result.append(ln(a[i]))
    return result


fn vsin_f32(a: List[Float32]) -> List[Float32]:
    """Vectorized sine function."""
    var result = List[Float32]()
    for i in range(len(a)):
        result.append(sin(a[i]))
    return result


fn vcos_f32(a: List[Float32]) -> List[Float32]:
    """Vectorized cosine function."""
    var result = List[Float32]()
    for i in range(len(a)):
        result.append(cos(a[i]))
    return result


fn vsqrt_f32(a: List[Float32]) -> List[Float32]:
    """Vectorized square root."""
    var result = List[Float32]()
    for i in range(len(a)):
        result.append(sqrt(a[i]))
    return result


# Utility functions

fn min(a: Int, b: Int) -> Int:
    return a if a < b else b


# ============================================================================
# Tests
# ============================================================================

test "vec4f creation":
    var v = Vec4f(1.0, 2.0, 3.0, 4.0)
    assert(v.x == 1.0)
    assert(v.w == 4.0)

test "vec4f broadcast":
    var v = Vec4f(5.0)
    assert(v.x == 5.0 and v.y == 5.0)

test "vec4f addition":
    var v1 = Vec4f(1.0, 2.0, 3.0, 4.0)
    var v2 = Vec4f(5.0, 6.0, 7.0, 8.0)
    var v3 = v1 + v2
    assert(v3.x == 6.0)
    assert(v3.w == 12.0)

test "vec4f multiplication":
    var v1 = Vec4f(2.0, 3.0, 4.0, 5.0)
    var v2 = Vec4f(1.0, 2.0, 3.0, 4.0)
    var v3 = v1 * v2
    assert(v3.x == 2.0)
    assert(v3.w == 20.0)

test "vec4f dot product":
    var v1 = Vec4f(1.0, 0.0, 0.0, 0.0)
    var v2 = Vec4f(1.0, 0.0, 0.0, 0.0)
    let dot = v1.dot(v2)
    assert(dot == 1.0)

test "vec4f sum":
    var v = Vec4f(1.0, 2.0, 3.0, 4.0)
    assert(v.sum() == 10.0)

test "vec4i operations":
    var v1 = Vec4i(1, 2, 3, 4)
    var v2 = Vec4i(5, 6, 7, 8)
    var v3 = v1 + v2
    assert(v3.sum() == 36)

test "vadd_f32 vectorized":
    var a = List[Float32]()
    var b = List[Float32]()
    for i in range(8):
        a.append(Float32(i))
        b.append(Float32(i * 2))
    
    let c = vadd_f32(a, b)
    assert(len(c) == 8)

test "vdot_f32 vectorized":
    var a = List[Float32]()
    var b = List[Float32]()
    for i in range(4):
        a.append(1.0)
        b.append(2.0)
    
    let dot = vdot_f32(a, b)
    assert(dot == 8.0)

test "vsum_f32 vectorized":
    var a = List[Float32]()
    for i in range(8):
        a.append(1.0)
    
    let sum = vsum_f32(a)
    assert(sum == 8.0)
