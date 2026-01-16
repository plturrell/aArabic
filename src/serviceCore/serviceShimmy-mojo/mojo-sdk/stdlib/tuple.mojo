# Tuple - Fixed-size Heterogeneous Collections
# Day 31: Immutable sequences of mixed types

from builtin import Int, Bool, String, Float64


struct Tuple2[T1, T2]:
    """Two-element tuple (pair).
    
    A Tuple2 stores two values of potentially different types.
    Tuples are immutable and provide indexed access to their elements.
    
    Type Parameters:
        T1: Type of first element
        T2: Type of second element
    
    Examples:
        ```mojo
        let pair = Tuple2[Int, String](42, "hello")
        print(pair.first())   # 42
        print(pair.second())  # "hello"
        ```
    """
    
    var _0: T1
    var _1: T2
    
    fn __init__(inout self, first: T1, second: T2):
        """Initialize a 2-tuple.
        
        Args:
            first: First element
            second: Second element
        """
        self._0 = first
        self._1 = second
    
    fn first(self) -> T1:
        """Get the first element.
        
        Returns:
            First element
        """
        return self._0
    
    fn second(self) -> T2:
        """Get the second element.
        
        Returns:
            Second element
        """
        return self._1
    
    fn __getitem__(self, index: Int) -> T1 | T2:
        """Access element by index.
        
        Args:
            index: Element index (0 or 1)
        
        Returns:
            Element at index
        """
        if index == 0:
            return self._0
        return self._1
    
    fn __len__(self) -> Int:
        """Get tuple length.
        
        Returns:
            Always 2
        """
        return 2
    
    fn __eq__(self, other: Tuple2[T1, T2]) -> Bool:
        """Check equality.
        
        Args:
            other: Tuple to compare with
        
        Returns:
            True if elements are equal
        """
        return self._0 == other._0 and self._1 == other._1
    
    fn __str__(self) -> String:
        """Return string representation.
        
        Returns:
            String like "(42, 'hello')"
        """
        return "(" + str(self._0) + ", " + str(self._1) + ")"


struct Tuple3[T1, T2, T3]:
    """Three-element tuple (triple).
    
    Type Parameters:
        T1: Type of first element
        T2: Type of second element
        T3: Type of third element
    """
    
    var _0: T1
    var _1: T2
    var _2: T3
    
    fn __init__(inout self, first: T1, second: T2, third: T3):
        """Initialize a 3-tuple."""
        self._0 = first
        self._1 = second
        self._2 = third
    
    fn first(self) -> T1:
        return self._0
    
    fn second(self) -> T2:
        return self._1
    
    fn third(self) -> T3:
        return self._2
    
    fn __len__(self) -> Int:
        return 3
    
    fn __str__(self) -> String:
        return "(" + str(self._0) + ", " + str(self._1) + ", " + str(self._2) + ")"


struct Tuple4[T1, T2, T3, T4]:
    """Four-element tuple.
    
    Type Parameters:
        T1, T2, T3, T4: Types of elements
    """
    
    var _0: T1
    var _1: T2
    var _2: T3
    var _3: T4
    
    fn __init__(inout self, first: T1, second: T2, third: T3, fourth: T4):
        """Initialize a 4-tuple."""
        self._0 = first
        self._1 = second
        self._2 = third
        self._3 = fourth
    
    fn first(self) -> T1:
        return self._0
    
    fn second(self) -> T2:
        return self._1
    
    fn third(self) -> T3:
        return self._2
    
    fn fourth(self) -> T4:
        return self._3
    
    fn __len__(self) -> Int:
        return 4
    
    fn __str__(self) -> String:
        return "(" + str(self._0) + ", " + str(self._1) + ", " + 
               str(self._2) + ", " + str(self._3) + ")"


struct Tuple5[T1, T2, T3, T4, T5]:
    """Five-element tuple.
    
    Type Parameters:
        T1, T2, T3, T4, T5: Types of elements
    """
    
    var _0: T1
    var _1: T2
    var _2: T3
    var _3: T4
    var _4: T5
    
    fn __init__(inout self, v0: T1, v1: T2, v2: T3, v3: T4, v4: T5):
        """Initialize a 5-tuple."""
        self._0 = v0
        self._1 = v1
        self._2 = v2
        self._3 = v3
        self._4 = v4
    
    fn get(self, index: Int) -> T1 | T2 | T3 | T4 | T5:
        """Get element by index."""
        if index == 0:
            return self._0
        if index == 1:
            return self._1
        if index == 2:
            return self._2
        if index == 3:
            return self._3
        return self._4
    
    fn __len__(self) -> Int:
        return 5
    
    fn __str__(self) -> String:
        return "(" + str(self._0) + ", " + str(self._1) + ", " + 
               str(self._2) + ", " + str(self._3) + ", " + str(self._4) + ")"


# Specialized tuple types for common use cases

struct IntPair:
    """Tuple of two integers."""
    
    var first: Int
    var second: Int
    
    fn __init__(inout self, first: Int, second: Int):
        self.first = first
        self.second = second
    
    fn swap(inout self):
        """Swap the two elements."""
        let temp = self.first
        self.first = self.second
        self.second = temp
    
    fn sum(self) -> Int:
        """Return sum of elements."""
        return self.first + self.second
    
    fn product(self) -> Int:
        """Return product of elements."""
        return self.first * self.second
    
    fn min(self) -> Int:
        """Return minimum element."""
        return self.first if self.first < self.second else self.second
    
    fn max(self) -> Int:
        """Return maximum element."""
        return self.first if self.first > self.second else self.second
    
    fn __str__(self) -> String:
        return "(" + str(self.first) + ", " + str(self.second) + ")"


struct FloatPair:
    """Tuple of two floating-point numbers."""
    
    var first: Float64
    var second: Float64
    
    fn __init__(inout self, first: Float64, second: Float64):
        self.first = first
        self.second = second
    
    fn swap(inout self):
        """Swap the two elements."""
        let temp = self.first
        self.first = self.second
        self.second = temp
    
    fn sum(self) -> Float64:
        """Return sum of elements."""
        return self.first + self.second
    
    fn product(self) -> Float64:
        """Return product of elements."""
        return self.first * self.second
    
    fn magnitude(self) -> Float64:
        """Return magnitude (treating as 2D vector)."""
        return sqrt(self.first * self.first + self.second * self.second)
    
    fn __str__(self) -> String:
        return "(" + str(self.first) + ", " + str(self.second) + ")"


struct Point2D:
    """2D point (x, y coordinates)."""
    
    var x: Float64
    var y: Float64
    
    fn __init__(inout self, x: Float64, y: Float64):
        self.x = x
        self.y = y
    
    fn distance_to(self, other: Point2D) -> Float64:
        """Calculate distance to another point."""
        let dx = self.x - other.x
        let dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)
    
    fn translate(inout self, dx: Float64, dy: Float64):
        """Move point by offset."""
        self.x += dx
        self.y += dy
    
    fn scale(inout self, factor: Float64):
        """Scale point from origin."""
        self.x *= factor
        self.y *= factor
    
    fn __str__(self) -> String:
        return "Point2D(" + str(self.x) + ", " + str(self.y) + ")"


struct Point3D:
    """3D point (x, y, z coordinates)."""
    
    var x: Float64
    var y: Float64
    var z: Float64
    
    fn __init__(inout self, x: Float64, y: Float64, z: Float64):
        self.x = x
        self.y = y
        self.z = z
    
    fn distance_to(self, other: Point3D) -> Float64:
        """Calculate distance to another point."""
        let dx = self.x - other.x
        let dy = self.y - other.y
        let dz = self.z - other.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    
    fn magnitude(self) -> Float64:
        """Calculate distance from origin."""
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    fn normalize(inout self):
        """Normalize to unit vector."""
        let mag = self.magnitude()
        if mag > 0:
            self.x /= mag
            self.y /= mag
            self.z /= mag
    
    fn __str__(self) -> String:
        return "Point3D(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"


struct RGB:
    """RGB color tuple (red, green, blue)."""
    
    var red: Int
    var green: Int
    var blue: Int
    
    fn __init__(inout self, red: Int, green: Int, blue: Int):
        """Initialize RGB color.
        
        Args:
            red: Red component (0-255)
            green: Green component (0-255)
            blue: Blue component (0-255)
        """
        self.red = clamp(red, 0, 255)
        self.green = clamp(green, 0, 255)
        self.blue = clamp(blue, 0, 255)
    
    fn to_hex(self) -> String:
        """Convert to hex string (#RRGGBB)."""
        return "#" + int_to_hex(self.red) + int_to_hex(self.green) + int_to_hex(self.blue)
    
    fn brighten(inout self, amount: Int):
        """Increase brightness."""
        self.red = clamp(self.red + amount, 0, 255)
        self.green = clamp(self.green + amount, 0, 255)
        self.blue = clamp(self.blue + amount, 0, 255)
    
    fn darken(inout self, amount: Int):
        """Decrease brightness."""
        self.brighten(-amount)
    
    fn __str__(self) -> String:
        return "RGB(" + str(self.red) + ", " + str(self.green) + ", " + str(self.blue) + ")"


struct RGBA:
    """RGBA color tuple (red, green, blue, alpha)."""
    
    var red: Int
    var green: Int
    var blue: Int
    var alpha: Int
    
    fn __init__(inout self, red: Int, green: Int, blue: Int, alpha: Int = 255):
        """Initialize RGBA color.
        
        Args:
            red: Red component (0-255)
            green: Green component (0-255)
            blue: Blue component (0-255)
            alpha: Alpha component (0-255, default: 255)
        """
        self.red = clamp(red, 0, 255)
        self.green = clamp(green, 0, 255)
        self.blue = clamp(blue, 0, 255)
        self.alpha = clamp(alpha, 0, 255)
    
    fn to_rgb(self) -> RGB:
        """Convert to RGB (discarding alpha)."""
        return RGB(self.red, self.green, self.blue)
    
    fn with_alpha(self, alpha: Int) -> RGBA:
        """Create new color with different alpha."""
        return RGBA(self.red, self.green, self.blue, alpha)
    
    fn __str__(self) -> String:
        return "RGBA(" + str(self.red) + ", " + str(self.green) + ", " + 
               str(self.blue) + ", " + str(self.alpha) + ")"


# Utility functions

fn make_pair[T1, T2](first: T1, second: T2) -> Tuple2[T1, T2]:
    """Create a 2-tuple.
    
    Args:
        first: First element
        second: Second element
    
    Returns:
        New Tuple2
    """
    return Tuple2[T1, T2](first, second)


fn make_triple[T1, T2, T3](first: T1, second: T2, third: T3) -> Tuple3[T1, T2, T3]:
    """Create a 3-tuple.
    
    Args:
        first: First element
        second: Second element
        third: Third element
    
    Returns:
        New Tuple3
    """
    return Tuple3[T1, T2, T3](first, second, third)


fn swap_pair[T](inout pair: Tuple2[T, T]):
    """Swap elements of a homogeneous pair.
    
    Args:
        pair: Pair to swap (modified in-place)
    """
    let temp = pair.first()
    pair._0 = pair.second()
    pair._1 = temp


# Helper functions

fn clamp(value: Int, min_val: Int, max_val: Int) -> Int:
    """Clamp value to range [min_val, max_val]."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


fn int_to_hex(value: Int) -> String:
    """Convert integer to 2-digit hex string."""
    let hex_chars = "0123456789ABCDEF"
    let high = (value // 16) % 16
    let low = value % 16
    return String(hex_chars[high]) + String(hex_chars[low])


fn sqrt(x: Float64) -> Float64:
    """Square root (placeholder - would use math library)."""
    # Simplified Newton's method
    var guess = x / 2.0
    for _ in range(10):
        guess = (guess + x / guess) / 2.0
    return guess


fn min(a: Int, b: Int) -> Int:
    """Return minimum of two integers."""
    return a if a < b else b


fn max(a: Int, b: Int) -> Int:
    """Return maximum of two integers."""
    return a if a > b else b
