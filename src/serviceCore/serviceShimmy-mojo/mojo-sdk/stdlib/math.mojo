# Math - Mathematical Functions
# Day 33: Trigonometry, exponentials, logarithms, and more

from builtin import Int, Float64, Bool


# Mathematical constants

let PI: Float64 = 3.141592653589793
let E: Float64 = 2.718281828459045
let TAU: Float64 = 6.283185307179586  # 2 * PI
let SQRT_2: Float64 = 1.4142135623730951
let SQRT_3: Float64 = 1.7320508075688772
let LN_2: Float64 = 0.6931471805599453
let LN_10: Float64 = 2.302585092994046
let LOG2_E: Float64 = 1.4426950408889634
let LOG10_E: Float64 = 0.4342944819032518


# Basic arithmetic functions

fn abs(x: Float64) -> Float64:
    """Absolute value.
    
    Args:
        x: Input value
    
    Returns:
        |x|
    """
    return x if x >= 0.0 else -x


fn abs_int(x: Int) -> Int:
    """Absolute value for integers.
    
    Args:
        x: Input value
    
    Returns:
        |x|
    """
    return x if x >= 0 else -x


fn min(a: Float64, b: Float64) -> Float64:
    """Minimum of two values.
    
    Args:
        a: First value
        b: Second value
    
    Returns:
        min(a, b)
    """
    return a if a < b else b


fn max(a: Float64, b: Float64) -> Float64:
    """Maximum of two values.
    
    Args:
        a: First value
        b: Second value
    
    Returns:
        max(a, b)
    """
    return a if a > b else b


fn clamp(x: Float64, min_val: Float64, max_val: Float64) -> Float64:
    """Clamp value to range [min_val, max_val].
    
    Args:
        x: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clamped value
    """
    return max(min_val, min(x, max_val))


fn sign(x: Float64) -> Float64:
    """Sign function.
    
    Args:
        x: Input value
    
    Returns:
        -1.0 if x < 0, 0.0 if x == 0, 1.0 if x > 0
    """
    if x < 0.0:
        return -1.0
    if x > 0.0:
        return 1.0
    return 0.0


# Power and root functions

fn sqrt(x: Float64) -> Float64:
    """Square root using Newton's method.
    
    Args:
        x: Non-negative input value
    
    Returns:
        √x
    """
    if x < 0.0:
        return 0.0  # NaN would be better
    if x == 0.0:
        return 0.0
    
    var guess = x / 2.0
    for _ in range(20):
        guess = (guess + x / guess) / 2.0
    return guess


fn cbrt(x: Float64) -> Float64:
    """Cube root using Newton's method.
    
    Args:
        x: Input value
    
    Returns:
        ∛x
    """
    let is_negative = x < 0.0
    let abs_x = abs(x)
    
    var guess = abs_x / 3.0
    for _ in range(20):
        guess = (2.0 * guess + abs_x / (guess * guess)) / 3.0
    
    return -guess if is_negative else guess


fn pow(base: Float64, exponent: Float64) -> Float64:
    """Power function: base^exponent.
    
    Args:
        base: Base value
        exponent: Exponent
    
    Returns:
        base^exponent
    """
    # Special cases
    if exponent == 0.0:
        return 1.0
    if exponent == 1.0:
        return base
    if base == 0.0:
        return 0.0
    
    # For integer exponents, use fast exponentiation
    let exp_int = Int(exponent)
    if Float64(exp_int) == exponent:
        return pow_int(base, exp_int)
    
    # General case: a^b = e^(b * ln(a))
    return exp(exponent * ln(base))


fn pow_int(base: Float64, exponent: Int) -> Float64:
    """Fast integer exponentiation.
    
    Args:
        base: Base value
        exponent: Integer exponent
    
    Returns:
        base^exponent
    """
    if exponent == 0:
        return 1.0
    
    let is_negative = exponent < 0
    var exp = abs_int(exponent)
    
    var result = 1.0
    var current_base = base
    
    while exp > 0:
        if exp % 2 == 1:
            result *= current_base
        current_base *= current_base
        exp //= 2
    
    return 1.0 / result if is_negative else result


# Exponential and logarithmic functions

fn exp(x: Float64) -> Float64:
    """Exponential function: e^x.
    
    Args:
        x: Input value
    
    Returns:
        e^x
    """
    # Taylor series: e^x = 1 + x + x^2/2! + x^3/3! + ...
    var sum = 1.0
    var term = 1.0
    
    for n in range(1, 30):
        term *= x / Float64(n)
        sum += term
        if abs(term) < 1e-15:
            break
    
    return sum


fn exp2(x: Float64) -> Float64:
    """Base-2 exponential: 2^x.
    
    Args:
        x: Input value
    
    Returns:
        2^x
    """
    return exp(x * LN_2)


fn exp10(x: Float64) -> Float64:
    """Base-10 exponential: 10^x.
    
    Args:
        x: Input value
    
    Returns:
        10^x
    """
    return exp(x * LN_10)


fn ln(x: Float64) -> Float64:
    """Natural logarithm: ln(x).
    
    Args:
        x: Positive input value
    
    Returns:
        ln(x)
    """
    if x <= 0.0:
        return 0.0  # NaN would be better
    
    # Use series expansion around x = 1
    # ln(x) = 2 * ((x-1)/(x+1) + 1/3((x-1)/(x+1))^3 + ...)
    let y = (x - 1.0) / (x + 1.0)
    let y2 = y * y
    
    var sum = y
    var term = y
    
    for n in range(1, 30):
        term *= y2
        sum += term / Float64(2 * n + 1)
        if abs(term) < 1e-15:
            break
    
    return 2.0 * sum


fn log2(x: Float64) -> Float64:
    """Base-2 logarithm.
    
    Args:
        x: Positive input value
    
    Returns:
        log₂(x)
    """
    return ln(x) * LOG2_E


fn log10(x: Float64) -> Float64:
    """Base-10 logarithm.
    
    Args:
        x: Positive input value
    
    Returns:
        log₁₀(x)
    """
    return ln(x) * LOG10_E


fn log(x: Float64, base: Float64) -> Float64:
    """Logarithm with arbitrary base.
    
    Args:
        x: Positive input value
        base: Logarithm base
    
    Returns:
        log_base(x)
    """
    return ln(x) / ln(base)


# Trigonometric functions

fn sin(x: Float64) -> Float64:
    """Sine function.
    
    Args:
        x: Angle in radians
    
    Returns:
        sin(x)
    """
    # Normalize to [-π, π]
    var angle = x
    while angle > PI:
        angle -= TAU
    while angle < -PI:
        angle += TAU
    
    # Taylor series: sin(x) = x - x^3/3! + x^5/5! - ...
    var sum = angle
    var term = angle
    
    for n in range(1, 15):
        term *= -(angle * angle) / Float64((2 * n) * (2 * n + 1))
        sum += term
        if abs(term) < 1e-15:
            break
    
    return sum


fn cos(x: Float64) -> Float64:
    """Cosine function.
    
    Args:
        x: Angle in radians
    
    Returns:
        cos(x)
    """
    # Normalize to [-π, π]
    var angle = x
    while angle > PI:
        angle -= TAU
    while angle < -PI:
        angle += TAU
    
    # Taylor series: cos(x) = 1 - x^2/2! + x^4/4! - ...
    var sum = 1.0
    var term = 1.0
    
    for n in range(1, 15):
        term *= -(angle * angle) / Float64((2 * n - 1) * (2 * n))
        sum += term
        if abs(term) < 1e-15:
            break
    
    return sum


fn tan(x: Float64) -> Float64:
    """Tangent function.
    
    Args:
        x: Angle in radians
    
    Returns:
        tan(x) = sin(x) / cos(x)
    """
    return sin(x) / cos(x)


fn asin(x: Float64) -> Float64:
    """Arcsine function.
    
    Args:
        x: Input value in [-1, 1]
    
    Returns:
        arcsin(x) in [-π/2, π/2]
    """
    if x < -1.0 or x > 1.0:
        return 0.0  # NaN would be better
    
    # Taylor series around 0
    var sum = x
    var term = x
    
    for n in range(1, 20):
        term *= x * x * Float64(2 * n - 1) / Float64(2 * n)
        sum += term / Float64(2 * n + 1)
        if abs(term) < 1e-15:
            break
    
    return sum


fn acos(x: Float64) -> Float64:
    """Arccosine function.
    
    Args:
        x: Input value in [-1, 1]
    
    Returns:
        arccos(x) in [0, π]
    """
    return PI / 2.0 - asin(x)


fn atan(x: Float64) -> Float64:
    """Arctangent function.
    
    Args:
        x: Input value
    
    Returns:
        arctan(x) in [-π/2, π/2]
    """
    # Use series for |x| <= 1, otherwise use atan(x) = π/2 - atan(1/x)
    if abs(x) <= 1.0:
        var sum = x
        var term = x
        
        for n in range(1, 30):
            term *= -x * x
            sum += term / Float64(2 * n + 1)
            if abs(term) < 1e-15:
                break
        
        return sum
    else:
        let result = PI / 2.0 - atan(1.0 / x)
        return result if x > 0.0 else -result


fn atan2(y: Float64, x: Float64) -> Float64:
    """Two-argument arctangent.
    
    Args:
        y: Y-coordinate
        x: X-coordinate
    
    Returns:
        Angle in radians in [-π, π]
    """
    if x > 0.0:
        return atan(y / x)
    elif x < 0.0:
        if y >= 0.0:
            return atan(y / x) + PI
        else:
            return atan(y / x) - PI
    else:  # x == 0
        if y > 0.0:
            return PI / 2.0
        elif y < 0.0:
            return -PI / 2.0
        else:
            return 0.0  # Undefined, but return 0


# Hyperbolic functions

fn sinh(x: Float64) -> Float64:
    """Hyperbolic sine.
    
    Args:
        x: Input value
    
    Returns:
        sinh(x) = (e^x - e^(-x)) / 2
    """
    let exp_x = exp(x)
    return (exp_x - 1.0 / exp_x) / 2.0


fn cosh(x: Float64) -> Float64:
    """Hyperbolic cosine.
    
    Args:
        x: Input value
    
    Returns:
        cosh(x) = (e^x + e^(-x)) / 2
    """
    let exp_x = exp(x)
    return (exp_x + 1.0 / exp_x) / 2.0


fn tanh(x: Float64) -> Float64:
    """Hyperbolic tangent.
    
    Args:
        x: Input value
    
    Returns:
        tanh(x) = sinh(x) / cosh(x)
    """
    let exp_2x = exp(2.0 * x)
    return (exp_2x - 1.0) / (exp_2x + 1.0)


# Angular conversion

fn degrees(radians: Float64) -> Float64:
    """Convert radians to degrees.
    
    Args:
        radians: Angle in radians
    
    Returns:
        Angle in degrees
    """
    return radians * 180.0 / PI


fn radians(degrees: Float64) -> Float64:
    """Convert degrees to radians.
    
    Args:
        degrees: Angle in degrees
    
    Returns:
        Angle in radians
    """
    return degrees * PI / 180.0


# Special functions

fn factorial(n: Int) -> Int:
    """Factorial function.
    
    Args:
        n: Non-negative integer
    
    Returns:
        n! = n × (n-1) × ... × 2 × 1
    """
    if n <= 1:
        return 1
    
    var result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result


fn gcd(a: Int, b: Int) -> Int:
    """Greatest common divisor (Euclidean algorithm).
    
    Args:
        a: First integer
        b: Second integer
    
    Returns:
        GCD(a, b)
    """
    var x = abs_int(a)
    var y = abs_int(b)
    
    while y != 0:
        let temp = y
        y = x % y
        x = temp
    
    return x


fn lcm(a: Int, b: Int) -> Int:
    """Least common multiple.
    
    Args:
        a: First integer
        b: Second integer
    
    Returns:
        LCM(a, b)
    """
    return abs_int(a * b) // gcd(a, b)


fn is_prime(n: Int) -> Bool:
    """Check if number is prime.
    
    Args:
        n: Integer to check
    
    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    var i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    
    return True


# Rounding functions

fn floor(x: Float64) -> Float64:
    """Floor function (round down).
    
    Args:
        x: Input value
    
    Returns:
        Largest integer ≤ x
    """
    let int_part = Int(x)
    if x >= 0.0 or Float64(int_part) == x:
        return Float64(int_part)
    return Float64(int_part - 1)


fn ceil(x: Float64) -> Float64:
    """Ceiling function (round up).
    
    Args:
        x: Input value
    
    Returns:
        Smallest integer ≥ x
    """
    let int_part = Int(x)
    if x <= 0.0 or Float64(int_part) == x:
        return Float64(int_part)
    return Float64(int_part + 1)


fn round(x: Float64) -> Float64:
    """Round to nearest integer.
    
    Args:
        x: Input value
    
    Returns:
        Nearest integer (ties round to even)
    """
    let f = floor(x)
    let c = ceil(x)
    
    if x - f < c - x:
        return f
    elif x - f > c - x:
        return c
    else:  # Tie: round to even
        if Int(f) % 2 == 0:
            return f
        return c


fn trunc(x: Float64) -> Float64:
    """Truncate to integer (round toward zero).
    
    Args:
        x: Input value
    
    Returns:
        Integer part of x
    """
    return Float64(Int(x))


fn fmod(x: Float64, y: Float64) -> Float64:
    """Floating-point modulo.
    
    Args:
        x: Dividend
        y: Divisor
    
    Returns:
        x mod y
    """
    return x - trunc(x / y) * y


# Miscellaneous

fn lerp(a: Float64, b: Float64, t: Float64) -> Float64:
    """Linear interpolation.
    
    Args:
        a: Start value
        b: End value
        t: Interpolation parameter [0, 1]
    
    Returns:
        a + t * (b - a)
    """
    return a + t * (b - a)


fn smoothstep(x: Float64) -> Float64:
    """Smooth interpolation (cubic Hermite).
    
    Args:
        x: Input in [0, 1]
    
    Returns:
        Smoothed value
    """
    let t = clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


fn is_nan(x: Float64) -> Bool:
    """Check if value is NaN.
    
    Args:
        x: Input value
    
    Returns:
        True if x is NaN
    """
    return x != x


fn is_inf(x: Float64) -> Bool:
    """Check if value is infinite.
    
    Args:
        x: Input value
    
    Returns:
        True if x is infinite
    """
    return abs(x) > 1e308
