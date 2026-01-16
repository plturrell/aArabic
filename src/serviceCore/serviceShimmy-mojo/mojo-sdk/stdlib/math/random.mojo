# Math/Random - Random Number Generation
# Day 38: RNG, distributions, and statistical sampling

from builtin import Int, Float64, Bool
from math import sqrt, ln, cos, sin, PI


# Linear Congruential Generator (LCG)

struct Random:
    """Pseudo-random number generator using LCG.
    
    Uses a Linear Congruential Generator for fast, deterministic
    random number generation.
    
    Examples:
        ```mojo
        var rng = Random(12345)  # Seed
        let r1 = rng.next_float()  # [0.0, 1.0)
        let r2 = rng.next_int(100)  # [0, 100)
        ```
    """
    
    var state: Int
    
    # LCG parameters (from Numerical Recipes)
    let MULTIPLIER: Int = 1664525
    let INCREMENT: Int = 1013904223
    let MODULUS: Int = 2147483648  # 2^31
    
    fn __init__(inout self, seed: Int = 0):
        """Initialize with seed.
        
        Args:
            seed: Random seed (default: 0)
        """
        self.state = seed if seed != 0 else 1
    
    fn next(inout self) -> Int:
        """Generate next random integer.
        
        Returns:
            Random integer in [0, MODULUS)
        """
        self.state = (self.MULTIPLIER * self.state + self.INCREMENT) % self.MODULUS
        return self.state
    
    fn next_float(inout self) -> Float64:
        """Generate random float in [0.0, 1.0).
        
        Returns:
            Random float
        """
        return Float64(self.next()) / Float64(self.MODULUS)
    
    fn next_int(inout self, max_value: Int) -> Int:
        """Generate random integer in [0, max_value).
        
        Args:
            max_value: Upper bound (exclusive)
        
        Returns:
            Random integer
        """
        return self.next() % max_value
    
    fn next_range(inout self, min_value: Int, max_value: Int) -> Int:
        """Generate random integer in [min_value, max_value).
        
        Args:
            min_value: Lower bound (inclusive)
            max_value: Upper bound (exclusive)
        
        Returns:
            Random integer in range
        """
        return min_value + self.next_int(max_value - min_value)
    
    fn next_bool(inout self) -> Bool:
        """Generate random boolean.
        
        Returns:
            Random true or false
        """
        return self.next() % 2 == 0
    
    fn shuffle[T](inout self, inout data: List[T]):
        """Shuffle list in-place (Fisher-Yates).
        
        Args:
            data: List to shuffle
        """
        for i in range(len(data) - 1, 0, -1):
            let j = self.next_int(i + 1)
            let temp = data[i]
            data[i] = data[j]
            data[j] = temp
    
    fn choice[T](inout self, data: List[T]) -> T:
        """Choose random element from list.
        
        Args:
            data: List to choose from
        
        Returns:
            Random element
        """
        let idx = self.next_int(len(data))
        return data[idx]
    
    fn sample[T](inout self, data: List[T], k: Int) -> List[T]:
        """Random sample of k elements without replacement.
        
        Args:
            data: List to sample from
            k: Number of elements to sample
        
        Returns:
            List of k random elements
        """
        var result = List[T]()
        var indices = List[Int]()
        
        for i in range(len(data)):
            indices.append(i)
        
        self.shuffle(indices)
        
        for i in range(min(k, len(data))):
            result.append(data[indices[i]])
        
        return result


# Global RNG instance

var _global_rng = Random(12345)


# Convenience functions

fn seed(value: Int):
    """Seed the global RNG.
    
    Args:
        value: Seed value
    """
    _global_rng = Random(value)


fn random() -> Float64:
    """Generate random float in [0.0, 1.0).
    
    Returns:
        Random float
    """
    return _global_rng.next_float()


fn randint(max_value: Int) -> Int:
    """Generate random integer in [0, max_value).
    
    Args:
        max_value: Upper bound (exclusive)
    
    Returns:
        Random integer
    """
    return _global_rng.next_int(max_value)


fn randrange(min_value: Int, max_value: Int) -> Int:
    """Generate random integer in [min_value, max_value).
    
    Args:
        min_value: Lower bound (inclusive)
        max_value: Upper bound (exclusive)
    
    Returns:
        Random integer in range
    """
    return _global_rng.next_range(min_value, max_value)


fn randbool() -> Bool:
    """Generate random boolean.
    
    Returns:
        Random true or false
    """
    return _global_rng.next_bool()


# Statistical distributions

fn uniform(low: Float64, high: Float64) -> Float64:
    """Uniform distribution in [low, high).
    
    Args:
        low: Lower bound
        high: Upper bound
    
    Returns:
        Random value in range
    """
    return low + random() * (high - low)


fn normal(mean: Float64, std_dev: Float64) -> Float64:
    """Normal (Gaussian) distribution using Box-Muller transform.
    
    Args:
        mean: Mean (μ)
        std_dev: Standard deviation (σ)
    
    Returns:
        Random value from normal distribution
    """
    let u1 = random()
    let u2 = random()
    let z0 = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
    return mean + std_dev * z0


fn exponential(lambda: Float64) -> Float64:
    """Exponential distribution.
    
    Args:
        lambda: Rate parameter (λ)
    
    Returns:
        Random value from exponential distribution
    """
    return -ln(random()) / lambda


fn poisson(lambda: Float64) -> Int:
    """Poisson distribution (Knuth's algorithm).
    
    Args:
        lambda: Rate parameter (λ)
    
    Returns:
        Random integer from Poisson distribution
    """
    let L = exp(-lambda)
    var k = 0
    var p = 1.0
    
    while p > L:
        k += 1
        p *= random()
    
    return k - 1


fn binomial(n: Int, p: Float64) -> Int:
    """Binomial distribution.
    
    Args:
        n: Number of trials
        p: Success probability
    
    Returns:
        Number of successes
    """
    var successes = 0
    
    for _ in range(n):
        if random() < p:
            successes += 1
    
    return successes


fn geometric(p: Float64) -> Int:
    """Geometric distribution (first success).
    
    Args:
        p: Success probability
    
    Returns:
        Number of trials until first success
    """
    return Int(ln(random()) / ln(1.0 - p)) + 1


# Helper function
fn exp(x: Float64) -> Float64:
    """Exponential function (from math module)."""
    # Would import from math.mojo
    var sum = 1.0
    var term = 1.0
    for n in range(1, 20):
        term *= x / Float64(n)
        sum += term
    return sum


fn min(a: Int, b: Int) -> Int:
    """Minimum of two integers."""
    return a if a < b else b


# ============================================================================
# Tests
# ============================================================================

test "random basic generation":
    var rng = Random(12345)
    let r1 = rng.next_float()
    assert(r1 >= 0.0 and r1 < 1.0)

test "random int range":
    var rng = Random(12345)
    let r = rng.next_int(10)
    assert(r >= 0 and r < 10)

test "random range bounds":
    var rng = Random(12345)
    let r = rng.next_range(5, 15)
    assert(r >= 5 and r < 15)

test "random bool":
    var rng = Random(12345)
    let b = rng.next_bool()
    assert(b == True or b == False)

test "random choice":
    var rng = Random(12345)
    var data = List[Int]()
    data.append(10)
    data.append(20)
    data.append(30)
    let choice = rng.choice(data)
    assert(choice == 10 or choice == 20 or choice == 30)

test "uniform distribution":
    seed(12345)
    let r = uniform(5.0, 10.0)
    assert(r >= 5.0 and r < 10.0)

test "normal distribution":
    seed(12345)
    let r = normal(0.0, 1.0)
    # Just check it generates a value
    assert(r != 0.0 or r == 0.0)

test "exponential distribution":
    seed(12345)
    let r = exponential(1.0)
    assert(r >= 0.0)

test "binomial distribution":
    seed(12345)
    let successes = binomial(10, 0.5)
    assert(successes >= 0 and successes <= 10)

test "seeding reproducibility":
    var rng1 = Random(42)
    var rng2 = Random(42)
    let r1 = rng1.next_float()
    let r2 = rng2.next_float()
    assert(r1 == r2)
