-- Basic Lean4 test file for conformance testing

-- Simple definition
def hello := "Hello, World!"

-- Natural number operations
def add_one (n : Nat) : Nat := n + 1

-- Boolean operations
def is_even (n : Nat) : Bool := n % 2 == 0

-- Simple theorem
theorem zero_add (n : Nat) : 0 + n = n := by
  simp

-- Type checking
#check Nat
#check Bool
#check String

-- Evaluation
#eval 1 + 1
#eval "Hello" ++ " World"

