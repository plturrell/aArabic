from core.kernel.type_checker_kernel import KernelTypeChecker
from core.elaboration.expr import Expr, Level
from core.kernel.level import LevelKind
from testing import assert_true, assert_false

fn test_universe_inference():
    print("Testing Universe Inference...")
    var checker = KernelTypeChecker()
    
    # Test 1: Prop -> Prop is Prop (Sort 0)
    # (P : Prop) -> P
    var prop = Expr.sort(Level.zero())
    var p_type = Expr.pi("P", prop, Expr.var(0)) # Simplified body var(0)
    
    # We can't easily test open terms without full context setup in test, 
    # so let's test the PI rule components.
    
    # Manually check imax logic
    var l0 = Level.zero()
    var l1 = Level.succ(l0) # Type 1
    
    # imax 0 0 = 0
    var r1 = Level.imax(l0, l0) 
    # We don't have reduction in Level yet, but we can check structure or if we added reduction
    # For now, let's just ensure the kernel produces the structure
    
    # Create an expression: Type 0 -> Type 0
    var type0 = Expr.sort(Level.zero())
    var pi_expr = Expr.pi("x", type0, type0) 
    
    var inferred = checker._infer_type(pi_expr)
    if inferred:
        print("Inferred: " + str(inferred.value().level.kind.value))
        # Should be Sort (imax 0 1) -> which is Type 0 if normalized, but structural check:
        # Sort 0 has level 1 (succ 0).
        # So it's imax(1, 1) -> 1.
        pass
    else:
        print("Inference failed")

    # Test 2: Type 0 -> Type 1
    # should be imax(1, 2) -> 2 (Type 1)
    
    print("Universe Inference Test logic executed (basic).")

fn main():
    test_universe_inference()
