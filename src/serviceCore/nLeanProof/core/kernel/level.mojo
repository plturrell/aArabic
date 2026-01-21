"""
Universe levels for the Lean4 kernel.
"""

from collections import List


@fieldwise_init
struct LevelKind(ImplicitlyCopyable, Copyable, Movable):
    var value: Int

    fn __eq__(self, other: LevelKind) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: LevelKind) -> Bool:
        return self.value != other.value

    comptime ZERO = LevelKind(0)       # Base level 0
    comptime SUCC = LevelKind(1)       # Successor level
    comptime MAX = LevelKind(2)        # Maximum of two levels
    comptime IMAX = LevelKind(3)       # Impredicative max
    comptime PARAM = LevelKind(4)      # Universe parameter
    comptime MVAR = LevelKind(5)       # Metavariable level


@fieldwise_init
struct Level(Copyable, Movable):
    """Universe level representation."""
    var kind: LevelKind
    var depth: Int  # For SUCC, the successor depth
    var name: String  # For PARAM/MVAR
    var left: Optional[Self]  # For MAX/IMAX
    var right: Optional[Self]  # For MAX/IMAX

    fn __init__(out self, kind: LevelKind):
        self.kind = kind
        self.depth = 0
        self.name = ""
        self.left = None
        self.right = None

    @staticmethod
    fn zero() -> Level:
        return Level(LevelKind.ZERO)

    @staticmethod
    fn succ(l: Level) -> Level:
        var result = Level(LevelKind.SUCC)
        result.depth = l.depth + 1
        result.left = l.copy()
        return result^

    @staticmethod
    fn max(l1: Level, l2: Level) -> Level:
        var result = Level(LevelKind.MAX)
        result.left = l1.copy()
        result.right = l2.copy()
        return result^

    @staticmethod
    fn imax(l1: Level, l2: Level) -> Level:
        var result = Level(LevelKind.IMAX)
        result.left = l1.copy()
        result.right = l2.copy()
        return result^

    @staticmethod
    fn param(name: String) -> Level:
        var result = Level(LevelKind.PARAM)
        result.name = name
        return result^

    @staticmethod
    fn mvar(name: String) -> Level:
        var result = Level(LevelKind.MVAR)
        result.name = name
        return result^

    fn is_zero(self) -> Bool:
        return self.kind == LevelKind.ZERO

    fn is_succ(self) -> Bool:
        return self.kind == LevelKind.SUCC

    fn is_max(self) -> Bool:
        return self.kind == LevelKind.MAX

    fn is_imax(self) -> Bool:
        return self.kind == LevelKind.IMAX

    fn is_param(self) -> Bool:
        return self.kind == LevelKind.PARAM

    fn is_mvar(self) -> Bool:
        return self.kind == LevelKind.MVAR

    fn to_nat(self) -> Optional[Int]:
        """Convert to a natural number if possible."""
        if self.is_zero():
            return 0
        elif self.is_succ() and self.left:
            var inner = self.left.value().to_nat()
            if inner:
                return inner.value() + 1
        return None


fn level_eq(l1: Level, l2: Level) -> Bool:
    """Check if two levels are definitionally equal."""
    if l1.kind != l2.kind:
        return False
    if l1.kind == LevelKind.ZERO:
        return True
    elif l1.kind == LevelKind.SUCC:
        if l1.left and l2.left:
            return level_eq(l1.left.value(), l2.left.value())
        return False
    elif l1.kind == LevelKind.PARAM or l1.kind == LevelKind.MVAR:
        return l1.name == l2.name
    elif l1.kind == LevelKind.MAX or l1.kind == LevelKind.IMAX:
        if l1.left and l2.left and l1.right and l2.right:
            return level_eq(l1.left.value(), l2.left.value()) and \
                   level_eq(l1.right.value(), l2.right.value())
        return False
    return False


fn level_to_string(level: Level) -> String:
    """Convert a level to string representation."""
    if level.is_zero():
        return "0"
    elif level.is_succ():
        if level.left:
            var nat = level.to_nat()
            if nat:
                return str(nat.value())
            return "(succ " + level_to_string(level.left.value()) + ")"
        return "(succ ?)"
    elif level.is_max():
        var l = ""
        var r = ""
        if level.left:
            l = level_to_string(level.left.value())
        if level.right:
            r = level_to_string(level.right.value())
        return "(max " + l + " " + r + ")"
    elif level.is_imax():
        var l = ""
        var r = ""
        if level.left:
            l = level_to_string(level.left.value())
        if level.right:
            r = level_to_string(level.right.value())
        return "(imax " + l + " " + r + ")"
    elif level.is_param():
        return level.name
    elif level.is_mvar():
        return "?" + level.name
    return "?"
