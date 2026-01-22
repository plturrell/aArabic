"""
Hierarchical names for the Lean4 kernel.
"""

from collections import List


@fieldwise_init
struct NameKind(ImplicitlyCopyable, Copyable, Movable):
    var value: Int

    fn __eq__(self, other: NameKind) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: NameKind) -> Bool:
        return self.value != other.value

    comptime ANONYMOUS = NameKind(0)  # Empty/anonymous name
    comptime STR = NameKind(1)        # String component
    comptime NUM = NameKind(2)        # Numeric component


@fieldwise_init
struct Name(Copyable, Movable):
    """Hierarchical name representation."""
    var kind: NameKind
    var str_value: String
    var num_value: Int
    var prefix: Optional[Self]

    fn __init__(out self):
        self.kind = NameKind.ANONYMOUS
        self.str_value = ""
        self.num_value = 0
        self.prefix = None

    fn anonymous() -> Name:
        return Name()

    fn mk_string(s: String, prefix: Name = Name.anonymous()) -> Name:
        var result = Name()
        result.kind = NameKind.STR
        result.str_value = s
        if prefix.kind != NameKind.ANONYMOUS:
            result.prefix = prefix
        return result

    fn mk_num(n: Int, prefix: Name = Name.anonymous()) -> Name:
        var result = Name()
        result.kind = NameKind.NUM
        result.num_value = n
        if prefix.kind != NameKind.ANONYMOUS:
            result.prefix = prefix
        return result

    fn is_anonymous(self) -> Bool:
        return self.kind == NameKind.ANONYMOUS

    fn is_string(self) -> Bool:
        return self.kind == NameKind.STR

    fn is_num(self) -> Bool:
        return self.kind == NameKind.NUM

    fn get_string(self) -> String:
        if self.is_string():
            return self.str_value
        return ""

    fn get_num(self) -> Int:
        if self.is_num():
            return self.num_value
        return 0

    fn get_prefix(self) -> Name:
        if self.prefix:
            return self.prefix.value()
        return Name.anonymous()

    fn append_string(self, s: String) -> Name:
        return Name.mk_string(s, self)

    fn append_num(self, n: Int) -> Name:
        return Name.mk_num(n, self)


fn name_eq(n1: Name, n2: Name) -> Bool:
    """Check if two names are equal."""
    if n1.kind != n2.kind:
        return False
    if n1.kind == NameKind.ANONYMOUS:
        return True
    elif n1.kind == NameKind.STR:
        if n1.str_value != n2.str_value:
            return False
    elif n1.kind == NameKind.NUM:
        if n1.num_value != n2.num_value:
            return False
    # Check prefixes
    if n1.prefix and n2.prefix:
        return name_eq(n1.prefix.value(), n2.prefix.value())
    elif not n1.prefix and not n2.prefix:
        return True
    return False


fn name_to_string(name: Name) -> String:
    """Convert a name to string representation."""
    if name.is_anonymous():
        return "[anonymous]"
    var parts = List[String]()
    var current = name
    while True:
        if current.is_string():
            parts.append(current.str_value)
        elif current.is_num():
            parts.append(str(current.num_value))
        elif current.is_anonymous():
            break
        if current.prefix:
            current = current.prefix.value()
        else:
            break
    # Reverse and join
    var result = String("")
    var i = len(parts) - 1
    while i >= 0:
        if len(result) > 0:
            result += "."
        result += parts[i]
        i -= 1
    return result


fn parse_name(s: String) -> Name:
    """Parse a string into a Name."""
    if len(s) == 0:
        return Name.anonymous()
    var result = Name.anonymous()
    var current = String("")
    for i in range(len(s)):
        var c = s[i]
        if c == ".":
            if len(current) > 0:
                result = result.append_string(current)
                current = ""
        else:
            current += c
    if len(current) > 0:
        result = result.append_string(current)
    return result
