"""
Runtime value representation for Lean4.
"""

from collections import List


@fieldwise_init
struct ValueKind(ImplicitlyCopyable, Copyable, Movable):
    """Kind of runtime value."""
    var value: Int

    fn __eq__(self, other: ValueKind) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: ValueKind) -> Bool:
        return self.value != other.value

    comptime SCALAR = ValueKind(0)      # Unboxed scalar (integers, etc.)
    comptime OBJECT = ValueKind(1)      # Reference-counted object
    comptime CLOSURE = ValueKind(2)     # Function closure
    comptime CTOR = ValueKind(3)        # Constructor application
    comptime THUNK = ValueKind(4)       # Lazy evaluation thunk
    comptime EXTERN = ValueKind(5)      # External/native value


@fieldwise_init
struct Object(Copyable, Movable):
    """Heap-allocated reference-counted object."""
    var ref_count: Int
    var tag: Int  # Constructor tag for inductives
    var fields: List[Value]

    fn __init__(out self, tag: Int):
        self.ref_count = 1
        self.tag = tag
        self.fields = List[Value]()

    fn add_field(mut self, value: Value):
        self.fields.append(value)

    fn inc_ref(mut self):
        self.ref_count += 1

    fn dec_ref(mut self) -> Bool:
        self.ref_count -= 1
        return self.ref_count == 0


@fieldwise_init
struct Closure(Copyable, Movable):
    """Function closure with captured environment."""
    var func_name: String
    var arity: Int
    var args_collected: Int
    var partial_args: List[Value]

    fn __init__(out self, func_name: String, arity: Int):
        self.func_name = func_name
        self.arity = arity
        self.args_collected = 0
        self.partial_args = List[Value]()

    fn apply(mut self, arg: Value) -> Optional[Closure]:
        """Apply an argument to the closure."""
        self.partial_args.append(arg)
        self.args_collected += 1
        if self.args_collected < self.arity:
            return self
        return None  # Fully applied


@fieldwise_init
struct Value(Copyable, Movable):
    """Runtime value."""
    var kind: ValueKind
    var scalar: Int
    var object: Optional[Object]
    var closure: Optional[Closure]
    var name: String  # For constructors/externs

    fn __init__(out self, kind: ValueKind):
        self.kind = kind
        self.scalar = 0
        self.object = None
        self.closure = None
        self.name = ""

    @staticmethod
    fn mk_scalar(n: Int) -> Value:
        var v = Value(ValueKind.SCALAR)
        v.scalar = n
        return v^

    @staticmethod
    fn mk_nat(n: Int) -> Value:
        return Value.mk_scalar(n)

    @staticmethod
    fn mk_bool(b: Bool) -> Value:
        return Value.mk_scalar(1 if b else 0)

    @staticmethod
    fn mk_unit() -> Value:
        return Value.mk_scalar(0)

    @staticmethod
    fn mk_object(tag: Int) -> Value:
        var v = Value(ValueKind.OBJECT)
        v.object = Object(tag)
        return v^

    @staticmethod
    fn mk_ctor(name: String, tag: Int) -> Value:
        var v = Value(ValueKind.CTOR)
        v.name = name
        v.object = Object(tag)
        return v^

    @staticmethod
    fn mk_closure(func_name: String, arity: Int) -> Value:
        var v = Value(ValueKind.CLOSURE)
        v.closure = Closure(func_name, arity)
        return v^

    fn is_scalar(self) -> Bool:
        return self.kind == ValueKind.SCALAR

    fn is_object(self) -> Bool:
        return self.kind == ValueKind.OBJECT

    fn is_closure(self) -> Bool:
        return self.kind == ValueKind.CLOSURE

    fn is_ctor(self) -> Bool:
        return self.kind == ValueKind.CTOR

    fn to_nat(self) -> Int:
        if self.is_scalar():
            return self.scalar
        return 0

    fn to_bool(self) -> Bool:
        if self.is_scalar():
            return self.scalar != 0
        return False

    fn get_tag(self) -> Int:
        if self.object:
            return self.object.value().tag
        return 0

    fn get_field(self, index: Int) -> Optional[Value]:
        if self.object:
            var obj = self.object.value()
            if index >= 0 and index < len(obj.fields):
                return obj.fields[index]
        return None
