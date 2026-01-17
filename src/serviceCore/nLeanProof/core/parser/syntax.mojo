"""
Syntax tree primitives for leanShimmy.
"""

from collections import List


@fieldwise_init
struct NodeKind(ImplicitlyCopyable, Copyable, Movable):
    var value: Int

    fn __eq__(self, other: NodeKind) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: NodeKind) -> Bool:
        return self.value != other.value

    comptime PROGRAM = NodeKind(0)
    comptime IMPORT = NodeKind(1)
    comptime NAMESPACE = NodeKind(2)
    comptime SECTION = NodeKind(3)
    comptime DEF = NodeKind(4)
    comptime THEOREM = NodeKind(5)
    comptime BY = NodeKind(6)
    comptime IDENT = NodeKind(7)
    comptime NUMBER = NodeKind(8)
    comptime STRING = NodeKind(9)
    comptime CHAR = NodeKind(10)
    comptime APP = NodeKind(11)
    comptime INFIX = NodeKind(12)
    comptime ERROR = NodeKind(13)


@fieldwise_init
struct SyntaxNode(Copyable, Movable):
    var kind: NodeKind
    var value: String
    var children: List[SyntaxNode]


fn node_kind_name(kind: NodeKind) -> String:
    if kind == NodeKind.PROGRAM:
        return "program"
    elif kind == NodeKind.IMPORT:
        return "import"
    elif kind == NodeKind.NAMESPACE:
        return "namespace"
    elif kind == NodeKind.SECTION:
        return "section"
    elif kind == NodeKind.DEF:
        return "def"
    elif kind == NodeKind.THEOREM:
        return "theorem"
    elif kind == NodeKind.BY:
        return "by"
    elif kind == NodeKind.IDENT:
        return "ident"
    elif kind == NodeKind.NUMBER:
        return "number"
    elif kind == NodeKind.STRING:
        return "string"
    elif kind == NodeKind.CHAR:
        return "char"
    elif kind == NodeKind.APP:
        return "app"
    elif kind == NodeKind.INFIX:
        return "infix"
    else:
        return "error"


@fieldwise_init
struct Task(Copyable, Movable):
    var kind: Int  # 0=open, 1=close, 2=space
    var node: SyntaxNode


fn node_to_string(root: SyntaxNode) -> String:
    var result = String("")
    var stack = List[Task]()
    stack.append(Task(0, root.copy()))

    while len(stack) > 0:
        var task = stack.pop()
        if task.kind == 2:
            result += " "
            continue

        if task.kind == 1:
            result += ")"
            continue

        var node = task.node.copy()
        result += "(" + node_kind_name(node.kind)
        if len(node.value) > 0:
            result += " " + node.value

        if len(node.children) == 0:
            result += ")"
            continue

        stack.append(Task(1, node.copy()))
        var i = len(node.children) - 1
        while i >= 0:
            stack.append(Task(0, node.children[i].copy()))
            stack.append(Task(2, node.copy()))
            i -= 1

    return result
