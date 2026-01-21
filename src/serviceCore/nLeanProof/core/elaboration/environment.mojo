"""
Environment for storing declarations and constants.
"""

from collections import Dict, List
from .expr import Expr, Level


@fieldwise_init
struct Declaration(Copyable, Movable):
    """A declaration in the environment."""
    var name: String
    var type: Expr
    var value: Optional[Expr]  # None for axioms/constants, Some for definitions
    var level_params: List[String]
    var is_theorem: Bool

    fn __init__(out self, name: String, type: Expr):
        self.name = name
        self.type = type.copy()
        self.value = None
        self.level_params = List[String]()
        self.is_theorem = False

    fn __copyinit__(out self, other: Declaration):
        self.name = other.name
        self.type = other.type.copy()
        self.value = other.value.copy() if other.value else None
        self.level_params = List[String]()
        for i in range(len(other.level_params)):
            self.level_params.append(other.level_params[i])
        self.is_theorem = other.is_theorem

    @staticmethod
    fn definition(name: String, type: Expr, value: Expr) -> Declaration:
        var decl = Declaration(name, type)
        decl.value = value.copy()
        return decl

    @staticmethod
    fn theorem(name: String, type: Expr, proof: Expr) -> Declaration:
        var decl = Declaration(name, type)
        decl.value = proof.copy()
        decl.is_theorem = True
        return decl

    @staticmethod
    fn axiom(name: String, type: Expr) -> Declaration:
        return Declaration(name, type)


@fieldwise_init
struct Environment(Copyable, Movable):
    """Global environment containing all declarations."""
    var declarations: Dict[String, Declaration]
    var imports: List[String]

    fn __init__(out self):
        self.declarations = Dict[String, Declaration]()
        self.imports = List[String]()
        self._init_builtins()

    fn _init_builtins(mut self):
        """Initialize built-in types and constants."""
        # Add basic types
        var prop_type = Expr.sort(Level.zero())
        var type_type = Expr.sort(Level.succ(Level.zero()))
        
        self.add_axiom("Prop", prop_type)
        self.add_axiom("Type", type_type)
        
        # Add basic logical constants
        var prop = Expr.const("Prop")
        self.add_axiom("True", prop)
        self.add_axiom("False", prop)
        
        # Add function type constructor
        var pi_type = Expr.pi("A", type_type, 
                             Expr.pi("B", type_type, type_type))
        self.add_axiom("Function", pi_type)

    fn add_declaration(mut self, decl: Declaration):
        """Add a declaration to the environment."""
        self.declarations[decl.name] = decl.copy()

    fn add_axiom(mut self, name: String, type: Expr):
        """Add an axiom to the environment."""
        self.add_declaration(Declaration.axiom(name, type))

    fn add_definition(mut self, name: String, type: Expr, value: Expr):
        """Add a definition to the environment."""
        self.add_declaration(Declaration.definition(name, type, value))

    fn add_theorem(mut self, name: String, type: Expr, proof: Expr):
        """Add a theorem to the environment."""
        self.add_declaration(Declaration.theorem(name, type, proof))

    fn get_declaration(self, name: String) -> Optional[Declaration]:
        """Get a declaration by name."""
        if name in self.declarations:
            return self.declarations[name]
        return None

    fn has_declaration(self, name: String) -> Bool:
        """Check if a declaration exists."""
        return name in self.declarations

    fn get_type(self, name: String) -> Optional[Expr]:
        """Get the type of a constant."""
        var decl = self.get_declaration(name)
        if decl:
            return decl.value().type
        return None

    fn get_value(self, name: String) -> Optional[Expr]:
        """Get the value of a definition."""
        var decl = self.get_declaration(name)
        if decl and decl.value().value:
            return decl.value().value.value()
        return None

    fn is_theorem(self, name: String) -> Bool:
        """Check if a declaration is a theorem."""
        var decl = self.get_declaration(name)
        if decl:
            return decl.value().is_theorem
        return False

    fn add_import(mut self, module_name: String):
        """Add an import to the environment."""
        self.imports.append(module_name)

    fn list_declarations(self) -> List[String]:
        """List all declaration names."""
        var names = List[String]()
        for item in self.declarations.items():
            names.append(item[].key)
        return names
