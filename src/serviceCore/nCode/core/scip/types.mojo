"""
SCIP (Source Code Intelligence Protocol) type definitions for Mojo.

These types mirror the SCIP protobuf definitions and provide type-safe
representations for code intelligence operations.
"""

from collections import List, Optional


# ========== Enums ==========

@value
struct ProtocolVersion:
    """SCIP protocol version."""
    var value: Int

    alias Unspecified = ProtocolVersion(0)

    fn __init__(inout self, value: Int = 0):
        self.value = value


@value
struct TextEncoding:
    """Text encoding for source files."""
    var value: Int

    alias Unspecified = TextEncoding(0)
    alias UTF8 = TextEncoding(1)
    alias UTF16 = TextEncoding(2)

    fn __init__(inout self, value: Int = 0):
        self.value = value


@value
struct PositionEncoding:
    """Encoding for character offsets in source ranges."""
    var value: Int

    alias Unspecified = PositionEncoding(0)
    alias UTF8CodeUnitOffsetFromLineStart = PositionEncoding(1)
    alias UTF16CodeUnitOffsetFromLineStart = PositionEncoding(2)
    alias UTF32CodeUnitOffsetFromLineStart = PositionEncoding(3)

    fn __init__(inout self, value: Int = 0):
        self.value = value


@value
struct SymbolRole:
    """Role of a symbol in an occurrence (bitset)."""
    var value: Int32

    alias Unspecified = SymbolRole(0)
    alias Definition = SymbolRole(0x1)
    alias Import = SymbolRole(0x2)
    alias WriteAccess = SymbolRole(0x4)
    alias ReadAccess = SymbolRole(0x8)
    alias Generated = SymbolRole(0x10)
    alias Test = SymbolRole(0x20)
    alias ForwardDefinition = SymbolRole(0x40)

    fn __init__(inout self, value: Int32 = 0):
        self.value = value

    fn has_role(self, role: Self) -> Bool:
        """Check if this role includes a specific role."""
        return (self.value & role.value) != 0


@value
struct Severity:
    """Diagnostic severity level."""
    var value: Int

    alias Unspecified = Severity(0)
    alias Error = Severity(1)
    alias Warning = Severity(2)
    alias Information = Severity(3)
    alias Hint = Severity(4)

    fn __init__(inout self, value: Int = 0):
        self.value = value


@value
struct DiagnosticTag:
    """Diagnostic tag for additional categorization."""
    var value: Int

    alias Unspecified = DiagnosticTag(0)
    alias Unnecessary = DiagnosticTag(1)
    alias Deprecated = DiagnosticTag(2)

    fn __init__(inout self, value: Int = 0):
        self.value = value


@value
struct Suffix:
    """Descriptor suffix indicating the kind of symbol."""
    var value: Int

    alias Unspecified = Suffix(0)
    alias Namespace = Suffix(1)
    alias Type = Suffix(2)
    alias Term = Suffix(3)
    alias Method = Suffix(4)
    alias TypeParameter = Suffix(5)
    alias Parameter = Suffix(6)
    alias Meta = Suffix(7)
    alias Local = Suffix(8)
    alias Macro = Suffix(9)

    fn __init__(inout self, value: Int = 0):
        self.value = value


@value
struct SyntaxKind:
    """Syntax highlighting category."""
    var value: Int

    alias Unspecified = SyntaxKind(0)
    alias Comment = SyntaxKind(1)
    alias PunctuationDelimiter = SyntaxKind(2)
    alias PunctuationBracket = SyntaxKind(3)
    alias Keyword = SyntaxKind(4)
    alias IdentifierOperator = SyntaxKind(5)
    alias Identifier = SyntaxKind(6)
    alias IdentifierBuiltin = SyntaxKind(7)
    alias IdentifierNull = SyntaxKind(8)
    alias IdentifierConstant = SyntaxKind(9)
    alias IdentifierMutableGlobal = SyntaxKind(10)
    alias IdentifierParameter = SyntaxKind(11)
    alias IdentifierLocal = SyntaxKind(12)
    alias IdentifierShadowed = SyntaxKind(13)
    alias IdentifierNamespace = SyntaxKind(14)
    alias IdentifierFunction = SyntaxKind(15)
    alias IdentifierFunctionDefinition = SyntaxKind(16)
    alias IdentifierMacro = SyntaxKind(17)
    alias IdentifierMacroDefinition = SyntaxKind(18)
    alias IdentifierType = SyntaxKind(19)
    alias IdentifierBuiltinType = SyntaxKind(20)
    alias IdentifierAttribute = SyntaxKind(21)
    alias StringLiteral = SyntaxKind(27)
    alias StringLiteralEscape = SyntaxKind(28)
    alias StringLiteralSpecial = SyntaxKind(29)
    alias StringLiteralKey = SyntaxKind(30)
    alias CharacterLiteral = SyntaxKind(31)
    alias NumericLiteral = SyntaxKind(32)
    alias BooleanLiteral = SyntaxKind(33)
    alias Tag = SyntaxKind(34)
    alias TagAttribute = SyntaxKind(35)
    alias TagDelimiter = SyntaxKind(36)

    fn __init__(inout self, value: Int = 0):
        self.value = value


@value
struct Kind:
    """Fine-grained symbol kind."""
    var value: Int

    alias Unspecified = Kind(0)
    alias Array = Kind(1)
    alias Assertion = Kind(2)
    alias AssociatedType = Kind(3)
    alias Attribute = Kind(4)
    alias Axiom = Kind(5)
    alias Boolean = Kind(6)
    alias Class = Kind(7)
    alias Constant = Kind(8)
    alias Constructor = Kind(9)
    alias DataFamily = Kind(10)
    alias Enum = Kind(11)
    alias EnumMember = Kind(12)
    alias Event = Kind(13)
    alias Fact = Kind(14)
    alias Field = Kind(15)
    alias File = Kind(16)
    alias Function = Kind(17)
    alias Getter = Kind(18)
    alias Grammar = Kind(19)
    alias Instance = Kind(20)
    alias Interface = Kind(21)
    alias Key = Kind(22)
    alias Lang = Kind(23)
    alias Lemma = Kind(24)
    alias Macro = Kind(25)
    alias Method = Kind(26)
    alias MethodReceiver = Kind(27)
    alias Message = Kind(28)
    alias Module = Kind(29)
    alias Namespace = Kind(30)
    alias Null = Kind(31)
    alias Number = Kind(32)
    alias Object = Kind(33)
    alias Operator = Kind(34)
    alias Package = Kind(35)
    alias PackageObject = Kind(36)
    alias Parameter = Kind(37)
    alias ParameterLabel = Kind(38)
    alias Pattern = Kind(39)
    alias Predicate = Kind(40)
    alias Property = Kind(41)
    alias Protocol = Kind(42)
    alias Quasiquoter = Kind(43)
    alias SelfParameter = Kind(44)
    alias Setter = Kind(45)
    alias Signature = Kind(46)
    alias Subscript = Kind(47)
    alias String = Kind(48)
    alias Struct = Kind(49)
    alias Tactic = Kind(50)
    alias Theorem = Kind(51)
    alias ThisParameter = Kind(52)
    alias Trait = Kind(53)
    alias Type = Kind(54)
    alias TypeAlias = Kind(55)
    alias TypeClass = Kind(56)
    alias TypeFamily = Kind(57)
    alias TypeParameter = Kind(58)
    alias Union = Kind(59)
    alias Value = Kind(60)
    alias Variable = Kind(61)
    alias Contract = Kind(62)
    alias Error = Kind(63)
    alias Library = Kind(64)
    alias Modifier = Kind(65)
    alias AbstractMethod = Kind(66)
    alias MethodSpecification = Kind(67)
    alias ProtocolMethod = Kind(68)
    alias PureVirtualMethod = Kind(69)
    alias TraitMethod = Kind(70)
    alias TypeClassMethod = Kind(71)
    alias Accessor = Kind(72)
    alias Delegate = Kind(73)
    alias MethodAlias = Kind(74)
    alias SingletonClass = Kind(75)
    alias SingletonMethod = Kind(76)
    alias StaticDataMember = Kind(77)
    alias StaticEvent = Kind(78)
    alias StaticField = Kind(79)
    alias StaticMethod = Kind(80)
    alias StaticProperty = Kind(81)
    alias StaticVariable = Kind(82)
    alias Extension = Kind(84)
    alias Mixin = Kind(85)
    alias Concept = Kind(86)

    fn __init__(inout self, value: Int = 0):
        self.value = value


@value
struct Language:
    """Programming language identifier."""
    var value: Int

    alias Unspecified = Language(0)
    alias CSharp = Language(1)
    alias Swift = Language(2)
    alias Dart = Language(3)
    alias Kotlin = Language(4)
    alias Scala = Language(5)
    alias Java = Language(6)
    alias Groovy = Language(7)
    alias Clojure = Language(8)
    alias CommonLisp = Language(9)
    alias Scheme = Language(10)
    alias Racket = Language(11)
    alias Lua = Language(12)
    alias Perl = Language(13)
    alias Raku = Language(14)
    alias Python = Language(15)
    alias Ruby = Language(16)
    alias Elixir = Language(17)
    alias Erlang = Language(18)
    alias PHP = Language(19)
    alias Hack = Language(20)
    alias Coffeescript = Language(21)
    alias JavaScript = Language(22)
    alias TypeScript = Language(23)
    alias Flow = Language(24)
    alias Vue = Language(25)
    alias CSS = Language(26)
    alias Less = Language(27)
    alias Sass = Language(28)
    alias SCSS = Language(29)
    alias HTML = Language(30)
    alias XML = Language(31)
    alias XSL = Language(32)
    alias Go = Language(33)
    alias C = Language(34)
    alias CPP = Language(35)
    alias Objective_C = Language(36)
    alias Objective_CPP = Language(37)
    alias Zig = Language(38)
    alias Ada = Language(39)
    alias Rust = Language(40)
    alias OCaml = Language(41)
    alias FSharp = Language(42)
    alias SML = Language(43)
    alias Haskell = Language(44)
    alias Agda = Language(45)
    alias Idris = Language(46)
    alias Coq = Language(47)
    alias Lean = Language(48)
    alias APL = Language(49)
    alias Dyalog = Language(50)
    alias J = Language(51)
    alias Matlab = Language(52)
    alias Wolfram = Language(53)
    alias R = Language(54)
    alias Julia = Language(55)
    alias Fortran = Language(56)
    alias Delphi = Language(57)
    alias Assembly = Language(58)
    alias COBOL = Language(59)
    alias ABAP = Language(60)
    alias SAS = Language(61)
    alias Razor = Language(62)
    alias VisualBasic = Language(63)
    alias ShellScript = Language(64)
    alias Fish = Language(65)
    alias Awk = Language(66)
    alias PowerShell = Language(67)
    alias Bat = Language(68)
    alias SQL = Language(69)
    alias PLSQL = Language(70)
    alias Prolog = Language(71)
    alias Ini = Language(72)
    alias TOML = Language(73)
    alias YAML = Language(74)
    alias JSON = Language(75)
    alias Jsonnet = Language(76)
    alias Nix = Language(77)
    alias Skylark = Language(78)
    alias Makefile = Language(79)
    alias Dockerfile = Language(80)
    alias BibTeX = Language(81)
    alias TeX = Language(82)
    alias LaTeX = Language(83)
    alias Markdown = Language(84)
    alias ReST = Language(85)
    alias AsciiDoc = Language(86)
    alias Diff = Language(88)
    alias Git_Config = Language(89)
    alias Handlebars = Language(90)
    alias Git_Commit = Language(91)
    alias Git_Rebase = Language(92)
    alias JavaScriptReact = Language(93)
    alias TypeScriptReact = Language(94)
    alias Solidity = Language(95)
    alias Apex = Language(96)
    alias CUDA = Language(97)
    alias GraphQL = Language(98)
    alias Pascal = Language(99)
    alias Protobuf = Language(100)
    alias Tcl = Language(101)
    alias Repro = Language(102)
    alias Thrift = Language(103)
    alias Verilog = Language(104)
    alias VHDL = Language(105)
    alias Svelte = Language(106)
    alias Slang = Language(107)
    alias Luau = Language(108)
    alias Justfile = Language(109)
    alias Nickel = Language(110)

    fn __init__(inout self, value: Int = 0):
        self.value = value



# ========== Core Structs ==========

@value
struct ToolInfo:
    """Information about the tool that produced the index."""
    var name: String
    var version: String
    var arguments: List[String]

    fn __init__(inout self, name: String = "", version: String = ""):
        self.name = name
        self.version = version
        self.arguments = List[String]()

    fn __init__(inout self, name: String, version: String, arguments: List[String]):
        self.name = name
        self.version = version
        self.arguments = arguments


@value
struct Package:
    """Unit of packaging and distribution (module in Go/JVM)."""
    var manager: String
    var name: String
    var version: String

    fn __init__(inout self, manager: String = "", name: String = "", version: String = ""):
        self.manager = manager
        self.name = name
        self.version = version


@value
struct Descriptor:
    """Symbol descriptor component."""
    var name: String
    var disambiguator: String
    var suffix: Suffix

    fn __init__(inout self, name: String = "", disambiguator: String = "", suffix: Suffix = Suffix.Unspecified):
        self.name = name
        self.disambiguator = disambiguator
        self.suffix = suffix


@value
struct Symbol:
    """Symbol identifier (similar to a URI for classes, methods, locals)."""
    var scheme: String
    var package: Package
    var descriptors: List[Descriptor]

    fn __init__(inout self, scheme: String = ""):
        self.scheme = scheme
        self.package = Package()
        self.descriptors = List[Descriptor]()

    fn __init__(inout self, scheme: String, package: Package, descriptors: List[Descriptor]):
        self.scheme = scheme
        self.package = package
        self.descriptors = descriptors


@value
struct Metadata:
    """Metadata about a SCIP index."""
    var version: ProtocolVersion
    var tool_info: ToolInfo
    var project_root: String
    var text_document_encoding: TextEncoding

    fn __init__(inout self):
        self.version = ProtocolVersion.Unspecified
        self.tool_info = ToolInfo()
        self.project_root = ""
        self.text_document_encoding = TextEncoding.Unspecified

    fn __init__(inout self, version: ProtocolVersion, tool_info: ToolInfo,
                project_root: String, text_document_encoding: TextEncoding):
        self.version = version
        self.tool_info = tool_info
        self.project_root = project_root
        self.text_document_encoding = text_document_encoding


@value
struct Diagnostic:
    """A diagnostic (error, warning, etc.) reported for a range."""
    var severity: Severity
    var code: String
    var message: String
    var source: String
    var tags: List[DiagnosticTag]

    fn __init__(inout self, message: String = ""):
        self.severity = Severity.Unspecified
        self.code = ""
        self.message = message
        self.source = ""
        self.tags = List[DiagnosticTag]()

    fn __init__(inout self, severity: Severity, code: String, message: String,
                source: String, tags: List[DiagnosticTag]):
        self.severity = severity
        self.code = code
        self.message = message
        self.source = source
        self.tags = tags


@value
struct Relationship:
    """Relationship between symbols (implements, type definition, etc.)."""
    var symbol: String
    var is_reference: Bool
    var is_implementation: Bool
    var is_type_definition: Bool
    var is_definition: Bool

    fn __init__(inout self, symbol: String = ""):
        self.symbol = symbol
        self.is_reference = False
        self.is_implementation = False
        self.is_type_definition = False
        self.is_definition = False

    fn __init__(inout self, symbol: String, is_reference: Bool, is_implementation: Bool,
                is_type_definition: Bool, is_definition: Bool):
        self.symbol = symbol
        self.is_reference = is_reference
        self.is_implementation = is_implementation
        self.is_type_definition = is_type_definition
        self.is_definition = is_definition


@value
struct Occurrence:
    """Associates a source position with a symbol and highlighting info."""
    var range: List[Int32]
    var symbol: String
    var symbol_roles: Int32
    var override_documentation: List[String]
    var syntax_kind: SyntaxKind
    var diagnostics: List[Diagnostic]
    var enclosing_range: List[Int32]

    fn __init__(inout self, symbol: String = ""):
        self.range = List[Int32]()
        self.symbol = symbol
        self.symbol_roles = 0
        self.override_documentation = List[String]()
        self.syntax_kind = SyntaxKind.Unspecified
        self.diagnostics = List[Diagnostic]()
        self.enclosing_range = List[Int32]()

    fn has_role(self, role: SymbolRole) -> Bool:
        """Check if this occurrence has a specific role."""
        return (self.symbol_roles & role.value) != 0

    fn is_definition(self) -> Bool:
        """Check if this occurrence is a definition."""
        return self.has_role(SymbolRole.Definition)

    fn start_line(self) -> Int32:
        """Get the start line (0-based)."""
        if len(self.range) >= 1:
            return self.range[0]
        return 0

    fn start_character(self) -> Int32:
        """Get the start character (0-based)."""
        if len(self.range) >= 2:
            return self.range[1]
        return 0

    fn end_line(self) -> Int32:
        """Get the end line (0-based)."""
        if len(self.range) == 4:
            return self.range[2]
        elif len(self.range) >= 1:
            return self.range[0]  # Same as start line for 3-element range
        return 0

    fn end_character(self) -> Int32:
        """Get the end character (0-based)."""
        if len(self.range) == 4:
            return self.range[3]
        elif len(self.range) == 3:
            return self.range[2]
        return 0



@value
struct SignatureDocumentation:
    """Signature documentation for a symbol (subset of Document)."""
    var language: String
    var text: String

    fn __init__(inout self, language: String = "", text: String = ""):
        self.language = language
        self.text = text


@value
struct SymbolInformation:
    """Metadata about a symbol (docstring, relationships, kind)."""
    var symbol: String
    var documentation: List[String]
    var relationships: List[Relationship]
    var kind: Kind
    var display_name: String
    var signature_documentation: Optional[SignatureDocumentation]
    var enclosing_symbol: String

    fn __init__(inout self, symbol: String = ""):
        self.symbol = symbol
        self.documentation = List[String]()
        self.relationships = List[Relationship]()
        self.kind = Kind.Unspecified
        self.display_name = ""
        self.signature_documentation = None
        self.enclosing_symbol = ""

    fn __init__(inout self, symbol: String, documentation: List[String],
                relationships: List[Relationship], kind: Kind, display_name: String):
        self.symbol = symbol
        self.documentation = documentation
        self.relationships = relationships
        self.kind = kind
        self.display_name = display_name
        self.signature_documentation = None
        self.enclosing_symbol = ""


@value
struct Document:
    """Metadata about a source file."""
    var language: String
    var relative_path: String
    var occurrences: List[Occurrence]
    var symbols: List[SymbolInformation]
    var text: String
    var position_encoding: PositionEncoding

    fn __init__(inout self, relative_path: String = ""):
        self.language = ""
        self.relative_path = relative_path
        self.occurrences = List[Occurrence]()
        self.symbols = List[SymbolInformation]()
        self.text = ""
        self.position_encoding = PositionEncoding.Unspecified

    fn __init__(inout self, language: String, relative_path: String,
                occurrences: List[Occurrence], symbols: List[SymbolInformation],
                text: String, position_encoding: PositionEncoding):
        self.language = language
        self.relative_path = relative_path
        self.occurrences = occurrences
        self.symbols = symbols
        self.text = text
        self.position_encoding = position_encoding

    fn definition_count(self) -> Int:
        """Count the number of definition occurrences."""
        var count = 0
        for i in range(len(self.occurrences)):
            if self.occurrences[i].is_definition():
                count += 1
        return count

    fn reference_count(self) -> Int:
        """Count the number of reference occurrences."""
        var count = 0
        for i in range(len(self.occurrences)):
            if not self.occurrences[i].is_definition():
                count += 1
        return count


@value
struct Index:
    """Complete SCIP index for a workspace rooted at a single directory."""
    var metadata: Metadata
    var documents: List[Document]
    var external_symbols: List[SymbolInformation]

    fn __init__(inout self):
        self.metadata = Metadata()
        self.documents = List[Document]()
        self.external_symbols = List[SymbolInformation]()

    fn __init__(inout self, metadata: Metadata, documents: List[Document],
                external_symbols: List[SymbolInformation]):
        self.metadata = metadata
        self.documents = documents
        self.external_symbols = external_symbols

    fn document_count(self) -> Int:
        """Get the number of documents in the index."""
        return len(self.documents)

    fn total_occurrences(self) -> Int:
        """Get the total number of occurrences across all documents."""
        var count = 0
        for i in range(len(self.documents)):
            count += len(self.documents[i].occurrences)
        return count

    fn total_symbols(self) -> Int:
        """Get the total number of symbols across all documents."""
        var count = 0
        for i in range(len(self.documents)):
            count += len(self.documents[i].symbols)
        return count + len(self.external_symbols)
