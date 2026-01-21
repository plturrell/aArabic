"""
Supported SCIP-producing indexers for nCode.

This module defines constants for supported language-specific SCIP indexers
and provides helper functions to run them.
"""

from collections import Dict


# ============================================================================
# Supported SCIP Indexers
# ============================================================================

# Core SCIP indexers
alias SCIP_TYPESCRIPT = "scip-typescript"
alias SCIP_PYTHON = "scip-python"
alias SCIP_JAVA = "scip-java"
alias SCIP_RUST = "rust-analyzer"  # Via rust-analyzer's SCIP export
alias SCIP_GO = "scip-go"
alias SCIP_DOTNET = "scip-dotnet"

# Additional programming language indexers
alias SCIP_RUBY = "scip-ruby"
alias SCIP_KOTLIN = "scip-kotlin"
alias SCIP_SCALA = "scip-scala"
alias SCIP_PHP = "scip-php"
alias SCIP_CLANG = "scip-clang"
alias SCIP_SQL = "scip-sql"

# LSP-based indexers
alias LSP_SWIFT = "sourcekit-lsp"
alias LSP_OBJC = "clangd"
alias LSP_PERL = "pls"
alias LSP_LUA = "lua-language-server"
alias LSP_HASKELL = "haskell-language-server"
alias LSP_ELIXIR = "elixir-ls"
alias LSP_ERLANG = "erlang_ls"
alias LSP_OCAML = "ocaml-lsp"
alias LSP_ZIG = "zls"
alias LSP_MOJO = "mojo-lsp"

# Tree-sitter based indexers (for data/config/markup languages)
alias TREESITTER_INDEXER = "ncode-treesitter"


# ============================================================================
# Language to Indexer Mapping
# ============================================================================

@value
struct IndexerInfo:
    """Information about a SCIP indexer."""
    var name: String
    var command: String
    var file_extensions: String
    var install_hint: String

    fn __init__(
        inout self,
        name: String,
        command: String,
        file_extensions: String,
        install_hint: String
    ):
        self.name = name
        self.command = command
        self.file_extensions = file_extensions
        self.install_hint = install_hint


fn get_supported_languages() -> List[String]:
    """Get a list of all supported languages."""
    var langs = List[String]()

    # Core languages with dedicated SCIP indexers
    langs.append("typescript")
    langs.append("javascript")
    langs.append("python")
    langs.append("java")
    langs.append("rust")
    langs.append("go")
    langs.append("csharp")
    langs.append("fsharp")
    langs.append("vb")

    # Additional programming languages
    langs.append("ruby")
    langs.append("kotlin")
    langs.append("scala")
    langs.append("php")
    langs.append("swift")
    langs.append("objective-c")
    langs.append("c")
    langs.append("cpp")
    langs.append("perl")
    langs.append("lua")
    langs.append("haskell")
    langs.append("elixir")
    langs.append("erlang")
    langs.append("ocaml")
    langs.append("zig")
    langs.append("mojo")

    # Data/Config languages (tree-sitter based)
    langs.append("json")
    langs.append("xml")
    langs.append("yaml")
    langs.append("toml")
    langs.append("sql")
    langs.append("graphql")
    langs.append("protobuf")
    langs.append("thrift")

    # Markup/Doc languages (tree-sitter based)
    langs.append("markdown")
    langs.append("html")
    langs.append("css")
    langs.append("scss")
    langs.append("less")

    return langs


fn get_indexer_info(language: String) -> IndexerInfo:
    """Get indexer information for a specific language.
    
    Args:
        language: Language identifier (e.g., "typescript", "python")
        
    Returns:
        IndexerInfo with command and metadata for the indexer
    """
    var lang_lower = language.lower()
    
    # TypeScript/JavaScript
    if lang_lower == "typescript" or lang_lower == "javascript" or lang_lower == "tsx" or lang_lower == "jsx":
        return IndexerInfo(
            name=SCIP_TYPESCRIPT,
            command="npx @sourcegraph/scip-typescript index --output index.scip",
            file_extensions=".ts,.tsx,.js,.jsx",
            install_hint="npm install -g @sourcegraph/scip-typescript"
        )
    
    # Python
    if lang_lower == "python":
        return IndexerInfo(
            name=SCIP_PYTHON,
            command="scip-python index . --output index.scip",
            file_extensions=".py,.pyi",
            install_hint="pip install scip-python"
        )
    
    # Java
    if lang_lower == "java":
        return IndexerInfo(
            name=SCIP_JAVA,
            command="scip-java index --output index.scip",
            file_extensions=".java",
            install_hint="See https://sourcegraph.github.io/scip-java/"
        )
    
    # Rust (via rust-analyzer)
    if lang_lower == "rust":
        return IndexerInfo(
            name=SCIP_RUST,
            command="rust-analyzer scip . --output index.scip",
            file_extensions=".rs",
            install_hint="Install rust-analyzer via rustup component add rust-analyzer"
        )
    
    # Go
    if lang_lower == "go" or lang_lower == "golang":
        return IndexerInfo(
            name=SCIP_GO,
            command="scip-go --output index.scip",
            file_extensions=".go",
            install_hint="go install github.com/sourcegraph/scip-go/cmd/scip-go@latest"
        )
    
    # .NET (C#, F#, VB)
    if lang_lower == "csharp" or lang_lower == "c#" or lang_lower == "fsharp" or lang_lower == "f#" or lang_lower == "vb" or lang_lower == "dotnet":
        return IndexerInfo(
            name=SCIP_DOTNET,
            command="scip-dotnet index --output index.scip",
            file_extensions=".cs,.fs,.vb",
            install_hint="dotnet tool install -g scip-dotnet"
        )

    # =========================================================================
    # Additional Programming Languages
    # =========================================================================

    # Ruby
    if lang_lower == "ruby" or lang_lower == "rb":
        return IndexerInfo(
            name=SCIP_RUBY,
            command="scip-ruby --output index.scip",
            file_extensions=".rb,.rake,.gemspec",
            install_hint="gem install scip-ruby"
        )

    # Kotlin (via scip-java with Kotlin support)
    if lang_lower == "kotlin" or lang_lower == "kt":
        return IndexerInfo(
            name=SCIP_KOTLIN,
            command="scip-java index --build-tool gradle --output index.scip",
            file_extensions=".kt,.kts",
            install_hint="See https://sourcegraph.github.io/scip-java/ (supports Kotlin)"
        )

    # Scala (via metals LSP)
    if lang_lower == "scala":
        return IndexerInfo(
            name=SCIP_SCALA,
            command="metals-scip . --output index.scip",
            file_extensions=".scala,.sc,.sbt",
            install_hint="Install Metals LSP: coursier install metals"
        )

    # PHP
    if lang_lower == "php":
        return IndexerInfo(
            name=SCIP_PHP,
            command="scip-php index --output index.scip",
            file_extensions=".php,.phtml",
            install_hint="composer global require sourcegraph/scip-php"
        )

    # Swift (via sourcekit-lsp)
    if lang_lower == "swift":
        return IndexerInfo(
            name=LSP_SWIFT,
            command="sourcekit-lsp-scip --output index.scip",
            file_extensions=".swift",
            install_hint="Install Xcode or swift toolchain (includes sourcekit-lsp)"
        )

    # Objective-C (via clangd)
    if lang_lower == "objective-c" or lang_lower == "objc" or lang_lower == "objectivec":
        return IndexerInfo(
            name=LSP_OBJC,
            command="clangd-indexer --output index.scip",
            file_extensions=".m,.mm,.h",
            install_hint="Install clangd via LLVM or Xcode"
        )

    # C/C++ (via scip-clang)
    if lang_lower == "c" or lang_lower == "cpp" or lang_lower == "c++" or lang_lower == "cxx":
        return IndexerInfo(
            name=SCIP_CLANG,
            command="scip-clang --compdb compile_commands.json --output index.scip",
            file_extensions=".c,.cpp,.cc,.cxx,.h,.hpp,.hxx",
            install_hint="Build scip-clang from source: https://github.com/sourcegraph/scip-clang"
        )

    # Perl (via PLS - Perl Language Server)
    if lang_lower == "perl" or lang_lower == "pl":
        return IndexerInfo(
            name=LSP_PERL,
            command="pls-scip --output index.scip",
            file_extensions=".pl,.pm,.pod",
            install_hint="cpanm PLS (Perl Language Server)"
        )

    # Lua (via lua-language-server)
    if lang_lower == "lua":
        return IndexerInfo(
            name=LSP_LUA,
            command="lua-language-server --scip --output index.scip",
            file_extensions=".lua",
            install_hint="Install lua-language-server: https://github.com/LuaLS/lua-language-server"
        )

    # Haskell (via haskell-language-server)
    if lang_lower == "haskell" or lang_lower == "hs":
        return IndexerInfo(
            name=LSP_HASKELL,
            command="haskell-language-server-scip --output index.scip",
            file_extensions=".hs,.lhs",
            install_hint="ghcup install hls"
        )

    # Elixir (via elixir-ls)
    if lang_lower == "elixir" or lang_lower == "ex":
        return IndexerInfo(
            name=LSP_ELIXIR,
            command="elixir-ls-scip --output index.scip",
            file_extensions=".ex,.exs",
            install_hint="Install elixir-ls: https://github.com/elixir-lsp/elixir-ls"
        )

    # Erlang (via erlang_ls)
    if lang_lower == "erlang" or lang_lower == "erl":
        return IndexerInfo(
            name=LSP_ERLANG,
            command="erlang_ls-scip --output index.scip",
            file_extensions=".erl,.hrl",
            install_hint="Install erlang_ls: https://github.com/erlang-ls/erlang_ls"
        )

    # OCaml (via ocaml-lsp)
    if lang_lower == "ocaml" or lang_lower == "ml":
        return IndexerInfo(
            name=LSP_OCAML,
            command="ocaml-lsp-scip --output index.scip",
            file_extensions=".ml,.mli",
            install_hint="opam install ocaml-lsp-server"
        )

    # Zig (via zls)
    if lang_lower == "zig":
        return IndexerInfo(
            name=LSP_ZIG,
            command="zls-scip --output index.scip",
            file_extensions=".zig",
            install_hint="Install zls: https://github.com/zigtools/zls"
        )

    # Mojo (via mojo lsp)
    if lang_lower == "mojo" or lang_lower == "🔥":
        return IndexerInfo(
            name=LSP_MOJO,
            command="mojo-lsp-scip --output index.scip",
            file_extensions=".mojo,.🔥",
            install_hint="Install Mojo SDK: https://www.modular.com/mojo"
        )

    # =========================================================================
    # Data/Config Languages (tree-sitter based)
    # =========================================================================

    # JSON
    if lang_lower == "json" or lang_lower == "jsonc":
        return IndexerInfo(
            name=TREESITTER_INDEXER,
            command="ncode-treesitter index --language json --output index.scip",
            file_extensions=".json,.jsonc",
            install_hint="pip install ncode-treesitter (tree-sitter-json based)"
        )

    # XML
    if lang_lower == "xml" or lang_lower == "xsl" or lang_lower == "xslt":
        return IndexerInfo(
            name=TREESITTER_INDEXER,
            command="ncode-treesitter index --language xml --output index.scip",
            file_extensions=".xml,.xsl,.xslt,.xsd,.svg",
            install_hint="pip install ncode-treesitter (tree-sitter-xml based)"
        )

    # YAML
    if lang_lower == "yaml" or lang_lower == "yml":
        return IndexerInfo(
            name=TREESITTER_INDEXER,
            command="ncode-treesitter index --language yaml --output index.scip",
            file_extensions=".yaml,.yml",
            install_hint="pip install ncode-treesitter (tree-sitter-yaml based)"
        )

    # TOML
    if lang_lower == "toml":
        return IndexerInfo(
            name=TREESITTER_INDEXER,
            command="ncode-treesitter index --language toml --output index.scip",
            file_extensions=".toml",
            install_hint="pip install ncode-treesitter (tree-sitter-toml based)"
        )

    # SQL
    if lang_lower == "sql":
        return IndexerInfo(
            name=SCIP_SQL,
            command="scip-sql index --output index.scip",
            file_extensions=".sql",
            install_hint="pip install scip-sql or use: ncode-treesitter index --language sql"
        )

    # GraphQL
    if lang_lower == "graphql" or lang_lower == "gql":
        return IndexerInfo(
            name=TREESITTER_INDEXER,
            command="ncode-treesitter index --language graphql --output index.scip",
            file_extensions=".graphql,.gql",
            install_hint="pip install ncode-treesitter (tree-sitter-graphql based)"
        )

    # Protobuf
    if lang_lower == "protobuf" or lang_lower == "proto":
        return IndexerInfo(
            name=TREESITTER_INDEXER,
            command="ncode-treesitter index --language protobuf --output index.scip",
            file_extensions=".proto",
            install_hint="pip install ncode-treesitter (tree-sitter-protobuf based)"
        )

    # Thrift
    if lang_lower == "thrift":
        return IndexerInfo(
            name=TREESITTER_INDEXER,
            command="ncode-treesitter index --language thrift --output index.scip",
            file_extensions=".thrift",
            install_hint="pip install ncode-treesitter (tree-sitter-thrift based)"
        )

    # =========================================================================
    # Markup/Doc Languages (tree-sitter based)
    # =========================================================================

    # Markdown
    if lang_lower == "markdown" or lang_lower == "md":
        return IndexerInfo(
            name=TREESITTER_INDEXER,
            command="ncode-treesitter index --language markdown --output index.scip",
            file_extensions=".md,.markdown,.mdx",
            install_hint="pip install ncode-treesitter (tree-sitter-markdown based)"
        )

    # HTML
    if lang_lower == "html" or lang_lower == "htm":
        return IndexerInfo(
            name=TREESITTER_INDEXER,
            command="ncode-treesitter index --language html --output index.scip",
            file_extensions=".html,.htm,.xhtml",
            install_hint="pip install ncode-treesitter (tree-sitter-html based)"
        )

    # CSS
    if lang_lower == "css":
        return IndexerInfo(
            name=TREESITTER_INDEXER,
            command="ncode-treesitter index --language css --output index.scip",
            file_extensions=".css",
            install_hint="pip install ncode-treesitter (tree-sitter-css based)"
        )

    # SCSS
    if lang_lower == "scss" or lang_lower == "sass":
        return IndexerInfo(
            name=TREESITTER_INDEXER,
            command="ncode-treesitter index --language scss --output index.scip",
            file_extensions=".scss,.sass",
            install_hint="pip install ncode-treesitter (tree-sitter-scss based)"
        )

    # LESS
    if lang_lower == "less":
        return IndexerInfo(
            name=TREESITTER_INDEXER,
            command="ncode-treesitter index --language less --output index.scip",
            file_extensions=".less",
            install_hint="pip install ncode-treesitter (tree-sitter-css based)"
        )

    # Unknown language - return empty info
    return IndexerInfo(
        name="",
        command="",
        file_extensions="",
        install_hint="Unknown language: " + language
    )


fn get_indexer_command(language: String) -> String:
    """Get the shell command to run the SCIP indexer for a language.
    
    Args:
        language: Language identifier (e.g., "typescript", "python")
        
    Returns:
        Shell command string to run the indexer, or empty string if unsupported
    """
    var info = get_indexer_info(language)
    return info.command


fn is_language_supported(language: String) -> Bool:
    """Check if a language has a supported SCIP indexer.
    
    Args:
        language: Language identifier
        
    Returns:
        True if the language is supported, False otherwise
    """
    var info = get_indexer_info(language)
    return len(info.name) > 0

