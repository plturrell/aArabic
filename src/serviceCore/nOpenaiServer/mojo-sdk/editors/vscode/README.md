# Mojo Language Support for VS Code

Official VS Code extension for the Mojo programming language, providing rich language support through the Language Server Protocol (LSP).

## Features

### 🎯 Core Language Features

- **Syntax Highlighting** - Full syntax highlighting for Mojo code
- **Code Completion** - Intelligent autocomplete for functions, variables, and types
- **Hover Information** - View type information and documentation on hover
- **Signature Help** - Parameter hints while typing function calls
- **Go to Definition** - Navigate to symbol definitions
- **Find All References** - Find all usages of a symbol
- **Rename Symbol** - Rename symbols across the workspace
- **Diagnostics** - Real-time error and warning detection

### 🔧 Code Actions & Refactoring

- **Quick Fixes** - Automated fixes for common errors
- **Extract Function** - Extract selected code to a new function
- **Extract Variable** - Extract expressions to variables
- **Inline** - Inline functions and variables
- **Organize Imports** - Automatically organize import statements

### 📝 Editor Features

- **Auto-closing Brackets** - Automatic bracket, parenthesis, and quote pairing
- **Smart Indentation** - Context-aware indentation
- **Code Folding** - Collapse/expand code blocks
- **Comment Toggling** - Quick line and block comments

## Requirements

- **VS Code** version 1.75.0 or higher
- **Mojo LSP Server** - The extension will auto-detect the server or you can configure the path

## Installation

### From VS Code Marketplace

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X / Cmd+Shift+X)
3. Search for "Mojo Language Support"
4. Click Install

### From VSIX

```bash
code --install-extension mojo-language-support-0.1.0.vsix
```

## Configuration

Configure the extension through VS Code settings:

### LSP Server

```json
{
  // Enable/disable LSP server
  "mojo.lsp.enabled": true,
  
  // Path to mojo-lsp executable (leave empty for auto-detection)
  "mojo.lsp.serverPath": "",
  
  // Trace LSP communication for debugging
  "mojo.lsp.trace.server": "off" // "off" | "messages" | "verbose"
}
```

### Feature Toggles

```json
{
  "mojo.diagnostics.enabled": true,
  "mojo.completion.enabled": true,
  "mojo.hover.enabled": true,
  "mojo.signatureHelp.enabled": true,
  "mojo.formatting.enabled": true
}
```

## Commands

- **Mojo: Restart Language Server** - Restart the LSP server
- **Mojo: Show Output Channel** - Show the LSP server output

## Keyboard Shortcuts

- `Ctrl+Space` / `Cmd+Space` - Trigger code completion
- `Ctrl+Shift+Space` / `Cmd+Shift+Space` - Trigger signature help
- `F12` - Go to definition
- `Shift+F12` - Find all references
- `F2` - Rename symbol
- `Ctrl+.` / `Cmd+.` - Show code actions

## File Extensions

The extension activates for files with the following extensions:
- `.mojo` - Mojo source files
- `.🔥` - Alternative Mojo extension (fire emoji)

## Troubleshooting

### LSP Server Not Found

If you see "Mojo LSP server executable not found":

1. Install the Mojo LSP server
2. Configure the path in settings: `mojo.lsp.serverPath`
3. Restart VS Code

### Server Not Starting

1. Check the Output channel: **Mojo: Show Output Channel**
2. Verify server path is correct
3. Try restarting the server: **Mojo: Restart Language Server**

### Performance Issues

1. Disable unused features in settings
2. Check LSP trace output for errors
3. Report issues with logs

## Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/mojo-lang/mojo-lsp.git
cd mojo-lsp/editors/vscode

# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Watch for changes
npm run watch
```

### Testing

```bash
npm test
```

### Packaging

```bash
# Install vsce
npm install -g @vscode/vsce

# Package extension
vsce package
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

## License

Apache-2.0

## Links

- [Repository](https://github.com/mojo-lang/mojo-lsp)
- [Issues](https://github.com/mojo-lang/mojo-lsp/issues)
- [Mojo Documentation](https://docs.modular.com/mojo/)

## Changelog

### 0.1.0 (Initial Release)

- Initial release with core LSP features
- Syntax highlighting
- Code completion
- Hover information
- Signature help
- Go to definition
- Find references
- Rename symbol
- Diagnostics
- Code actions and refactoring

---

**Enjoy coding in Mojo!** 🔥
