# HyperShimmy Developer Guide

**Version**: 1.0.0  
**Last Updated**: January 16, 2026

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/hypershimmy.git
cd hypershimmy

# Install dependencies
./scripts/setup/install_dependencies.sh

# Build
zig build

# Run tests
zig build test-all

# Start server
zig build run
```

## Prerequisites

### Required

- **Zig**: 0.12.0 or later
- **Mojo**: Latest version
- **Git**: For version control

### Optional

- **Docker**: For containerized services
- **Qdrant**: Vector database
- **Node.js**: For SAPUI5 development

## Installation

### 1. Install Zig

```bash
# macOS
brew install zig

# Linux
curl -O https://ziglang.org/download/0.12.0/zig-linux-x86_64-0.12.0.tar.xz
tar -xf zig-linux-x86_64-0.12.0.tar.xz
export PATH=$PATH:$(pwd)/zig-linux-x86_64-0.12.0
```

### 2. Install Mojo

Follow instructions at [modular.com/mojo](https://www.modular.com/mojo)

### 3. Install Optional Services

```bash
# Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Or use docker-compose
docker-compose up -d
```

## Project Structure

```
hypershimmy/
├── server/          # Zig backend code
├── mojo/            # Mojo AI code
├── io/              # I/O operations
├── webapp/          # SAPUI5 frontend
├── tests/           # Test files
├── scripts/         # Build/deployment scripts
├── docs/            # Documentation
└── build.zig        # Build configuration
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Changes

Edit files in appropriate directories.

### 3. Run Tests

```bash
# Unit tests
zig build test

# Integration tests
zig build test-integration

# All tests
zig build test-all
```

### 4. Format Code

```bash
zig fmt .
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add my feature"
```

### 6. Push and Create PR

```bash
git push origin feature/my-feature
```

## Building

### Debug Build

```bash
zig build
```

### Release Build

```bash
zig build -Doptimize=ReleaseFast
```

### Run Build

```bash
zig build run
```

## Testing

### Run All Tests

```bash
./scripts/run_unit_tests.sh
./scripts/run_integration_tests.sh
```

### Run Specific Test

```bash
zig test tests/unit/test_sources.zig
```

### Add New Test

Create test file in `tests/unit/` or `tests/integration/`:

```zig
const std = @import("std");
const testing = std.testing;

test "my new test" {
    try testing.expectEqual(2, 1 + 1);
}
```

Update `build.zig` to include new test.

## Debugging

### Enable Debug Logging

Set environment variable:

```bash
export LOG_LEVEL=DEBUG
```

### Use Zig Debugger

```bash
zig build -Doptimize=Debug
lldb ./zig-out/bin/hypershimmy
```

### Common Issues

**Issue**: Port already in use  
**Solution**: Kill process on port 8080 or change port

**Issue**: Mojo not found  
**Solution**: Ensure Mojo is in PATH

**Issue**: Build errors  
**Solution**: Check Zig version: `zig version`

## Code Style

### Zig

- Use `snake_case` for functions and variables
- Use `PascalCase` for types
- 4-space indentation
- Run `zig fmt` before committing

### Mojo

- Follow Python PEP 8 style
- Use type annotations
- Document functions with docstrings

## API Development

### Add New Endpoint

1. Define in `server/odata*.zig`
2. Add route in `server/main.zig`
3. Implement handler
4. Add tests
5. Update API.md documentation

### Add New Entity

1. Define struct in `server/entities.zig`
2. Add to metadata
3. Implement CRUD operations
4. Add validation
5. Write tests

## Frontend Development

### SAPUI5 Setup

```bash
cd webapp
npm install
npm start
```

### Add New View

1. Create XML view in `webapp/view/`
2. Create controller in `webapp/controller/`
3. Add route in `manifest.json`
4. Update i18n texts

## Contributing

### Pull Request Process

1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Ensure all tests pass
5. Update documentation
6. Submit PR

### Code Review Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted
- [ ] No compiler warnings
- [ ] Changes described in PR

## Best Practices

1. **Write tests first** (TDD)
2. **Keep functions small** (< 50 lines)
3. **Document public APIs**
4. **Handle errors explicitly**
5. **Use meaningful names**
6. **Avoid premature optimization**
7. **Keep dependencies minimal**

## Resources

- [Zig Documentation](https://ziglang.org/documentation/)
- [Mojo Documentation](https://docs.modular.com/mojo/)
- [SAPUI5 SDK](https://sapui5.hana.ondemand.com/)
- [OData V4 Spec](https://www.odata.org/documentation/)

## Support

- GitHub Issues
- Developer Discord
- Stack Overflow (tag: hypershimmy)

---

**For API reference**, see [API.md](API.md)  
**For architecture**, see [ARCHITECTURE.md](ARCHITECTURE.md)  
**For deployment**, see [DEPLOYMENT.md](DEPLOYMENT.md)
