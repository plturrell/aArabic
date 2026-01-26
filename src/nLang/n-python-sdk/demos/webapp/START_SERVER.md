same # Intelligence SDK Server Startup Guide

## Quick Start (Zig Server - Recommended)

To start the Intelligence SDK Zig HTTP server on port 8084:

```bash
cd /Users/user/Documents/arabic_folder/src/nLang/n-python-sdk/demos
zig build run
```

The server will be available at: `http://localhost:8084`

## Prerequisites

- Zig 0.15+ installed
- No additional dependencies needed!

## Why Zig?

- **Fast compilation**: Zig compiles directly to native code
- **Zero dependencies**: No Node.js or npm packages required
- **Memory efficient**: Native performance with low overhead
- **Integrated**: Matches the n-c-sdk architecture

## Build Only (Without Running)

```bash
cd /Users/user/Documents/arabic_folder/src/nLang/n-python-sdk/demos
zig build
```

Then run the compiled binary:
```bash
./zig-out/bin/intelligence-sdk-server
```

## Troubleshooting

### Port Already in Use
If port 8084 is already in use, edit `server.zig` and change the PORT constant, then rebuild.

### Zig Not Found
Install Zig from: https://ziglang.org/download/

### Build Errors
Clean the build cache:
```bash
rm -rf .zig-cache zig-out
zig build run
```

## Integration with Product Switch

Once running, the Intelligence SDK will be accessible from:
- n-c-sdk Product Switch → "⚛ intelligence sdk"
- All service Product Switches → "⚛ intelligence sdk"

The Intelligence SDK's Product Switch allows navigation to:
- process sdk (port 8080)
- nLocalModels (port 8081)
- nAgentFlow (port 8082)
- nGrounding (port 8083)