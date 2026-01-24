# Lean4 Client (Zig)

Zig SDK for the Lean4 HTTP service.

## Build

```bash
cd src/serviceCore/nLocalModels/integrations/lean4_client
zig build
```

## Zig Usage

```zig
const lean4 = @import("lean4_client.zig");

var client = try lean4.Lean4Client.init(allocator, "127.0.0.1", 8002);
defer client.deinit();

var response = try client.check(.{ .code = "#check 1" });
defer client.freeCheckResponse(&response);
```

## C ABI

The shared library exports:

```text
lean4_client_create(host, port)
lean4_client_destroy(client)
lean4_check_json(client, code)
lean4_run_json(client, code, stdin)
lean4_request_json(client, endpoint, payload_json)
lean4_free_json(ptr)
```
