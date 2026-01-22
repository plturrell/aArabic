#!/bin/bash
# Guardrails Integration Script
# Completes the remaining integration steps for guardrails

set -e

echo "🛡️  Guardrails Integration Script"
echo "================================"

cd "$(dirname "$0")/.."

# Step 1: Verify guardrails validator exists
if [ ! -f "orchestration/agents/guardrails/validator.zig" ]; then
    echo "❌ Error: Guardrails validator not found"
    exit 1
fi

echo "✅ Guardrails validator found"

# Step 2: Create integration patch
cat > /tmp/guardrails_integration.patch << 'EOF'
Add guardrails integration to openai_http_server.zig:

1. After line 10 (imports), add:
   // Orchestration Agents
   const Guardrails = @import("orchestration/agents/guardrails/validator.zig");

2. After line 54 (global HANA store), add:
   // Global guardrails validator
   var guardrails_validator: ?Guardrails.GuardrailsValidator = null;

3. After warmStart() in main() (~line 895), add:
   // Initialize guardrails validator
   guardrails_validator = Guardrails.GuardrailsValidator.init(allocator);
   std.debug.print("🛡️  Guardrails validator initialized\n", .{});

4. In handleChat(), after building prompt (~line 750), add:
   // Guardrails: Input validation
   if (guardrails_validator) |*guard| {
       const input_result = try guard.validateInput(prompt);
       if (!input_result.passed) {
           metrics.recordRequest(.chat, false);
           return Response{ 
               .status = 400, 
               .body = try errorBody(input_result.reason) 
           };
       }
   }

5. In handleChat(), after generateText() (~line 770), add:
   // Guardrails: Output validation
   if (guardrails_validator) |*guard| {
       const output_result = try guard.validateOutput(output);
       if (!output_result.passed) {
           if (output_result.masked_content) |masked| {
               allocator.free(output);
               output = try allocator.dupe(u8, masked);
           } else {
               metrics.recordRequest(.chat, false);
               return Response{ 
                   .status = 400, 
                   .body = try errorBody("Output validation failed") 
               };
           }
       }
   }

6. In handleConnection(), add route (~line 1100):
   } else if (mem.eql(u8, method, "GET") and mem.eql(u8, clean_path, "/v1/guardrails/metrics")) {
       response = try handleGuardrailsMetrics();

7. Add handler function (before main):
   fn handleGuardrailsMetrics() !Response {
       if (guardrails_validator) |*guard| {
           const body = try guard.getMetricsJson(allocator);
           return Response{ .status = 200, .body = body };
       }
       return Response{ 
           .status = 503, 
           .body = try errorBody("Guardrails not initialized") 
       };
   }
EOF

echo "📝 Integration patch created at /tmp/guardrails_integration.patch"
echo ""
echo "⚠️  MANUAL STEPS REQUIRED:"
echo "================================"
echo ""
echo "The openai_http_server.zig file is large. Please manually apply these changes:"
echo ""
cat /tmp/guardrails_integration.patch
echo ""
echo "================================"
echo ""
echo "After making these changes, run:"
echo "  cd src/serviceCore/nOpenaiServer"
echo "  zig build-exe openai_http_server.zig -O ReleaseFast"
echo "  ./openai_http_server"
echo ""
echo "Then test with:"
echo "  curl http://localhost:11434/v1/guardrails/metrics | jq"
echo ""
echo "✅ Integration guide complete!"
