# Trial Balance - nServices Integration Layer

This module provides integration with all nServices through SAP BTP Destination services.

## Overview

The integration layer handles communication between the Trial Balance application and the following services:
- **nAgentFlow**: Workflow orchestration
- **nAgentMeta**: Metadata management
- **nLocalModels**: AI inference and narrative generation
- **nGrounding**: Context enrichment
- **HANA Cloud**: Data persistence

## Architecture

```
Trial Balance Backend
    ↓
Destination Client (OAuth2 Authentication)
    ↓
SAP BTP Destination Service
    ↓
[nAgentFlow | nAgentMeta | nLocalModels | nGrounding | HANA Cloud]
```

## Configuration

All service endpoints are configured in the SAP BTP Cockpit as Destinations. The application uses the Destination service to dynamically resolve and authenticate to these services.

### Required Destinations

1. **AGENT_FLOW**
   - Type: HTTP
   - URL: https://agent-flow.{domain}/api/v1
   - Authentication: OAuth2ClientCredentials

2. **AGENT_META**
   - Type: HTTP
   - URL: https://agent-meta.{domain}/api/v1
   - Authentication: OAuth2ClientCredentials

3. **LOCAL_MODELS**
   - Type: HTTP
   - URL: https://local-models.{domain}/api/v1
   - Authentication: OAuth2ClientCredentials

4. **GROUNDING**
   - Type: HTTP
   - URL: https://grounding.{domain}/api/v1
   - Authentication: OAuth2ClientCredentials

5. **HANA_CLOUD**
   - Type: HTTP
   - Authentication: NoAuthentication (handled by HDI container)

## Usage

### Initialize Client

```zig
const std = @import("std");
const DestinationClient = @import("destination_client.zig").DestinationClient;

var client = try DestinationClient.init(allocator, .LOCAL_MODELS);
defer client.deinit();

// Authenticate
try client.authenticate();

// Make request
const response = try client.post("/inference", request_body);
```

### Service-Specific Clients

```zig
// nLocalModels - AI Narrative Generation
var local_models = try LocalModelsClient.init(allocator);
defer local_models.deinit();

const narrative = try local_models.generateNarrative(trial_balance_data);

// nAgentFlow - Workflow Orchestration
var agent_flow = try AgentFlowClient.init(allocator);
defer agent_flow.deinit();

const workflow_id = try agent_flow.startWorkflow("approval", entry_data);
```

## API Patterns

### Request Format

All requests follow a standard JSON format:

```json
{
  "action": "action_name",
  "data": {
    "param1": "value1",
    "param2": "value2"
  },
  "context": {
    "user_id": "user123",
    "company_code": "1000"
  }
}
```

### Response Format

```json
{
  "status": "success|error",
  "data": {
    "result": "..."
  },
  "message": "Optional message",
  "timestamp": "2026-01-26T10:00:00Z"
}
```

## Error Handling

All integration errors should be properly handled and logged:

```zig
const response = client.post("/endpoint", data) catch |err| {
    std.log.err("Failed to call service: {}", .{err});
    return error.ServiceCallFailed;
};
```

## Security

- All communications use HTTPS
- OAuth2 authentication with client credentials
- Tokens are automatically refreshed
- Sensitive data is never logged

## Testing

Unit tests are provided for each client:

```bash
zig build test
```

## Development

When adding new service integrations:

1. Add destination type to `DestinationType` enum
2. Add base URL mapping in `DestinationClient.init()`
3. Create service-specific client struct
4. Implement required methods
5. Add tests
6. Update documentation

## Troubleshooting

### Authentication Errors
- Verify destination configuration in BTP Cockpit
- Check OAuth2 client credentials
- Ensure token endpoint is accessible

### Connection Errors
- Verify service URLs are correct
- Check network connectivity
- Review proxy settings

### Data Format Errors
- Validate JSON structure
- Check data types match API spec
- Review API version compatibility

## Related Documentation

- [SAP BTP Destination Service](https://help.sap.com/docs/CP_CONNECTIVITY/cca91383641e40ffbe03bdc78f00f681/e4f1d97cbb571014a247d10f9f9a685d.html)
- [nAgentFlow API Reference](../../../serviceCore/nAgentFlow/README.md)
- [nLocalModels API Reference](../../../serviceCore/nLocalModels/README.md)