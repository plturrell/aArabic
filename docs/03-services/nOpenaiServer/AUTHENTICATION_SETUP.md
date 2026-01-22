# Authentication Setup Guide

**Purpose:** Integrate user authentication and authorization  
**Status:** Implementation guide for Day 12+

---

## Overview

This guide outlines authentication integration for the nOpenaiServer application, enabling user identification, session management, and secure API access.

## Current Status

### Implemented ✅
- Basic user_id field in database (`demo-user` placeholder)
- HANA Cloud connection with secure credentials
- SSL/TLS encrypted connections

### Pending ⏳
- User authentication system
- Session management
- Authorization/permissions
- JWT token validation

---

## Architecture

### Recommended Approach: SAP Identity Authentication Service (IAS)

```
┌─────────────────┐
│   UI5 Frontend  │
│  (Browser)      │
└────────┬────────┘
         │ 1. Login redirect
         ↓
┌─────────────────┐
│   SAP IAS       │  ← OAuth 2.0 / SAML 2.0
│  (Authentication)│
└────────┬────────┘
         │ 2. ID Token + Access Token
         ↓
┌─────────────────┐
│  HTTP Server    │  ← JWT validation
│  (Zig Backend)  │
└────────┬────────┘
         │ 3. Extract user_id
         ↓
┌─────────────────┐
│   HANA Cloud    │  ← Store with user_id
│  (NUCLEUS.*)    │
└─────────────────┘
```

---

## Implementation Plan

### Phase 1: Frontend Authentication (Day 12)

#### 1.1 Add SAP UI5 Authentication

```javascript
// webapp/Component.js
sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/core/BusyIndicator"
], function (UIComponent, BusyIndicator) {
    "use strict";

    return UIComponent.extend("llm.server.dashboard.Component", {
        
        init: function () {
            UIComponent.prototype.init.apply(this, arguments);
            
            // Initialize authentication
            this._initAuthentication();
        },

        _initAuthentication: function () {
            var that = this;
            
            // Check for existing session
            var token = localStorage.getItem("access_token");
            
            if (!token) {
                // Redirect to SAP IAS login
                this._redirectToLogin();
            } else {
                // Validate token
                this._validateToken(token).then(function (userData) {
                    // Store user data
                    that.getModel("user").setData(userData);
                    
                    // Continue with app initialization
                    that._initializeApp();
                }).catch(function () {
                    // Token invalid, re-authenticate
                    that._redirectToLogin();
                });
            }
        },

        _redirectToLogin: function () {
            var authUrl = "https://<tenant>.authentication.sap.hana.ondemand.com/oauth/authorize";
            var clientId = "<your-client-id>";
            var redirectUri = encodeURIComponent(window.location.origin + "/callback");
            
            window.location.href = authUrl + 
                "?response_type=code" +
                "&client_id=" + clientId +
                "&redirect_uri=" + redirectUri +
                "&scope=openid email profile";
        },

        _validateToken: function (token) {
            return fetch("/api/v1/auth/validate", {
                headers: {
                    "Authorization": "Bearer " + token
                }
            }).then(function (response) {
                if (!response.ok) throw new Error("Token invalid");
                return response.json();
            });
        }
    });
});
```

#### 1.2 OAuth Callback Handler

```javascript
// webapp/controller/Callback.controller.js
sap.ui.define([
    "sap/ui/core/mvc/Controller"
], function (Controller) {
    "use strict";

    return Controller.extend("llm.server.dashboard.controller.Callback", {
        
        onInit: function () {
            // Extract authorization code from URL
            var urlParams = new URLSearchParams(window.location.search);
            var code = urlParams.get("code");
            
            if (code) {
                this._exchangeCodeForToken(code);
            } else {
                // Handle error
                this._handleAuthError();
            }
        },

        _exchangeCodeForToken: function (code) {
            var that = this;
            
            fetch("/api/v1/auth/token", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    code: code,
                    redirect_uri: window.location.origin + "/callback"
                })
            })
            .then(function (response) {
                return response.json();
            })
            .then(function (data) {
                // Store tokens
                localStorage.setItem("access_token", data.access_token);
                localStorage.setItem("refresh_token", data.refresh_token);
                localStorage.setItem("user_id", data.user_id);
                
                // Redirect to main app
                window.location.href = "/";
            })
            .catch(function (error) {
                that._handleAuthError(error);
            });
        }
    });
});
```

### Phase 2: Backend JWT Validation (Day 12)

#### 2.1 Add JWT Library to Zig

Add dependency in `build.zig`:
```zig
// JWT validation library
const jwt_dep = b.dependency("zig-jwt", .{});
exe.root_module.addImport("jwt", jwt_dep.module("jwt"));
```

#### 2.2 Create Auth Middleware

```zig
// auth/jwt_validator.zig
const std = @import("std");
const jwt = @import("jwt");

pub const UserContext = struct {
    user_id: []const u8,
    email: []const u8,
    roles: []const []const u8,
};

pub fn validateToken(
    allocator: std.mem.Allocator,
    token: []const u8,
    public_key: []const u8,
) !UserContext {
    // Decode and verify JWT
    const decoded = try jwt.decode(
        allocator,
        token,
        public_key,
        .RS256,  // RSA signature
    );
    defer decoded.deinit();

    // Extract claims
    const user_id = try decoded.getClaim("sub");
    const email = try decoded.getClaim("email");
    
    return UserContext{
        .user_id = try allocator.dupe(u8, user_id),
        .email = try allocator.dupe(u8, email),
        .roles = &[_][]const u8{},  // TODO: Extract roles
    };
}
```

#### 2.3 Update HTTP Handlers

```zig
// In openai_http_server.zig
fn handleSavePrompt(
    allocator: mem.Allocator,
    request: *HttpRequest,
) ![]const u8 {
    // Extract and validate JWT
    const auth_header = request.getHeader("Authorization") orelse {
        return error.Unauthorized;
    };
    
    if (!mem.startsWith(u8, auth_header, "Bearer ")) {
        return error.InvalidAuthHeader;
    }
    
    const token = auth_header[7..];  // Skip "Bearer "
    const user_ctx = try validateToken(allocator, token, public_key);
    defer user_ctx.deinit();
    
    // Use real user_id instead of "demo-user"
    const prompt_data = PromptRecord{
        .prompt_text = request.body.prompt_text,
        .model_name = request.body.model_name,
        .user_id = user_ctx.user_id,  // ← Real user ID
        .prompt_mode_id = request.body.prompt_mode_id,
        .tags = request.body.tags,
    };
    
    // Save with authenticated user
    const prompt_id = try PromptHistory.savePrompt(
        allocator,
        hana_config,
        prompt_data,
    );
    
    // Return success
    // ...
}
```

### Phase 3: Frontend Integration (Day 12)

#### 3.1 Update PromptTesting Controller

```javascript
// Replace hardcoded "demo-user"
onSaveToHistory: function () {
    var oData = this._oTestModel.getData();
    
    if (!oData.response) {
        MessageToast.show("No response to save");
        return;
    }
    
    // Get authenticated user ID
    var sUserId = localStorage.getItem("user_id") || "anonymous";
    var sAccessToken = localStorage.getItem("access_token");
    
    var oPromptData = {
        prompt_text: oData.promptText,
        model_name: oPreset.model_id,
        user_id: sUserId,  // ← From authentication
        prompt_mode_id: this._getModeId(oData.selectedMode),
        tags: oData.selectedMode
    };
    
    // Include JWT in request
    fetch("/api/v1/prompts", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + sAccessToken
        },
        body: JSON.stringify(oPromptData)
    })
    // ...
}
```

#### 3.2 Token Refresh Logic

```javascript
_refreshToken: function () {
    var that = this;
    var refreshToken = localStorage.getItem("refresh_token");
    
    return fetch("/api/v1/auth/refresh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            refresh_token: refreshToken
        })
    })
    .then(function (response) {
        return response.json();
    })
    .then(function (data) {
        localStorage.setItem("access_token", data.access_token);
        return data.access_token;
    });
}
```

---

## Alternative: Keycloak Integration

If using Keycloak instead of SAP IAS:

### Configuration

```javascript
// webapp/keycloak-config.js
var keycloak = new Keycloak({
    url: 'http://localhost:8080/auth',
    realm: 'nucleus',
    clientId: 'nopenai-frontend'
});

keycloak.init({
    onLoad: 'login-required',
    checkLoginIframe: false
}).then(function (authenticated) {
    if (authenticated) {
        // Store token
        localStorage.setItem("access_token", keycloak.token);
        localStorage.setItem("user_id", keycloak.tokenParsed.sub);
    }
});
```

---

## Security Considerations

### 1. Token Storage
- **Access Token:** localStorage (short-lived, 15 min)
- **Refresh Token:** HttpOnly cookie (long-lived, 7 days)
- **User Data:** Session storage (cleared on tab close)

### 2. CORS Configuration

```zig
// In openai_http_server.zig
fn handleCORS(response: *HttpResponse) void {
    response.setHeader("Access-Control-Allow-Origin", allowed_origin);
    response.setHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    response.setHeader("Access-Control-Allow-Headers", "Authorization, Content-Type");
    response.setHeader("Access-Control-Allow-Credentials", "true");
}
```

### 3. Rate Limiting

```zig
// Per-user rate limiting
const RateLimiter = struct {
    requests: std.StringHashMap(u32),
    window_start: i64,
    
    pub fn checkLimit(self: *RateLimiter, user_id: []const u8) !bool {
        const now = std.time.timestamp();
        
        // Reset window every minute
        if (now - self.window_start > 60) {
            self.requests.clearRetainingCapacity();
            self.window_start = now;
        }
        
        const count = self.requests.get(user_id) orelse 0;
        if (count >= 100) {  // 100 requests per minute
            return error.RateLimitExceeded;
        }
        
        try self.requests.put(user_id, count + 1);
        return true;
    }
};
```

---

## Testing Authentication

### Unit Tests

```zig
// tests/auth_test.zig
test "JWT validation with valid token" {
    const allocator = std.testing.allocator;
    
    const token = "eyJhbGciOiJSUzI1NiIs...";
    const public_key = try readPublicKey(allocator);
    
    const user_ctx = try validateToken(allocator, token, public_key);
    defer user_ctx.deinit();
    
    try std.testing.expectEqualStrings("user123", user_ctx.user_id);
}

test "JWT validation with expired token" {
    // Should return error.TokenExpired
}
```

### Integration Tests

```bash
# Test authentication flow
./scripts/test_authentication.sh

# Expected:
# ✓ Login redirect works
# ✓ Token exchange succeeds
# ✓ JWT validation passes
# ✓ User ID extracted correctly
# ✓ API calls with auth work
```

---

## Implementation Timeline

| Day | Task | Status |
|-----|------|--------|
| 11 | Documentation (this guide) | ✅ Complete |
| 12 | Frontend OAuth integration | ⏳ Planned |
| 12 | Backend JWT validation | ⏳ Planned |
| 13 | Session management | ⏳ Planned |
| 13 | Authorization/roles | ⏳ Planned |
| 14 | Testing & debugging | ⏳ Planned |
| 15 | Production deployment | ⏳ Planned |

---

## References

- [SAP IAS Documentation](https://help.sap.com/docs/IDENTITY_AUTHENTICATION)
- [OAuth 2.0 Specification](https://oauth.net/2/)
- [JWT.io](https://jwt.io/)
- [Zig JWT Library](https://github.com/daurnimator/zig-jwt)

---

**Last Updated:** January 21, 2026  
**Status:** 📋 Implementation Guide  
**Next:** Day 12 - Begin implementation
