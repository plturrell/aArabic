use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Complete Keycloak Authentication & Authorization API Client
pub struct KeycloakClient {
    base_url: String,
    admin_token: Option<String>,
    client: Client,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Realm {
    pub id: Option<String>,
    pub realm: String,
    pub enabled: bool,
    #[serde(rename = "displayName")]
    pub display_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientRep {
    pub id: Option<String>,
    #[serde(rename = "clientId")]
    pub client_id: String,
    pub enabled: bool,
    pub protocol: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Option<String>,
    pub username: String,
    pub email: Option<String>,
    pub enabled: bool,
    #[serde(rename = "firstName")]
    pub first_name: Option<String>,
    #[serde(rename = "lastName")]
    pub last_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: Option<String>,
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Group {
    pub id: Option<String>,
    pub name: String,
    pub path: Option<String>,
}

impl KeycloakClient {
    /// Create new Keycloak client
    pub fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            admin_token: None,
            client: Client::new(),
        }
    }

    /// Create client with admin token
    pub fn with_token(base_url: String, token: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            admin_token: Some(token),
            client: Client::new(),
        }
    }

    /// Helper to build request
    fn request(&self, method: &str, endpoint: &str) -> reqwest::blocking::RequestBuilder {
        let url = format!("{}/{}", self.base_url, endpoint.trim_start_matches('/'));
        let mut req = match method {
            "GET" => self.client.get(&url),
            "POST" => self.client.post(&url),
            "PUT" => self.client.put(&url),
            "DELETE" => self.client.delete(&url),
            _ => self.client.get(&url),
        };
        
        if let Some(token) = &self.admin_token {
            req = req.bearer_auth(token);
        }
        
        req
    }

    // ========================================================================
    // REALM OPERATIONS
    // ========================================================================

    /// List realms
    pub fn list_realms(&self) -> Result<Vec<Realm>> {
        let response = self.request("GET", "admin/realms").send()?;
        Ok(response.json()?)
    }

    /// Get realm
    pub fn get_realm(&self, realm: &str) -> Result<Realm> {
        let response = self.request("GET", &format!("admin/realms/{}", realm)).send()?;
        Ok(response.json()?)
    }

    /// Create realm
    pub fn create_realm(&self, realm: &Realm) -> Result<()> {
        self.request("POST", "admin/realms")
            .json(realm)
            .send()?;
        Ok(())
    }

    /// Update realm
    pub fn update_realm(&self, realm_name: &str, realm: &Realm) -> Result<()> {
        self.request("PUT", &format!("admin/realms/{}", realm_name))
            .json(realm)
            .send()?;
        Ok(())
    }

    /// Delete realm
    pub fn delete_realm(&self, realm: &str) -> Result<()> {
        self.request("DELETE", &format!("admin/realms/{}", realm)).send()?;
        Ok(())
    }

    // ========================================================================
    // CLIENT OPERATIONS
    // ========================================================================

    /// List clients
    pub fn list_clients(&self, realm: &str) -> Result<Vec<ClientRep>> {
        let response = self.request("GET", &format!("admin/realms/{}/clients", realm)).send()?;
        Ok(response.json()?)
    }

    /// Get client
    pub fn get_client(&self, realm: &str, id: &str) -> Result<ClientRep> {
        let response = self.request("GET", &format!("admin/realms/{}/clients/{}", realm, id)).send()?;
        Ok(response.json()?)
    }

    /// Create client
    pub fn create_client(&self, realm: &str, client: &ClientRep) -> Result<()> {
        self.request("POST", &format!("admin/realms/{}/clients", realm))
            .json(client)
            .send()?;
        Ok(())
    }

    /// Delete client
    pub fn delete_client(&self, realm: &str, id: &str) -> Result<()> {
        self.request("DELETE", &format!("admin/realms/{}/clients/{}", realm, id)).send()?;
        Ok(())
    }

    // ========================================================================
    // USER OPERATIONS
    // ========================================================================

    /// List users
    pub fn list_users(&self, realm: &str) -> Result<Vec<User>> {
        let response = self.request("GET", &format!("admin/realms/{}/users", realm)).send()?;
        Ok(response.json()?)
    }

    /// Get user
    pub fn get_user(&self, realm: &str, id: &str) -> Result<User> {
        let response = self.request("GET", &format!("admin/realms/{}/users/{}", realm, id)).send()?;
        Ok(response.json()?)
    }

    /// Create user
    pub fn create_user(&self, realm: &str, user: &User) -> Result<()> {
        self.request("POST", &format!("admin/realms/{}/users", realm))
            .json(user)
            .send()?;
        Ok(())
    }

    /// Update user
    pub fn update_user(&self, realm: &str, id: &str, user: &User) -> Result<()> {
        self.request("PUT", &format!("admin/realms/{}/users/{}", realm, id))
            .json(user)
            .send()?;
        Ok(())
    }

    /// Delete user
    pub fn delete_user(&self, realm: &str, id: &str) -> Result<()> {
        self.request("DELETE", &format!("admin/realms/{}/users/{}", realm, id)).send()?;
        Ok(())
    }

    // ========================================================================
    // ROLE OPERATIONS
    // ========================================================================

    /// List realm roles
    pub fn list_realm_roles(&self, realm: &str) -> Result<Vec<Role>> {
        let response = self.request("GET", &format!("admin/realms/{}/roles", realm)).send()?;
        Ok(response.json()?)
    }

    /// Get realm role
    pub fn get_realm_role(&self, realm: &str, name: &str) -> Result<Role> {
        let response = self.request("GET", &format!("admin/realms/{}/roles/{}", realm, name)).send()?;
        Ok(response.json()?)
    }

    /// Create realm role
    pub fn create_realm_role(&self, realm: &str, role: &Role) -> Result<()> {
        self.request("POST", &format!("admin/realms/{}/roles", realm))
            .json(role)
            .send()?;
        Ok(())
    }

    /// Delete realm role
    pub fn delete_realm_role(&self, realm: &str, name: &str) -> Result<()> {
        self.request("DELETE", &format!("admin/realms/{}/roles/{}", realm, name)).send()?;
        Ok(())
    }

    // ========================================================================
    // GROUP OPERATIONS
    // ========================================================================

    /// List groups
    pub fn list_groups(&self, realm: &str) -> Result<Vec<Group>> {
        let response = self.request("GET", &format!("admin/realms/{}/groups", realm)).send()?;
        Ok(response.json()?)
    }

    /// Get group
    pub fn get_group(&self, realm: &str, id: &str) -> Result<Group> {
        let response = self.request("GET", &format!("admin/realms/{}/groups/{}", realm, id)).send()?;
        Ok(response.json()?)
    }

    /// Create group
    pub fn create_group(&self, realm: &str, group: &Group) -> Result<()> {
        self.request("POST", &format!("admin/realms/{}/groups", realm))
            .json(group)
            .send()?;
        Ok(())
    }

    /// Delete group
    pub fn delete_group(&self, realm: &str, id: &str) -> Result<()> {
        self.request("DELETE", &format!("admin/realms/{}/groups/{}", realm, id)).send()?;
        Ok(())
    }

    // ========================================================================
    // SESSION OPERATIONS
    // ========================================================================

    /// Get user sessions
    pub fn get_user_sessions(&self, realm: &str, user_id: &str) -> Result<Vec<Value>> {
        let response = self.request("GET", &format!("admin/realms/{}/users/{}/sessions", realm, user_id))
            .send()?;
        Ok(response.json()?)
    }

    /// Logout user
    pub fn logout_user(&self, realm: &str, user_id: &str) -> Result<()> {
        self.request("POST", &format!("admin/realms/{}/users/{}/logout", realm, user_id))
            .send()?;
        Ok(())
    }

    // ========================================================================
    // OAUTH2 / OIDC TOKEN OPERATIONS (Pure Rust Implementation)
    // ========================================================================

    /// Get access token using password grant (OAuth2)
    pub fn get_token(
        &self,
        realm: &str,
        username: &str,
        password: &str,
        client_id: &str,
        client_secret: Option<&str>,
        scope: Option<&str>,
    ) -> Result<TokenResponse> {
        let token_url = format!("{}/realms/{}/protocol/openid-connect/token", self.base_url, realm);
        
        let mut form_data = HashMap::new();
        form_data.insert("grant_type", "password");
        form_data.insert("username", username);
        form_data.insert("password", password);
        form_data.insert("client_id", client_id);
        
        if let Some(secret) = client_secret {
            form_data.insert("client_secret", secret);
        }
        if let Some(s) = scope {
            form_data.insert("scope", s);
        } else {
            form_data.insert("scope", "openid");
        }
        
        let response = self.client
            .post(&token_url)
            .form(&form_data)
            .send()?;
        
        Ok(response.json()?)
    }

    /// Refresh access token (OAuth2)
    pub fn refresh_token(
        &self,
        realm: &str,
        refresh_token: &str,
        client_id: &str,
        client_secret: Option<&str>,
    ) -> Result<TokenResponse> {
        let token_url = format!("{}/realms/{}/protocol/openid-connect/token", self.base_url, realm);
        
        let mut form_data = HashMap::new();
        form_data.insert("grant_type", "refresh_token");
        form_data.insert("refresh_token", refresh_token);
        form_data.insert("client_id", client_id);
        
        if let Some(secret) = client_secret {
            form_data.insert("client_secret", secret);
        }
        
        let response = self.client
            .post(&token_url)
            .form(&form_data)
            .send()?;
        
        Ok(response.json()?)
    }

    /// Validate/introspect token (OAuth2)
    pub fn validate_token(
        &self,
        realm: &str,
        token: &str,
        client_id: &str,
        client_secret: Option<&str>,
    ) -> Result<TokenIntrospection> {
        let introspect_url = format!("{}/realms/{}/protocol/openid-connect/token/introspect", self.base_url, realm);
        
        let mut form_data = HashMap::new();
        form_data.insert("token", token);
        form_data.insert("client_id", client_id);
        
        if let Some(secret) = client_secret {
            form_data.insert("client_secret", secret);
        }
        
        let response = self.client
            .post(&introspect_url)
            .form(&form_data)
            .send()?;
        
        Ok(response.json()?)
    }

    /// Get user info from access token (OIDC)
    pub fn get_user_info_from_token(&self, realm: &str, access_token: &str) -> Result<UserInfo> {
        let userinfo_url = format!("{}/realms/{}/protocol/openid-connect/userinfo", self.base_url, realm);
        
        let response = self.client
            .get(&userinfo_url)
            .bearer_auth(access_token)
            .send()?;
        
        Ok(response.json()?)
    }

    /// Logout via refresh token (OIDC)
    pub fn logout_via_token(
        &self,
        realm: &str,
        refresh_token: &str,
        client_id: &str,
        client_secret: Option<&str>,
    ) -> Result<()> {
        let logout_url = format!("{}/realms/{}/protocol/openid-connect/logout", self.base_url, realm);
        
        let mut form_data = HashMap::new();
        form_data.insert("refresh_token", refresh_token);
        form_data.insert("client_id", client_id);
        
        if let Some(secret) = client_secret {
            form_data.insert("client_secret", secret);
        }
        
        self.client
            .post(&logout_url)
            .form(&form_data)
            .send()?;
        
        Ok(())
    }
}

// ============================================================================
// OAUTH2 / OIDC DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub token_type: String,
    pub expires_in: u64,
    pub refresh_expires_in: Option<u64>,
    pub scope: Option<String>,
    pub session_state: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenIntrospection {
    pub active: bool,
    pub username: Option<String>,
    pub email: Option<String>,
    pub exp: Option<u64>,
    pub iat: Option<u64>,
    pub sub: Option<String>,
    pub typ: Option<String>,
    pub azp: Option<String>,
    pub session_state: Option<String>,
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub sub: String,
    pub email: Option<String>,
    pub email_verified: Option<bool>,
    pub name: Option<String>,
    pub preferred_username: Option<String>,
    pub given_name: Option<String>,
    pub family_name: Option<String>,
}
