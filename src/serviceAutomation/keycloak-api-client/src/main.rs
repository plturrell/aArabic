use anyhow::Result;
use clap::{Parser, Subcommand};
use keycloak_api_client::*;

#[derive(Parser)]
#[command(name = "keycloak-cli")]
#[command(about = "Keycloak Client\n\nAuthentication and authorization management")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = "http://localhost:8080", env = "KEYCLOAK_URL")]
    url: String,

    #[arg(short, long, env = "KEYCLOAK_ADMIN_TOKEN")]
    token: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Realm operations
    ListRealms,
    GetRealm { realm: String },
    DeleteRealm { realm: String },
    
    // Client operations
    ListClients { realm: String },
    GetClient { realm: String, id: String },
    DeleteClient { realm: String, id: String },
    
    // User operations
    ListUsers { realm: String },
    GetUser { realm: String, id: String },
    DeleteUser { realm: String, id: String },
    LogoutUser { realm: String, user_id: String },
    
    // Role operations
    ListRoles { realm: String },
    GetRole { realm: String, name: String },
    DeleteRole { realm: String, name: String },
    
    // Group operations
    ListGroups { realm: String },
    GetGroup { realm: String, id: String },
    DeleteGroup { realm: String, id: String },
    
    // Session operations
    GetSessions { realm: String, user_id: String },
    
    // OAuth2/OIDC operations
    GetToken {
        realm: String,
        username: String,
        password: String,
        #[arg(long, default_value = "admin-cli")]
        client_id: String,
        #[arg(long)]
        client_secret: Option<String>,
        #[arg(long, default_value = "openid")]
        scope: Option<String>,
    },
    RefreshToken {
        realm: String,
        refresh_token: String,
        #[arg(long, default_value = "admin-cli")]
        client_id: String,
        #[arg(long)]
        client_secret: Option<String>,
    },
    ValidateToken {
        realm: String,
        token: String,
        #[arg(long, default_value = "admin-cli")]
        client_id: String,
        #[arg(long)]
        client_secret: Option<String>,
    },
    GetUserInfo {
        realm: String,
        access_token: String,
    },
    LogoutViaToken {
        realm: String,
        refresh_token: String,
        #[arg(long, default_value = "admin-cli")]
        client_id: String,
        #[arg(long)]
        client_secret: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    let client = if let Some(token) = cli.token {
        KeycloakClient::with_token(cli.url, token)
    } else {
        KeycloakClient::new(cli.url)
    };

    match cli.command {
        Commands::ListRealms => {
            let realms = client.list_realms()?;
            println!("🏰 Realms ({}):", realms.len());
            for realm in realms {
                let status = if realm.enabled { "🟢" } else { "⚪" };
                println!("   {} {}", status, realm.realm);
            }
        }

        Commands::GetRealm { realm } => {
            let r = client.get_realm(&realm)?;
            println!("🏰 Realm: {}", r.realm);
            println!("   Enabled: {}", r.enabled);
            if let Some(name) = r.display_name {
                println!("   Display: {}", name);
            }
        }

        Commands::DeleteRealm { realm } => {
            client.delete_realm(&realm)?;
            println!("✅ Deleted realm: {}", realm);
        }

        Commands::ListClients { realm } => {
            let clients = client.list_clients(&realm)?;
            println!("🔧 Clients ({}):", clients.len());
            for c in clients {
                println!("   • {}", c.client_id);
            }
        }

        Commands::GetClient { realm, id } => {
            let c = client.get_client(&realm, &id)?;
            println!("🔧 Client: {}", c.client_id);
            println!("   Enabled: {}", c.enabled);
        }

        Commands::DeleteClient { realm, id } => {
            client.delete_client(&realm, &id)?;
            println!("✅ Deleted client: {}", id);
        }

        Commands::ListUsers { realm } => {
            let users = client.list_users(&realm)?;
            println!("👤 Users ({}):", users.len());
            for user in users {
                println!("   • {}", user.username);
            }
        }

        Commands::GetUser { realm, id } => {
            let user = client.get_user(&realm, &id)?;
            println!("👤 User: {}", user.username);
            if let Some(email) = user.email {
                println!("   Email: {}", email);
            }
        }

        Commands::DeleteUser { realm, id } => {
            client.delete_user(&realm, &id)?;
            println!("✅ Deleted user: {}", id);
        }

        Commands::LogoutUser { realm, user_id } => {
            client.logout_user(&realm, &user_id)?;
            println!("✅ Logged out user: {}", user_id);
        }

        Commands::ListRoles { realm } => {
            let roles = client.list_realm_roles(&realm)?;
            println!("🎭 Roles ({}):", roles.len());
            for role in roles {
                println!("   • {}", role.name);
            }
        }

        Commands::GetRole { realm, name } => {
            let role = client.get_realm_role(&realm, &name)?;
            println!("🎭 Role: {}", role.name);
            if let Some(desc) = role.description {
                println!("   Description: {}", desc);
            }
        }

        Commands::DeleteRole { realm, name } => {
            client.delete_realm_role(&realm, &name)?;
            println!("✅ Deleted role: {}", name);
        }

        Commands::ListGroups { realm } => {
            let groups = client.list_groups(&realm)?;
            println!("👥 Groups ({}):", groups.len());
            for group in groups {
                println!("   • {}", group.name);
            }
        }

        Commands::GetGroup { realm, id } => {
            let group = client.get_group(&realm, &id)?;
            println!("👥 Group: {}", group.name);
        }

        Commands::DeleteGroup { realm, id } => {
            client.delete_group(&realm, &id)?;
            println!("✅ Deleted group: {}", id);
        }

        Commands::GetSessions { realm, user_id } => {
            let sessions = client.get_user_sessions(&realm, &user_id)?;
            println!("🔐 Sessions ({}):", sessions.len());
        }

        Commands::GetToken { realm, username, password, client_id, client_secret, scope } => {
            let token = client.get_token(
                &realm,
                &username,
                &password,
                &client_id,
                client_secret.as_deref(),
                scope.as_deref(),
            )?;
            println!("🔑 Access Token Retrieved:");
            println!("   Token: {}...", &token.access_token[..50.min(token.access_token.len())]);
            println!("   Type: {}", token.token_type);
            println!("   Expires In: {}s", token.expires_in);
            if let Some(refresh) = &token.refresh_token {
                println!("   Refresh Token: {}...", &refresh[..30.min(refresh.len())]);
            }
            if let Some(scope_val) = &token.scope {
                println!("   Scope: {}", scope_val);
            }
        }

        Commands::RefreshToken { realm, refresh_token, client_id, client_secret } => {
            let token = client.refresh_token(
                &realm,
                &refresh_token,
                &client_id,
                client_secret.as_deref(),
            )?;
            println!("🔄 Token Refreshed:");
            println!("   New Access Token: {}...", &token.access_token[..50.min(token.access_token.len())]);
            println!("   Expires In: {}s", token.expires_in);
        }

        Commands::ValidateToken { realm, token, client_id, client_secret } => {
            let result = client.validate_token(
                &realm,
                &token,
                &client_id,
                client_secret.as_deref(),
            )?;
            println!("✅ Token Validation:");
            println!("   Active: {}", result.active);
            if let Some(username) = &result.username {
                println!("   Username: {}", username);
            }
            if let Some(email) = &result.email {
                println!("   Email: {}", email);
            }
            if let Some(exp) = result.exp {
                println!("   Expires At: {}", exp);
            }
        }

        Commands::GetUserInfo { realm, access_token } => {
            let info = client.get_user_info_from_token(&realm, &access_token)?;
            println!("👤 User Info:");
            println!("   Subject: {}", info.sub);
            if let Some(email) = &info.email {
                println!("   Email: {}", email);
            }
            if let Some(name) = &info.name {
                println!("   Name: {}", name);
            }
            if let Some(username) = &info.preferred_username {
                println!("   Username: {}", username);
            }
        }

        Commands::LogoutViaToken { realm, refresh_token, client_id, client_secret } => {
            client.logout_via_token(
                &realm,
                &refresh_token,
                &client_id,
                client_secret.as_deref(),
            )?;
            println!("✅ Logged out successfully");
        }
    }

    Ok(())
}
