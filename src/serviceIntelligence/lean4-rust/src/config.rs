/// Configuration module for the Lean4 Parser service
/// Supports environment variable configuration with sensible defaults

use once_cell::sync::Lazy;
use std::env;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Host to bind to
    pub host: String,
    /// Port to bind to
    pub port: u16,
    /// Maximum request body size in bytes
    pub max_body_size: usize,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
}

/// CORS configuration
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Allowed origins (comma-separated, or "*" for any)
    pub allowed_origins: Vec<String>,
    /// Whether to allow any origin
    pub allow_any_origin: bool,
    /// Max age for CORS preflight cache in seconds
    pub max_age_secs: u32,
}

/// File system configuration
#[derive(Debug, Clone)]
pub struct FileSystemConfig {
    /// Default output directory for generated files
    pub output_dir: String,
    /// Allowed base paths for file operations (security)
    pub allowed_paths: Vec<String>,
    /// Maximum input file size in bytes
    pub max_input_size: usize,
}

/// Logging configuration
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
}

/// Complete application configuration
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub cors: CorsConfig,
    pub filesystem: FileSystemConfig,
    pub log: LogConfig,
}

impl AppConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        Self {
            server: ServerConfig {
                host: env::var("LEAN4_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: env::var("LEAN4_PORT")
                    .ok()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(8002),
                max_body_size: env::var("LEAN4_MAX_BODY_SIZE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10 * 1024 * 1024), // 10MB default
                request_timeout_secs: env::var("LEAN4_REQUEST_TIMEOUT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(30),
            },
            cors: CorsConfig {
                allowed_origins: env::var("LEAN4_CORS_ORIGINS")
                    .map(|s| s.split(',').map(|o| o.trim().to_string()).collect())
                    .unwrap_or_else(|_| vec![]),
                allow_any_origin: env::var("LEAN4_CORS_ALLOW_ANY")
                    .map(|v| v.to_lowercase() == "true" || v == "1")
                    .unwrap_or(false), // Default to restrictive
                max_age_secs: env::var("LEAN4_CORS_MAX_AGE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3600),
            },
            filesystem: FileSystemConfig {
                output_dir: env::var("LEAN4_OUTPUT_DIR")
                    .unwrap_or_else(|_| "/tmp/lean4_generated".to_string()),
                allowed_paths: env::var("LEAN4_ALLOWED_PATHS")
                    .map(|s| s.split(',').map(|p| p.trim().to_string()).collect())
                    .unwrap_or_else(|_| vec!["/tmp".to_string()]),
                max_input_size: env::var("LEAN4_MAX_INPUT_SIZE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5 * 1024 * 1024), // 5MB default
            },
            log: LogConfig {
                level: env::var("LEAN4_LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            },
        }
    }

    /// Get bind address string
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.server.host, self.server.port)
    }
}

/// Global configuration instance
pub static CONFIG: Lazy<AppConfig> = Lazy::new(AppConfig::from_env);

/// Validate that a path is within allowed directories
pub fn is_path_allowed(path: &str) -> bool {
    use std::path::Path;

    let path = match Path::new(path).canonicalize() {
        Ok(p) => p,
        Err(_) => {
            // If path doesn't exist yet, check parent
            let parent = Path::new(path).parent();
            match parent.and_then(|p| p.canonicalize().ok()) {
                Some(p) => p,
                None => return false,
            }
        }
    };

    let path_str = path.to_string_lossy();

    CONFIG.filesystem.allowed_paths.iter().any(|allowed| {
        let allowed_path = match Path::new(allowed).canonicalize() {
            Ok(p) => p,
            Err(_) => return false,
        };
        path_str.starts_with(&allowed_path.to_string_lossy().as_ref())
    })
}

/// Sanitize a path to prevent traversal attacks
pub fn sanitize_path(base_dir: &str, filename: &str) -> Option<String> {
    use std::path::Path;

    // Remove any path traversal attempts
    let sanitized: String = filename
        .replace("..", "")
        .replace(['/', '\\'], "_")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
        .collect();

    if sanitized.is_empty() {
        return None;
    }

    let full_path = Path::new(base_dir).join(&sanitized);
    let full_path_str = full_path.to_string_lossy().to_string();

    // Verify it's within allowed paths
    if is_path_allowed(&full_path_str) {
        Some(full_path_str)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        // Note: This test may fail if env vars are set
        let config = AppConfig::from_env();
        assert_eq!(config.server.port, 8002);
        assert!(!config.cors.allow_any_origin);
    }

    #[test]
    fn test_bind_address() {
        let config = AppConfig {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 3000,
                max_body_size: 1024,
                request_timeout_secs: 30,
            },
            cors: CorsConfig {
                allowed_origins: vec![],
                allow_any_origin: false,
                max_age_secs: 3600,
            },
            filesystem: FileSystemConfig {
                output_dir: "/tmp".to_string(),
                allowed_paths: vec!["/tmp".to_string()],
                max_input_size: 1024,
            },
            log: LogConfig {
                level: "info".to_string(),
            },
        };

        assert_eq!(config.bind_address(), "127.0.0.1:3000");
    }

    #[test]
    fn test_sanitize_path() {
        // Path traversal attempts are sanitized - ".." and "/" are removed
        // "../etc/passwd" becomes "etcpasswd" (safe filename, no traversal)
        let result = sanitize_path("/tmp", "../etc/passwd");
        // The result should either be None (blocked) or a safe path under /tmp
        if let Some(path) = result {
            // Should be under /tmp, not /etc
            assert!(path.starts_with("/tmp"));
            assert!(!path.contains("/etc/"));
        }

        // Normal filenames should work
        let result = sanitize_path("/tmp", "test_file.txt");
        assert!(result.is_some());
        if let Some(path) = &result {
            assert!(path.ends_with("test_file.txt"));
        }
    }
}
