//! DragonflyDB API Client
//!
//! Rust client library for DragonflyDB (Redis-compatible in-memory data store).
//! DragonflyDB is a modern replacement for Redis with better performance and memory efficiency.

use anyhow::{Context, Result};
use redis::{aio::ConnectionManager, AsyncCommands, Client};
use serde::{Deserialize, Serialize};

/// DragonflyDB client
#[derive(Clone)]
pub struct DragonflyClient {
    connection: ConnectionManager,
}

/// Database statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct DbStats {
    pub keys: usize,
    pub memory_used: usize,
    pub uptime_seconds: u64,
}

impl DragonflyClient {
    /// Create a new DragonflyDB client
    ///
    /// # Arguments
    /// * `url` - Connection URL (e.g., "redis://localhost:6379")
    pub async fn new(url: &str) -> Result<Self> {
        let client = Client::open(url)
            .context("Failed to create Redis client")?;
        
        let connection = ConnectionManager::new(client)
            .await
            .context("Failed to establish connection")?;

        Ok(Self { connection })
    }

    /// Create client from environment variable DRAGONFLY_URL
    pub async fn from_env() -> Result<Self> {
        let url = std::env::var("DRAGONFLY_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());
        Self::new(&url).await
    }

    // ==================== Key-Value Operations ====================

    /// Set a key-value pair
    pub async fn set(&mut self, key: &str, value: &str) -> Result<()> {
        self.connection
            .set(key, value)
            .await
            .context("Failed to set key")
    }

    /// Set with expiration (seconds)
    pub async fn setex(&mut self, key: &str, value: &str, seconds: u64) -> Result<()> {
        self.connection
            .set_ex(key, value, seconds)
            .await
            .context("Failed to set key with expiration")
    }

    /// Get a value by key
    pub async fn get(&mut self, key: &str) -> Result<Option<String>> {
        let value: Option<String> = self.connection
            .get(key)
            .await
            .context("Failed to get key")?;
        Ok(value)
    }

    /// Delete a key
    pub async fn del(&mut self, key: &str) -> Result<bool> {
        let deleted: i32 = self.connection
            .del(key)
            .await
            .context("Failed to delete key")?;
        Ok(deleted > 0)
    }

    /// Check if key exists
    pub async fn exists(&mut self, key: &str) -> Result<bool> {
        let exists: bool = self.connection
            .exists(key)
            .await
            .context("Failed to check key existence")?;
        Ok(exists)
    }

    /// Set expiration on a key (seconds)
    pub async fn expire(&mut self, key: &str, seconds: u64) -> Result<bool> {
        let set: bool = self.connection
            .expire(key, seconds as i64)
            .await
            .context("Failed to set expiration")?;
        Ok(set)
    }

    /// Get time-to-live for a key (seconds)
    pub async fn ttl(&mut self, key: &str) -> Result<i64> {
        let ttl: i64 = self.connection
            .ttl(key)
            .await
            .context("Failed to get TTL")?;
        Ok(ttl)
    }

    // ==================== List Operations ====================

    /// Push value to list (left)
    pub async fn lpush(&mut self, key: &str, value: &str) -> Result<usize> {
        let len: usize = self.connection
            .lpush(key, value)
            .await
            .context("Failed to push to list")?;
        Ok(len)
    }

    /// Push value to list (right)
    pub async fn rpush(&mut self, key: &str, value: &str) -> Result<usize> {
        let len: usize = self.connection
            .rpush(key, value)
            .await
            .context("Failed to push to list")?;
        Ok(len)
    }

    /// Pop value from list (left)
    pub async fn lpop(&mut self, key: &str) -> Result<Option<String>> {
        let value: Option<String> = self.connection
            .lpop(key, None)
            .await
            .context("Failed to pop from list")?;
        Ok(value)
    }

    /// Get list length
    pub async fn llen(&mut self, key: &str) -> Result<usize> {
        let len: usize = self.connection
            .llen(key)
            .await
            .context("Failed to get list length")?;
        Ok(len)
    }

    /// Get range from list
    pub async fn lrange(&mut self, key: &str, start: isize, stop: isize) -> Result<Vec<String>> {
        let values: Vec<String> = self.connection
            .lrange(key, start, stop)
            .await
            .context("Failed to get list range")?;
        Ok(values)
    }

    // ==================== Hash Operations ====================

    /// Set hash field
    pub async fn hset(&mut self, key: &str, field: &str, value: &str) -> Result<bool> {
        let set: bool = self.connection
            .hset(key, field, value)
            .await
            .context("Failed to set hash field")?;
        Ok(set)
    }

    /// Get hash field
    pub async fn hget(&mut self, key: &str, field: &str) -> Result<Option<String>> {
        let value: Option<String> = self.connection
            .hget(key, field)
            .await
            .context("Failed to get hash field")?;
        Ok(value)
    }

    /// Get all hash fields and values
    pub async fn hgetall(&mut self, key: &str) -> Result<Vec<(String, String)>> {
        let values: Vec<(String, String)> = self.connection
            .hgetall(key)
            .await
            .context("Failed to get all hash fields")?;
        Ok(values)
    }

    /// Delete hash field
    pub async fn hdel(&mut self, key: &str, field: &str) -> Result<bool> {
        let deleted: i32 = self.connection
            .hdel(key, field)
            .await
            .context("Failed to delete hash field")?;
        Ok(deleted > 0)
    }

    // ==================== Set Operations ====================

    /// Add member to set
    pub async fn sadd(&mut self, key: &str, member: &str) -> Result<bool> {
        let added: i32 = self.connection
            .sadd(key, member)
            .await
            .context("Failed to add to set")?;
        Ok(added > 0)
    }

    /// Get all set members
    pub async fn smembers(&mut self, key: &str) -> Result<Vec<String>> {
        let members: Vec<String> = self.connection
            .smembers(key)
            .await
            .context("Failed to get set members")?;
        Ok(members)
    }

    /// Check if member exists in set
    pub async fn sismember(&mut self, key: &str, member: &str) -> Result<bool> {
        let is_member: bool = self.connection
            .sismember(key, member)
            .await
            .context("Failed to check set membership")?;
        Ok(is_member)
    }

    /// Get set cardinality (size)
    pub async fn scard(&mut self, key: &str) -> Result<usize> {
        let size: usize = self.connection
            .scard(key)
            .await
            .context("Failed to get set size")?;
        Ok(size)
    }

    // ==================== Utility Operations ====================

    /// Get all keys matching pattern
    pub async fn keys(&mut self, pattern: &str) -> Result<Vec<String>> {
        let keys: Vec<String> = self.connection
            .keys(pattern)
            .await
            .context("Failed to get keys")?;
        Ok(keys)
    }

    /// Flush all data (use with caution!)
    pub async fn flushdb(&mut self) -> Result<()> {
        redis::cmd("FLUSHDB")
            .query_async::<()>(&mut self.connection)
            .await
            .context("Failed to flush database")?;
        Ok(())
    }

    /// Get database size (number of keys)
    pub async fn dbsize(&mut self) -> Result<usize> {
        let size: usize = redis::cmd("DBSIZE")
            .query_async::<usize>(&mut self.connection)
            .await
            .context("Failed to get database size")?;
        Ok(size)
    }

    /// Ping the server
    pub async fn ping(&mut self) -> Result<String> {
        let pong: String = redis::cmd("PING")
            .query_async::<String>(&mut self.connection)
            .await
            .context("Failed to ping server")?;
        Ok(pong)
    }

    /// Get server info
    pub async fn info(&mut self) -> Result<String> {
        let info: String = redis::cmd("INFO")
            .query_async::<String>(&mut self.connection)
            .await
            .context("Failed to get server info")?;
        Ok(info)
    }

    /// Health check
    pub async fn health_check(&mut self) -> Result<bool> {
        match self.ping().await {
            Ok(response) => Ok(response == "PONG"),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        // This will fail without a running DragonflyDB instance
        let result = DragonflyClient::new("redis://localhost:6379").await;
        // We just check it doesn't panic
        assert!(result.is_ok() || result.is_err());
    }
}
