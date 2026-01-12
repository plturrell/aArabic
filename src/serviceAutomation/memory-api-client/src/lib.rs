use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

pub struct MemoryClient {
    store: Arc<RwLock<HashMap<String, (String, Option<SystemTime>)>>>,
}

impl MemoryClient {
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn get(&self, key: &str) -> Result<Option<String>> {
        let store = self.store.read().unwrap();
        if let Some((value, expiry)) = store.get(key) {
            if let Some(exp) = expiry {
                if SystemTime::now() > *exp {
                    return Ok(None);
                }
            }
            Ok(Some(value.clone()))
        } else {
            Ok(None)
        }
    }

    pub fn set(&self, key: &str, value: &str) -> Result<()> {
        let mut store = self.store.write().unwrap();
        store.insert(key.to_string(), (value.to_string(), None));
        Ok(())
    }

    pub fn set_with_ttl(&self, key: &str, value: &str, ttl_secs: u64) -> Result<()> {
        let mut store = self.store.write().unwrap();
        let expiry = SystemTime::now() + Duration::from_secs(ttl_secs);
        store.insert(key.to_string(), (value.to_string(), Some(expiry)));
        Ok(())
    }

    pub fn delete(&self, key: &str) -> Result<bool> {
        let mut store = self.store.write().unwrap();
        Ok(store.remove(key).is_some())
    }

    pub fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.get(key)?.is_some())
    }

    pub fn keys(&self) -> Result<Vec<String>> {
        let store = self.store.read().unwrap();
        Ok(store.keys().cloned().collect())
    }

    pub fn clear(&self) -> Result<()> {
        let mut store = self.store.write().unwrap();
        store.clear();
        Ok(())
    }

    pub fn len(&self) -> Result<usize> {
        let store = self.store.read().unwrap();
        Ok(store.len())
    }
}

impl Default for MemoryClient {
    fn default() -> Self {
        Self::new()
    }
}
