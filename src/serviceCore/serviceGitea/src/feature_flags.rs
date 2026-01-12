/// Feature Flags Module
/// Manages feature rollout, A/B testing, and gradual deployment

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlag {
    pub name: String,
    pub enabled: bool,
    pub rollout_percentage: f32,
    pub target_users: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

pub struct FeatureFlagManager {
    flags: HashMap<String, FeatureFlag>,
}

impl FeatureFlagManager {
    pub fn new() -> Self {
        Self {
            flags: HashMap::new(),
        }
    }

    pub fn set_flag(
        &mut self,
        name: &str,
        enabled: bool,
        rollout_percentage: Option<f32>,
        target_users: Option<Vec<String>>,
    ) {
        let now = chrono::Utc::now().to_rfc3339();
        
        let flag = self.flags.entry(name.to_string()).or_insert_with(|| FeatureFlag {
            name: name.to_string(),
            enabled,
            rollout_percentage: 100.0,
            target_users: Vec::new(),
            created_at: now.clone(),
            updated_at: now.clone(),
        });

        flag.enabled = enabled;
        flag.rollout_percentage = rollout_percentage.unwrap_or(100.0);
        flag.target_users = target_users.unwrap_or_default();
        flag.updated_at = now;
    }

    pub fn get_flag(&self, name: &str) -> Option<&FeatureFlag> {
        self.flags.get(name)
    }

    pub fn is_enabled(&self, name: &str, user_id: &str) -> bool {
        match self.flags.get(name) {
            Some(flag) => {
                if !flag.enabled {
                    return false;
                }

                // Check if user is explicitly targeted
                if !flag.target_users.is_empty() {
                    return flag.target_users.contains(&user_id.to_string());
                }

                // Check rollout percentage
                self.should_enable_for_user(user_id, flag.rollout_percentage)
            }
            None => false,
        }
    }

    fn should_enable_for_user(&self, user_id: &str, percentage: f32) -> bool {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        user_id.hash(&mut hasher);
        let hash = hasher.finish();

        ((hash % 100) as f32) < percentage
    }
}

impl Default for FeatureFlagManager {
    fn default() -> Self {
        Self::new()
    }
}