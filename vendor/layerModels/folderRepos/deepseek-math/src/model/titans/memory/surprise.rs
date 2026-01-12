//! Hull-White Stochastic Surprise Gate (SSG) for memory consolidation.
//!
//! Implements the Titans surprise mechanism using Hull-White dynamics:
//! - Surprise is a Z-scored deviation from calibrated expectation
//! - Memory consolidation triggers on statistically significant events
//! - Full audit trail for governance and compliance
//!
//! Based on: "AI Narrative Stability" - Transplanting Stochastic Calculus
//! from Finance to Create Auditable, Long-Term Memory for AI.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Minimum standard deviation to prevent division by zero.
const MIN_STD_DEV: f32 = 1e-9;

/// Hull-White model parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HullWhiteParams {
    /// Mean reversion speed (α)
    pub alpha: f32,
    /// Long-run mean (θ)
    pub theta: f32,
    /// Volatility (σ)
    pub sigma: f32,
}

impl Default for HullWhiteParams {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            theta: 0.0,
            sigma: 1.0,
        }
    }
}

impl HullWhiteParams {
    pub fn new(alpha: f32, theta: f32, sigma: f32) -> Self {
        Self { alpha, theta, sigma }
    }

    /// Validate parameters for numerical stability.
    pub fn validate(&self) -> bool {
        self.alpha >= 0.0 && self.sigma > 0.0 &&
        self.alpha.is_finite() && self.theta.is_finite() && self.sigma.is_finite()
    }

    /// Compute conditional expectation: E[r_t | r_0]
    /// E[r_t] = r_0 * exp(-α*t) + θ * (1 - exp(-α*t))
    pub fn expectation(&self, r0: f32, dt: f32) -> f32 {
        if self.alpha.abs() < 1e-10 {
            // Degenerate case: Brownian motion
            r0
        } else {
            let decay = (-self.alpha * dt).exp();
            r0 * decay + self.theta * (1.0 - decay)
        }
    }

    /// Compute conditional variance: Var[r_t | r_0]
    /// Var[r_t] = (σ² / 2α) * (1 - exp(-2αt))
    pub fn variance(&self, dt: f32) -> f32 {
        if self.alpha.abs() < 1e-10 {
            // Brownian motion limit
            self.sigma * self.sigma * dt
        } else {
            let factor = 1.0 - (-2.0 * self.alpha * dt).exp();
            (self.sigma * self.sigma / (2.0 * self.alpha)) * factor
        }
    }

    /// Compute conditional standard deviation.
    pub fn std_dev(&self, dt: f32) -> f32 {
        self.variance(dt).sqrt().max(MIN_STD_DEV)
    }
}

/// Configuration for the Stochastic Surprise Gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseConfig {
    /// Z-score threshold for triggering memory consolidation
    pub z_threshold: f32,
    /// Hull-White parameters
    pub hull_white: HullWhiteParams,
    /// Window size for online calibration
    pub calibration_window: usize,
    /// Target consolidation rate (fraction of samples to store)
    pub target_consolidation_rate: f32,
    /// Enable adaptive calibration
    pub adaptive_calibration: bool,
    /// Learning rate for threshold adaptation
    pub threshold_lr: f32,
}

impl Default for SurpriseConfig {
    fn default() -> Self {
        Self {
            z_threshold: 2.0,  // 2 standard deviations
            hull_white: HullWhiteParams::default(),
            calibration_window: 1000,
            target_consolidation_rate: 0.2,
            adaptive_calibration: true,
            threshold_lr: 0.01,
        }
    }
}

/// Result of a Hull-White surprise computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HullWhiteSurprise {
    /// Raw Z-score
    pub z_score: f32,
    /// Gated surprise magnitude (max(0, Z - threshold))
    pub surprise: f32,
    /// Whether this triggered consolidation
    pub triggered: bool,
    /// P-value for statistical significance
    pub p_value: f32,
    /// Significance level description
    pub significance: SignificanceLevel,
}


impl HullWhiteSurprise {
    /// Create a zeroed surprise record.
    pub fn none() -> Self {
        Self {
            z_score: 0.0,
            surprise: 0.0,
            triggered: false,
            p_value: 1.0,
            significance: SignificanceLevel::Insignificant,
        }
    }
}

/// Significance level based on Z-score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignificanceLevel {
    Insignificant,  // Z < 1
    Marginal,       // 1 <= Z < 2
    Significant,    // 2 <= Z < 3
    HighlySignificant, // 3 <= Z < 4
    Extreme,        // Z >= 4
}

impl SignificanceLevel {
    pub fn from_z_score(z: f32) -> Self {
        let z_abs = z.abs();
        if z_abs < 1.0 {
            SignificanceLevel::Insignificant
        } else if z_abs < 2.0 {
            SignificanceLevel::Marginal
        } else if z_abs < 3.0 {
            SignificanceLevel::Significant
        } else if z_abs < 4.0 {
            SignificanceLevel::HighlySignificant
        } else {
            SignificanceLevel::Extreme
        }
    }
}

/// Audit trail for a memory decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseAuditTrail {
    /// Timestamp of the decision
    pub timestamp_ms: u64,
    /// Observed value
    pub observed: f32,
    /// Expected value from model
    pub expected: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Computed surprise
    pub surprise: HullWhiteSurprise,
    /// Model parameters at decision time
    pub params: HullWhiteParams,
    /// Time step used
    pub dt: f32,
    /// Domain/context identifier
    pub domain: String,
}

impl SurpriseAuditTrail {
    /// Convert to JSON for logging.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Stochastic Surprise Gate using Hull-White dynamics.
#[derive(Debug, Clone)]
pub struct SurpriseGate {
    config: SurpriseConfig,
    /// Running observations for calibration
    observations: VecDeque<f32>,
    /// Current calibrated parameters
    current_params: HullWhiteParams,
    /// Last observed value (for dt computation)
    last_value: f32,
    /// Total observations
    count: u64,
    /// Total consolidations
    consolidations: u64,
    /// Audit trail buffer
    audit_buffer: VecDeque<SurpriseAuditTrail>,
}

impl SurpriseGate {
    pub fn new(config: SurpriseConfig) -> Self {
        let current_params = config.hull_white.clone();
        Self {
            observations: VecDeque::with_capacity(config.calibration_window),
            current_params,
            last_value: 0.0,
            count: 0,
            consolidations: 0,
            audit_buffer: VecDeque::with_capacity(100),
            config,
        }
    }

    /// Compute surprise for a new observation using Hull-White dynamics.
    /// Returns Z-score based surprise with full audit trail.
    pub fn compute(&mut self, observed: f32, dt: f32) -> HullWhiteSurprise {
        // Compute expectation and standard deviation
        let expected = self.current_params.expectation(self.last_value, dt);
        let std_dev = self.current_params.std_dev(dt);

        // Compute Z-score
        let z_score = (observed - expected).abs() / std_dev;

        // Compute p-value using error function approximation
        let p_value = 2.0 * (1.0 - erf_approx(z_score / std::f32::consts::SQRT_2));

        // Gated surprise
        let surprise = (z_score - self.config.z_threshold).max(0.0);
        let triggered = z_score > self.config.z_threshold;
        let significance = SignificanceLevel::from_z_score(z_score);

        // Update state
        self.last_value = observed;
        self.count += 1;
        if triggered {
            self.consolidations += 1;
        }

        // Add to observation window for calibration
        if self.observations.len() >= self.config.calibration_window {
            self.observations.pop_front();
        }
        self.observations.push_back(observed);

        // Adaptive calibration
        if self.config.adaptive_calibration && self.count % 100 == 0 {
            self.calibrate();
        }

        HullWhiteSurprise {
            z_score,
            surprise,
            triggered,
            p_value,
            significance,
        }
    }


    /// Compute surprise and generate audit trail.
    pub fn compute_with_audit(&mut self, observed: f32, dt: f32, domain: &str) -> (HullWhiteSurprise, SurpriseAuditTrail) {
        let expected = self.current_params.expectation(self.last_value, dt);
        let std_dev = self.current_params.std_dev(dt);
        let surprise = self.compute(observed, dt);

        let trail = SurpriseAuditTrail {
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            observed,
            expected,
            std_dev,
            surprise: surprise.clone(),
            params: self.current_params.clone(),
            dt,
            domain: domain.to_string(),
        };

        // Store in audit buffer
        if self.audit_buffer.len() >= 100 {
            self.audit_buffer.pop_front();
        }
        self.audit_buffer.push_back(trail.clone());

        (surprise, trail)
    }

    /// Check if surprise should trigger consolidation.
    pub fn should_consolidate(&self, surprise: &HullWhiteSurprise) -> bool {
        surprise.triggered
    }

    /// Compute learning rate (theta) based on surprise.
    /// Higher surprise = stronger memory write.
    pub fn theta(&self, surprise: &HullWhiteSurprise) -> f32 {
        if surprise.triggered {
            // Scale from 0 to 1 based on how far above threshold
            let scale = (surprise.z_score - self.config.z_threshold) / self.config.z_threshold;
            scale.min(1.0).max(0.0)
        } else {
            0.0
        }
    }

    /// Compute forget gate (alpha) based on surprise.
    /// High surprise = reduced forgetting (preserve important memories).
    pub fn alpha(&self, surprise: &HullWhiteSurprise) -> f32 {
        let base_alpha = 0.9;  // Default forgetting rate
        if surprise.triggered {
            // Reduce forgetting for surprising events
            base_alpha * (1.0 - 0.5 * surprise.surprise.min(1.0))
        } else {
            base_alpha
        }
    }

    /// Calibrate Hull-White parameters from observed data (MLE).
    fn calibrate(&mut self) {
        if self.observations.len() < 10 {
            return;
        }

        let obs: Vec<f32> = self.observations.iter().copied().collect();
        let n = obs.len();

        // Estimate theta as sample mean
        let theta: f32 = obs.iter().sum::<f32>() / n as f32;

        // Estimate alpha via AR(1) regression
        // r_{t+1} - theta ≈ (1 - alpha*dt) * (r_t - theta)
        let dt = 1.0;  // Assume unit time steps
        let mut sum_xy = 0.0f32;
        let mut sum_xx = 0.0f32;
        for i in 0..n-1 {
            let x = obs[i] - theta;
            let y = obs[i+1] - theta;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let rho = if sum_xx > 1e-10 { sum_xy / sum_xx } else { 0.9 };
        let alpha = -rho.ln().max(0.01) / dt;

        // Estimate sigma from residuals
        let mut residual_var = 0.0f32;
        for i in 0..n-1 {
            let expected = obs[i] * (-alpha * dt).exp() + theta * (1.0 - (-alpha * dt).exp());
            let residual = obs[i+1] - expected;
            residual_var += residual * residual;
        }
        residual_var /= (n - 1) as f32;

        let theoretical_var = (1.0 - (-2.0 * alpha * dt).exp()) / (2.0 * alpha);
        let sigma = if theoretical_var > 1e-10 {
            (residual_var / theoretical_var).sqrt()
        } else {
            residual_var.sqrt()
        };

        // Update parameters with validation
        let new_params = HullWhiteParams::new(alpha.max(0.01), theta, sigma.max(0.01));
        if new_params.validate() {
            self.current_params = new_params;
        }

        // Adapt threshold based on consolidation rate
        self.adapt_threshold();
    }

    /// Adapt threshold to achieve target consolidation rate.
    fn adapt_threshold(&mut self) {
        if self.count < 100 {
            return;
        }

        let actual_rate = self.consolidations as f32 / self.count as f32;
        let rate_error = actual_rate - self.config.target_consolidation_rate;

        // Increase threshold if consolidating too much, decrease if too little
        self.config.z_threshold += rate_error * self.config.threshold_lr;
        self.config.z_threshold = self.config.z_threshold.clamp(0.5, 4.0);
    }

    /// Reset the gate state.
    pub fn reset(&mut self) {
        self.observations.clear();
        self.last_value = 0.0;
        self.count = 0;
        self.consolidations = 0;
        self.current_params = self.config.hull_white.clone();
    }

    /// Get current statistics.
    pub fn stats(&self) -> SurpriseStats {
        let consolidation_rate = if self.count > 0 {
            self.consolidations as f32 / self.count as f32
        } else {
            0.0
        };

        SurpriseStats {
            count: self.count,
            consolidations: self.consolidations,
            consolidation_rate,
            z_threshold: self.config.z_threshold,
            current_alpha: self.current_params.alpha,
            current_theta: self.current_params.theta,
            current_sigma: self.current_params.sigma,
        }
    }

    /// Get recent audit trails.
    pub fn get_audit_trails(&self) -> Vec<SurpriseAuditTrail> {
        self.audit_buffer.iter().cloned().collect()
    }

    /// Get current z-threshold.
    pub fn current_threshold(&self) -> f32 {
        self.config.z_threshold
    }

    /// Get current Hull-White parameters.
    pub fn current_params(&self) -> &HullWhiteParams {
        &self.current_params
    }
}

/// Statistics from the surprise gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseStats {
    pub count: u64,
    pub consolidations: u64,
    pub consolidation_rate: f32,
    pub z_threshold: f32,
    pub current_alpha: f32,
    pub current_theta: f32,
    pub current_sigma: f32,
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
/// Accuracy: max error ~1.5e-7
fn erf_approx(x: f32) -> f32 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Forget gate for memory decay modulation.
#[derive(Debug, Clone)]
pub struct ForgetGate {
    base_alpha: f32,
    min_alpha: f32,
    max_alpha: f32,
}

impl Default for ForgetGate {
    fn default() -> Self {
        Self {
            base_alpha: 0.9,
            min_alpha: 0.5,
            max_alpha: 0.99,
        }
    }
}

impl ForgetGate {
    pub fn new(base_alpha: f32) -> Self {
        Self {
            base_alpha,
            min_alpha: 0.5,
            max_alpha: 0.99,
        }
    }

    /// Compute alpha (memory retention) based on surprise.
    pub fn alpha(&self, surprise: &HullWhiteSurprise) -> f32 {
        if surprise.triggered {
            // High surprise = less forgetting
            let retention_boost = surprise.surprise.min(1.0) * 0.5;
            (self.base_alpha - retention_boost).clamp(self.min_alpha, self.max_alpha)
        } else {
            self.base_alpha
        }
    }
}

/// Compute surprise from prediction vs actual outcome (legacy compatibility).
pub fn compute_surprise(
    predicted_tools: &[String],
    actual_tools: &[String],
    predicted_confidence: f64,
    actual_confidence: f64,
) -> f32 {
    // Tool overlap surprise
    let tool_overlap = predicted_tools.iter()
        .filter(|t| actual_tools.contains(t))
        .count() as f32;
    let max_tools = predicted_tools.len().max(actual_tools.len()) as f32;
    let tool_surprise = if max_tools > 0.0 {
        1.0 - (tool_overlap / max_tools)
    } else {
        0.5
    };

    // Confidence surprise
    let conf_surprise = (predicted_confidence - actual_confidence).abs() as f32;

    // Combined surprise (weighted average)
    0.6 * tool_surprise + 0.4 * conf_surprise
}


#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Hull-White Parameters Tests ====================

    #[test]
    fn test_hull_white_params_default() {
        let params = HullWhiteParams::default();
        assert!(params.validate());
        assert_eq!(params.alpha, 0.1);
        assert_eq!(params.theta, 0.0);
        assert_eq!(params.sigma, 1.0);
    }

    #[test]
    fn test_hull_white_params_validation() {
        // Valid params
        assert!(HullWhiteParams::new(0.1, 0.0, 1.0).validate());
        assert!(HullWhiteParams::new(0.0, 0.5, 0.5).validate()); // alpha=0 is valid (Brownian)

        // Invalid params
        assert!(!HullWhiteParams::new(-0.1, 0.0, 1.0).validate()); // negative alpha
        assert!(!HullWhiteParams::new(0.1, 0.0, 0.0).validate());  // zero sigma
        assert!(!HullWhiteParams::new(0.1, 0.0, -1.0).validate()); // negative sigma
        assert!(!HullWhiteParams::new(f32::NAN, 0.0, 1.0).validate()); // NaN
        assert!(!HullWhiteParams::new(0.1, f32::INFINITY, 1.0).validate()); // Inf
    }

    #[test]
    fn test_hull_white_expectation() {
        let params = HullWhiteParams::new(0.5, 1.0, 0.2);

        // At t=0, expectation should equal r0
        let exp_t0 = params.expectation(0.5, 0.0);
        assert!((exp_t0 - 0.5).abs() < 1e-6);

        // As t -> infinity, expectation should approach theta
        let exp_tinf = params.expectation(0.5, 100.0);
        assert!((exp_tinf - 1.0).abs() < 1e-6);

        // Intermediate time
        let exp_t1 = params.expectation(0.0, 1.0);
        // E[r_1] = 0 * exp(-0.5) + 1.0 * (1 - exp(-0.5))
        let expected = 1.0 * (1.0 - (-0.5_f32).exp());
        assert!((exp_t1 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_hull_white_expectation_brownian() {
        // When alpha ≈ 0, should behave like Brownian motion
        let params = HullWhiteParams::new(1e-12, 1.0, 0.5);
        let exp = params.expectation(0.5, 1.0);
        assert!((exp - 0.5).abs() < 1e-6); // r0 unchanged
    }

    #[test]
    fn test_hull_white_variance() {
        let params = HullWhiteParams::new(0.5, 1.0, 0.2);

        // At t=0, variance should be 0
        let var_t0 = params.variance(0.0);
        assert!(var_t0.abs() < 1e-6);

        // Variance should increase with time
        let var_t1 = params.variance(1.0);
        let var_t2 = params.variance(2.0);
        assert!(var_t2 > var_t1);

        // As t -> infinity, variance should approach σ²/(2α)
        let var_tinf = params.variance(100.0);
        let expected_var = 0.2 * 0.2 / (2.0 * 0.5);
        assert!((var_tinf - expected_var).abs() < 1e-6);
    }

    #[test]
    fn test_hull_white_variance_brownian() {
        // When alpha ≈ 0, Var[r_t] = σ² * t
        let params = HullWhiteParams::new(1e-12, 1.0, 0.5);
        let var = params.variance(2.0);
        let expected = 0.5 * 0.5 * 2.0;
        assert!((var - expected).abs() < 1e-4);
    }

    // ==================== Error Function Tests ====================

    #[test]
    fn test_erf_approx() {
        // Known values
        assert!((erf_approx(0.0) - 0.0).abs() < 1e-6);
        assert!((erf_approx(1.0) - 0.8427).abs() < 1e-3);
        assert!((erf_approx(2.0) - 0.9953).abs() < 1e-3);
        assert!((erf_approx(-1.0) - (-0.8427)).abs() < 1e-3);

        // Symmetry
        assert!((erf_approx(0.5) + erf_approx(-0.5)).abs() < 1e-6);
    }

    // ==================== Significance Level Tests ====================

    #[test]
    fn test_significance_levels() {
        assert_eq!(SignificanceLevel::from_z_score(0.5), SignificanceLevel::Insignificant);
        assert_eq!(SignificanceLevel::from_z_score(1.5), SignificanceLevel::Marginal);
        assert_eq!(SignificanceLevel::from_z_score(2.5), SignificanceLevel::Significant);
        assert_eq!(SignificanceLevel::from_z_score(3.5), SignificanceLevel::HighlySignificant);
        assert_eq!(SignificanceLevel::from_z_score(5.0), SignificanceLevel::Extreme);

        // Negative Z-scores should use absolute value
        assert_eq!(SignificanceLevel::from_z_score(-2.5), SignificanceLevel::Significant);
    }

    // ==================== Surprise Gate Tests ====================

    #[test]
    fn test_surprise_gate_creation() {
        let config = SurpriseConfig::default();
        let gate = SurpriseGate::new(config);
        let stats = gate.stats();

        assert_eq!(stats.count, 0);
        assert_eq!(stats.consolidations, 0);
        assert_eq!(stats.consolidation_rate, 0.0);
    }

    #[test]
    fn test_surprise_gate_compute_normal() {
        let config = SurpriseConfig::default();
        let mut gate = SurpriseGate::new(config);

        // First observation establishes baseline
        let s1 = gate.compute(0.0, 1.0);
        assert!(s1.z_score >= 0.0);

        // Similar observation should have low surprise
        let s2 = gate.compute(0.01, 1.0);
        assert!(s2.z_score < 2.0); // Below default threshold
        assert!(!s2.triggered);
    }

    #[test]
    fn test_surprise_gate_compute_extreme() {
        let mut config = SurpriseConfig::default();
        config.hull_white = HullWhiteParams::new(0.1, 0.0, 0.1); // Low volatility
        let mut gate = SurpriseGate::new(config);

        // Establish baseline
        gate.compute(0.0, 1.0);
        gate.compute(0.0, 1.0);
        gate.compute(0.0, 1.0);

        // Extreme deviation should trigger
        let s = gate.compute(1.0, 1.0); // 10 sigma deviation!
        assert!(s.z_score > 2.0);
        assert!(s.triggered);
        assert!(matches!(s.significance, SignificanceLevel::Significant |
                                          SignificanceLevel::HighlySignificant |
                                          SignificanceLevel::Extreme));
    }

    #[test]
    fn test_surprise_gate_theta_alpha() {
        let config = SurpriseConfig::default();
        let gate = SurpriseGate::new(config);

        // Non-triggered surprise
        let low_surprise = HullWhiteSurprise {
            z_score: 1.0,
            surprise: 0.0,
            triggered: false,
            p_value: 0.32,
            significance: SignificanceLevel::Marginal,
        };
        assert_eq!(gate.theta(&low_surprise), 0.0);
        assert_eq!(gate.alpha(&low_surprise), 0.9); // base alpha

        // Triggered surprise
        let high_surprise = HullWhiteSurprise {
            z_score: 4.0,
            surprise: 2.0,
            triggered: true,
            p_value: 0.0001,
            significance: SignificanceLevel::HighlySignificant,
        };
        assert!(gate.theta(&high_surprise) > 0.0);
        assert!(gate.alpha(&high_surprise) < 0.9); // reduced forgetting
    }

    #[test]
    fn test_surprise_gate_should_consolidate() {
        let config = SurpriseConfig::default();
        let gate = SurpriseGate::new(config);

        let triggered = HullWhiteSurprise {
            z_score: 3.0,
            surprise: 1.0,
            triggered: true,
            p_value: 0.003,
            significance: SignificanceLevel::Significant,
        };
        assert!(gate.should_consolidate(&triggered));

        let not_triggered = HullWhiteSurprise {
            z_score: 1.5,
            surprise: 0.0,
            triggered: false,
            p_value: 0.13,
            significance: SignificanceLevel::Marginal,
        };
        assert!(!gate.should_consolidate(&not_triggered));
    }

    #[test]
    fn test_surprise_gate_stats_tracking() {
        let config = SurpriseConfig::default();
        let mut gate = SurpriseGate::new(config);

        // Generate some observations
        for i in 0..10 {
            gate.compute(i as f32 * 0.1, 1.0);
        }

        let stats = gate.stats();
        assert_eq!(stats.count, 10);
    }

    #[test]
    fn test_surprise_gate_reset() {
        let config = SurpriseConfig::default();
        let mut gate = SurpriseGate::new(config);

        // Generate observations
        for _ in 0..10 {
            gate.compute(0.5, 1.0);
        }
        assert_eq!(gate.stats().count, 10);

        // Reset
        gate.reset();
        let stats = gate.stats();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.consolidations, 0);
    }

    // ==================== Audit Trail Tests ====================

    #[test]
    fn test_audit_trail_generation() {
        let config = SurpriseConfig::default();
        let mut gate = SurpriseGate::new(config);

        let (surprise, trail) = gate.compute_with_audit(0.5, 1.0, "test_domain");

        assert_eq!(trail.domain, "test_domain");
        assert_eq!(trail.observed, 0.5);
        assert!(trail.timestamp_ms > 0);
        assert_eq!(trail.surprise.z_score, surprise.z_score);
    }

    #[test]
    fn test_audit_trail_json_serialization() {
        let trail = SurpriseAuditTrail {
            timestamp_ms: 1704326400000,
            observed: 0.5,
            expected: 0.3,
            std_dev: 0.1,
            surprise: HullWhiteSurprise {
                z_score: 2.0,
                surprise: 0.0,
                triggered: false,
                p_value: 0.046,
                significance: SignificanceLevel::Marginal,
            },
            params: HullWhiteParams::new(0.1, 0.0, 0.1),
            dt: 1.0,
            domain: "finance".to_string(),
        };

        let json = trail.to_json();
        assert!(json.contains("\"timestamp_ms\""));
        assert!(json.contains("\"z_score\""));
        assert!(json.contains("\"domain\""));
        assert!(json.contains("finance"));
    }

    #[test]
    fn test_audit_trail_buffer() {
        let config = SurpriseConfig::default();
        let mut gate = SurpriseGate::new(config);

        // Generate multiple observations with audit
        for i in 0..5 {
            gate.compute_with_audit(i as f32 * 0.1, 1.0, &format!("domain_{}", i));
        }

        let trails = gate.get_audit_trails();
        assert_eq!(trails.len(), 5);
        assert_eq!(trails[0].domain, "domain_0");
        assert_eq!(trails[4].domain, "domain_4");
    }

    // ==================== Forget Gate Tests ====================

    #[test]
    fn test_forget_gate_default() {
        let gate = ForgetGate::default();

        let low_surprise = HullWhiteSurprise::none();
        assert_eq!(gate.alpha(&low_surprise), 0.9);
    }

    #[test]
    fn test_forget_gate_high_surprise() {
        let gate = ForgetGate::new(0.9);

        let high_surprise = HullWhiteSurprise {
            z_score: 4.0,
            surprise: 1.0,
            triggered: true,
            p_value: 0.0001,
            significance: SignificanceLevel::Extreme,
        };

        // High surprise should reduce forgetting (lower alpha)
        let alpha = gate.alpha(&high_surprise);
        assert!(alpha < 0.9);
        assert!(alpha >= 0.5); // min_alpha
    }

    // ==================== HullWhiteSurprise Tests ====================

    #[test]
    fn test_hull_white_surprise_none() {
        let surprise = HullWhiteSurprise::none();
        assert_eq!(surprise.z_score, 0.0);
        assert_eq!(surprise.surprise, 0.0);
        assert!(!surprise.triggered);
        assert_eq!(surprise.p_value, 1.0);
        assert_eq!(surprise.significance, SignificanceLevel::Insignificant);
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_end_to_end_surprise_flow() {
        // Simulate a time series with a shock
        let mut config = SurpriseConfig::default();
        config.hull_white = HullWhiteParams::new(0.2, 0.0, 0.1);
        config.z_threshold = 2.0;
        let mut gate = SurpriseGate::new(config);

        // Normal observations (around 0)
        let normal_values = vec![0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.03, 0.01];
        for val in &normal_values {
            let s = gate.compute(*val, 1.0);
            // Most should not trigger (small deviations)
            if s.triggered {
                println!("Unexpectedly triggered at {}: z={}", val, s.z_score);
            }
        }

        // Inject a shock
        let shock = gate.compute(0.8, 1.0); // Large deviation
        assert!(shock.z_score > 2.0, "Shock z_score {} should exceed threshold", shock.z_score);

        // More normal observations
        for val in &normal_values {
            gate.compute(*val, 1.0);
        }

        let stats = gate.stats();
        println!("Final stats: {:?}", stats);
        assert!(stats.consolidations >= 1, "Should have at least 1 consolidation from shock");
    }

    #[test]
    fn test_calibration_stability() {
        let config = SurpriseConfig {
            adaptive_calibration: true,
            calibration_window: 100,
            ..SurpriseConfig::default()
        };
        let mut gate = SurpriseGate::new(config);

        // Generate observations that should trigger calibration
        for i in 0..200 {
            let val = (i as f32 * 0.1).sin() * 0.5;
            gate.compute(val, 1.0);
        }

        let stats = gate.stats();
        // After calibration, parameters should be reasonable
        assert!(stats.current_alpha > 0.0);
        assert!(stats.current_sigma > 0.0);
    }
}