# mhc_metrics.mojo
# Day 45: mHC (meta-Homeostatic Control) Integration for TAU2-Bench
# Provides stability metrics and comparison for benchmark evaluation

from collections import Dict, List


struct MHCStabilityMetrics:
    """mHC stability metrics for TAU2-Bench evaluation."""
    var amplification_factor: Float32
    var norm_before: Float32
    var norm_after: Float32
    var max_activation: Float32
    var is_stable: Bool
    var sinkhorn_iterations: Int
    
    fn __init__(out self):
        self.amplification_factor = 1.0
        self.norm_before = 0.0
        self.norm_after = 0.0
        self.max_activation = 0.0
        self.is_stable = True
        self.sinkhorn_iterations = 0
    
    fn calculate_stability(self) -> Bool:
        """Stability if amplification in [0.9, 1.1] range."""
        return self.amplification_factor >= 0.9 and self.amplification_factor <= 1.1


struct MHCBenchmarkScorer:
    """Integrates mHC stability into TAU2-Bench scoring."""
    var stability_weight: Float32
    var stability_samples: List[MHCStabilityMetrics]
    var mhc_enabled: Bool
    
    fn __init__(out self, mhc_enabled: Bool = True, stability_weight: Float32 = 0.15):
        self.mhc_enabled = mhc_enabled
        self.stability_weight = stability_weight
        self.stability_samples = List[MHCStabilityMetrics]()
    
    fn record_stability(mut self, metrics: MHCStabilityMetrics):
        """Record stability measurement from an evaluation step."""
        self.stability_samples.append(metrics)
    
    fn get_stability_score(self) -> Float32:
        """Compute overall stability score from samples."""
        if len(self.stability_samples) == 0:
            return 1.0  # Assume stable if no measurements
        var stable_count = 0
        for i in range(len(self.stability_samples)):
            if self.stability_samples[i].is_stable:
                stable_count += 1
        return Float32(stable_count) / Float32(len(self.stability_samples))
    
    fn adjust_score(self, base_score: Float32) -> Float32:
        """Adjust benchmark score to include stability component."""
        if not self.mhc_enabled:
            return base_score
        let stability = self.get_stability_score()
        let adjusted = base_score * (1.0 - self.stability_weight) + stability * self.stability_weight
        return adjusted


struct MHCComparisonResult:
    """Results comparing benchmark runs with/without mHC."""
    var score_without_mhc: Float32
    var score_with_mhc: Float32
    var stability_score: Float32
    var improvement_pct: Float32
    
    fn __init__(out self, without_mhc: Float32, with_mhc: Float32, stability: Float32):
        self.score_without_mhc = without_mhc
        self.score_with_mhc = with_mhc
        self.stability_score = stability
        if without_mhc > 0:
            self.improvement_pct = (with_mhc - without_mhc) / without_mhc * 100.0
        else:
            self.improvement_pct = 0.0


fn compare_with_without_mhc(
    base_score: Float32,
    scorer_with_mhc: MHCBenchmarkScorer
) -> MHCComparisonResult:
    """
    Compare benchmark results with and without mHC stability metrics.
    
    Args:
        base_score: Raw benchmark score without mHC adjustment
        scorer_with_mhc: Scorer with stability measurements
        
    Returns:
        MHCComparisonResult with comparison data
    """
    let stability = scorer_with_mhc.get_stability_score()
    let adjusted = scorer_with_mhc.adjust_score(base_score)
    return MHCComparisonResult(base_score, adjusted, stability)


fn add_mhc_to_test_results(
    results: Dict[String, Float32],
    scorer: MHCBenchmarkScorer
) -> Dict[String, Float32]:
    """
    Add mHC stability measurements to test result dictionary.
    
    Args:
        results: Existing test results
        scorer: MHC benchmark scorer with stability data
        
    Returns:
        Results dictionary with mHC metrics added
    """
    var updated = results
    updated["mhc_stability_score"] = scorer.get_stability_score()
    updated["mhc_sample_count"] = Float32(len(scorer.stability_samples))
    updated["mhc_enabled"] = 1.0 if scorer.mhc_enabled else 0.0
    return updated

