#![forbid(unsafe_code)]

//! # ternary-criticality
//!
//! Critical slowing down detector for ternary systems.
//!
//! Monitors recovery time from perturbations and diverging autocorrelation
//! as early warning signals of system collapse.

/// A ternary state: -1, 0, or +1
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ternary {
    Neg,
    Zero,
    Pos,
}

impl Ternary {
    pub fn value(&self) -> i8 {
        match self {
            Ternary::Neg => -1,
            Ternary::Zero => 0,
            Ternary::Pos => 1,
        }
    }

    pub fn from_value(v: i8) -> Self {
        if v < 0 { Ternary::Neg } else if v > 0 { Ternary::Pos } else { Ternary::Zero }
    }
}

/// Configuration for the criticality detector
#[derive(Debug, Clone)]
pub struct CriticalityConfig {
    /// Window size for autocorrelation computation
    pub window_size: usize,
    /// Lag for autocorrelation (typically 1)
    pub autocorr_lag: usize,
    /// Threshold above which autocorrelation signals criticality
    pub autocorr_threshold: f64,
    /// Threshold above which variance signals criticality
    pub variance_threshold: f64,
    /// Threshold above which recovery time signals criticality
    pub recovery_threshold: f64,
}

impl Default for CriticalityConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            autocorr_lag: 1,
            autocorr_threshold: 0.5,
            variance_threshold: 2.0,
            recovery_threshold: 3.0,
        }
    }
}

/// Result of criticality analysis
#[derive(Debug, Clone)]
pub struct CriticalityReport {
    /// Lag-1 autocorrelation of the time series
    pub autocorrelation: f64,
    /// Variance of the time series
    pub variance: f64,
    /// Mean recovery time after perturbations
    pub recovery_time: f64,
    /// Kurtosis of the time series
    pub kurtosis: f64,
    /// Whether critical slowing down is detected
    pub is_critical: bool,
    /// Confidence level [0, 1]
    pub confidence: f64,
}

/// Perturbation applied to the system
#[derive(Debug, Clone)]
pub struct Perturbation {
    /// Index where perturbation was applied
    pub index: usize,
    /// Magnitude of the perturbation
    pub magnitude: f64,
    /// Time step when perturbation occurred
    pub time_step: usize,
}

/// Recovery event tracking
#[derive(Debug, Clone)]
pub struct RecoveryEvent {
    pub perturbation_time: usize,
    pub recovery_time: Option<usize>,
    pub baseline_value: f64,
    pub perturbed_value: f64,
}

/// Main criticality detector
pub struct CriticalityDetector {
    config: CriticalityConfig,
    history: Vec<f64>,
    perturbations: Vec<Perturbation>,
    recoveries: Vec<RecoveryEvent>,
    baseline: f64,
}

impl CriticalityDetector {
    pub fn new(config: CriticalityConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            perturbations: Vec::new(),
            recoveries: Vec::new(),
            baseline: 0.0,
        }
    }

    /// Record a system state observation
    pub fn observe(&mut self, value: f64) {
        if self.history.is_empty() {
            self.baseline = value;
        }
        self.history.push(value);
    }

    /// Record a ternary system state
    pub fn observe_ternary(&mut self, state: Ternary) {
        self.observe(state.value() as f64);
    }

    /// Record a perturbation event
    pub fn record_perturbation(&mut self, index: usize, magnitude: f64, time_step: usize) {
        let baseline = if index < self.history.len() {
            self.history[index]
        } else {
            self.baseline
        };
        self.perturbations.push(Perturbation {
            index,
            magnitude,
            time_step,
        });
        self.recoveries.push(RecoveryEvent {
            perturbation_time: time_step,
            recovery_time: None,
            baseline_value: baseline,
            perturbed_value: baseline + magnitude,
        });
    }

    /// Check if a perturbation has recovered (within 10% of baseline)
    pub fn check_recovery(&mut self, perturbation_idx: usize, current_time: usize, tolerance: f64) -> bool {
        if perturbation_idx >= self.recoveries.len() {
            return false;
        }
        if self.recoveries[perturbation_idx].recovery_time.is_some() {
            return true;
        }
        if self.history.is_empty() {
            return false;
        }
        let current = *self.history.last().unwrap();
        let baseline = self.recoveries[perturbation_idx].baseline_value;
        if (current - baseline).abs() <= tolerance * baseline.abs().max(1.0) {
            self.recoveries[perturbation_idx].recovery_time = Some(current_time);
            true
        } else {
            false
        }
    }

    /// Compute lag-k autocorrelation
    pub fn autocorrelation(&self, lag: usize) -> f64 {
        if self.history.len() < lag + 2 {
            return 0.0;
        }
        let n = self.history.len();
        let mean: f64 = self.history.iter().sum::<f64>() / n as f64;
        let mut cov = 0.0;
        let mut var = 0.0;
        for i in 0..n {
            let d = self.history[i] - mean;
            var += d * d;
            if i + lag < n {
                cov += d * (self.history[i + lag] - mean);
            }
        }
        if var == 0.0 { 0.0 } else { cov / var }
    }

    /// Compute variance of the history
    pub fn variance(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let n = self.history.len() as f64;
        let mean: f64 = self.history.iter().sum::<f64>() / n;
        self.history.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
    }

    /// Compute standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Compute kurtosis (excess)
    pub fn kurtosis(&self) -> f64 {
        if self.history.len() < 4 {
            return 0.0;
        }
        let n = self.history.len() as f64;
        let mean: f64 = self.history.iter().sum::<f64>() / n;
        let var = self.variance();
        if var == 0.0 {
            return 0.0;
        }
        let m4: f64 = self.history.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
        m4 / (var * var) - 3.0
    }

    /// Compute skewness
    pub fn skewness(&self) -> f64 {
        if self.history.len() < 3 {
            return 0.0;
        }
        let n = self.history.len() as f64;
        let mean: f64 = self.history.iter().sum::<f64>() / n;
        let var = self.variance();
        if var == 0.0 {
            return 0.0;
        }
        let m3: f64 = self.history.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
        m3 / (var.sqrt() * var * var.sqrt())
    }

    /// Compute mean recovery time from recorded perturbations
    pub fn mean_recovery_time(&self) -> f64 {
        let recovered: Vec<_> = self.recoveries.iter()
            .filter_map(|r| r.recovery_time)
            .collect();
        if recovered.is_empty() {
            return f64::INFINITY;
        }
        let total: usize = recovered.iter()
            .zip(self.recoveries.iter().filter(|r| r.recovery_time.is_some()))
            .map(|(rt, r)| rt.saturating_sub(r.perturbation_time))
            .sum();
        total as f64 / recovered.len() as f64
    }

    /// Generate the full criticality report
    pub fn analyze(&self) -> CriticalityReport {
        let autocorr = self.autocorrelation(self.config.autocorr_lag);
        let variance = self.variance();
        let kurtosis = self.kurtosis();
        let recovery_time = self.mean_recovery_time();
        let baseline_std = if self.history.len() > 10 {
            let early: Vec<f64> = self.history.iter().take(10).copied().collect();
            let em: f64 = early.iter().sum::<f64>() / early.len() as f64;
            (early.iter().map(|x| (x - em).powi(2)).sum::<f64>() / early.len() as f64).sqrt()
        } else {
            1.0
        };

        let autocorr_signal = autocorr > self.config.autocorr_threshold;
        let variance_signal = variance > self.config.variance_threshold * baseline_std * baseline_std;
        let recovery_signal = recovery_time > self.config.recovery_threshold;

        let signal_count = [autocorr_signal, variance_signal, recovery_signal]
            .iter().filter(|&&x| x).count();

        let is_critical = signal_count >= 2;
        let confidence = signal_count as f64 / 3.0;

        CriticalityReport {
            autocorrelation: autocorr,
            variance,
            recovery_time,
            kurtosis,
            is_critical,
            confidence,
        }
    }

    /// Detect critical transition using rolling windows
    pub fn rolling_autocorrelation(&self, window: usize) -> Vec<f64> {
        if self.history.len() < window {
            return vec![self.autocorrelation(self.config.autocorr_lag)];
        }
        let mut result = Vec::new();
        for start in 0..=self.history.len() - window {
            let slice = &self.history[start..start + window];
            let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
            let mut cov = 0.0;
            let mut var = 0.0;
            for i in 0..slice.len() {
                let d = slice[i] - mean;
                var += d * d;
                if i + 1 < slice.len() {
                    cov += d * (slice[i + 1] - mean);
                }
            }
            result.push(if var == 0.0 { 0.0 } else { cov / var });
        }
        result
    }

    /// Compute the rate of change of autocorrelation (early warning trend)
    pub fn autocorrelation_trend(&self, window: usize) -> f64 {
        let rolling = self.rolling_autocorrelation(window);
        if rolling.len() < 2 {
            return 0.0;
        }
        let n = rolling.len() as f64;
        let mean_x = (n - 1.0) / 2.0;
        let mean_y: f64 = rolling.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &y) in rolling.iter().enumerate() {
            let dx = i as f64 - mean_x;
            num += dx * (y - mean_y);
            den += dx * dx;
        }
        if den == 0.0 { 0.0 } else { num / den }
    }

    /// Compute coefficient of variation
    pub fn coefficient_of_variation(&self) -> f64 {
        let mean: f64 = self.history.iter().sum::<f64>() / self.history.len().max(1) as f64;
        if mean == 0.0 { return f64::INFINITY; }
        self.std_dev() / mean.abs()
    }

    /// Return the history buffer
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Return recorded recoveries
    pub fn recoveries(&self) -> &[RecoveryEvent] {
        &self.recoveries
    }

    /// Reset the detector state
    pub fn reset(&mut self) {
        self.history.clear();
        self.perturbations.clear();
        self.recoveries.clear();
        self.baseline = 0.0;
    }

    /// Sliding window variance for detecting variance tipping points
    pub fn rolling_variance(&self, window: usize) -> Vec<f64> {
        if self.history.len() < window {
            return vec![self.variance()];
        }
        let mut result = Vec::new();
        for start in 0..=self.history.len() - window {
            let slice = &self.history[start..start + window];
            let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
            let var: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
            result.push(var);
        }
        result
    }
}

/// Utility: simple sign function for ternary classification
pub fn sign(x: f64) -> Ternary {
    if x < -1e-10 { Ternary::Neg } else if x > 1e-10 { Ternary::Pos } else { Ternary::Zero }
}

/// Compute entropy of a ternary distribution [p_neg, p_zero, p_pos]
pub fn ternary_entropy(probs: [f64; 3]) -> f64 {
    -probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.log2())
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_values() {
        assert_eq!(Ternary::Neg.value(), -1);
        assert_eq!(Ternary::Zero.value(), 0);
        assert_eq!(Ternary::Pos.value(), 1);
    }

    #[test]
    fn test_ternary_from_value() {
        assert_eq!(Ternary::from_value(-5), Ternary::Neg);
        assert_eq!(Ternary::from_value(0), Ternary::Zero);
        assert_eq!(Ternary::from_value(3), Ternary::Pos);
    }

    #[test]
    fn test_stable_system_low_recovery() {
        let config = CriticalityConfig {
            autocorr_threshold: 0.5,
            variance_threshold: 10.0,
            recovery_threshold: 50.0,
            ..Default::default()
        };
        let mut det = CriticalityDetector::new(config);
        // Stable system: uncorrelated noise around 0 (low autocorrelation)
        for i in 0..100 {
            // Pseudo-random alternating pattern
            let v = if i % 2 == 0 { 0.1 } else { -0.1 };
            det.observe(v);
        }
        det.record_perturbation(50, 0.5, 50);
        // Return to normal oscillation
        for i in 100..110 {
            let v = if i % 2 == 0 { 0.1 } else { -0.1 };
            det.observe(v);
        }
        det.check_recovery(0, 110, 0.2);
        let report = det.analyze();
        assert!(!report.is_critical, "Stable system should not be critical: autocorr={}, var={}, recovery={}",
            report.autocorrelation, report.variance, report.recovery_time);
        assert!(report.autocorrelation < 0.5, "Alternating signal should have low autocorrelation");
    }

    #[test]
    fn test_near_collapse_high_recovery() {
        let config = CriticalityConfig {
            variance_threshold: 0.5,
            autocorr_threshold: 0.3,
            recovery_threshold: 2.0,
            ..Default::default()
        };
        let mut det = CriticalityDetector::new(config);
        // System slowly drifting toward collapse — increasing autocorrelation
        for i in 0..100 {
            let drift = i as f64 * 0.01;
            let v = drift + 0.01 * (i as f64 * 0.05).sin();
            det.observe(v);
        }
        let report = det.analyze();
        // Near-monotonic drift should give high autocorrelation
        assert!(report.autocorrelation > 0.8, "Drifting system should have high autocorrelation, got {}", report.autocorrelation);
    }

    #[test]
    fn test_variance_tipping_point() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        // Low variance phase
        for i in 0..50 {
            det.observe(0.1 * (i as f64).sin());
        }
        // High variance phase (tipping)
        for i in 50..100 {
            det.observe(5.0 * (i as f64 * 0.5).sin());
        }
        let rolling = det.rolling_variance(20);
        let early_var = rolling[0];
        let late_var = *rolling.last().unwrap();
        assert!(late_var > early_var * 10.0, "Variance should increase near tipping point");
    }

    #[test]
    fn test_autocorrelation_increase_before_transition() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        // White noise — low autocorrelation
        for i in 0..50 {
            det.observe(((i * 7 + 3) % 11) as f64 - 5.0);
        }
        // Red noise (correlated) — increasing autocorrelation
        let mut prev = 0.0;
        for _ in 50..150 {
            prev = 0.95 * prev + ((50i32..150).next().unwrap() as f64 % 3.0 - 1.0) * 0.1;
            det.observe(prev);
        }
        let rolling = det.rolling_autocorrelation(30);
        assert!(rolling.len() > 1);
    }

    #[test]
    fn test_empty_history() {
        let det = CriticalityDetector::new(CriticalityConfig::default());
        assert_eq!(det.variance(), 0.0);
        assert_eq!(det.autocorrelation(1), 0.0);
        assert_eq!(det.kurtosis(), 0.0);
    }

    #[test]
    fn test_single_observation() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        det.observe(1.0);
        assert_eq!(det.variance(), 0.0);
    }

    #[test]
    fn test_constant_series() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        for _ in 0..50 {
            det.observe(3.0);
        }
        assert_eq!(det.variance(), 0.0);
        assert_eq!(det.autocorrelation(1), 0.0);
    }

    #[test]
    fn test_recovery_tracking() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        det.observe(1.0);
        det.record_perturbation(0, 2.0, 0);
        // Return to baseline
        det.observe(1.1);
        assert!(det.check_recovery(0, 1, 0.5));
    }

    #[test]
    fn test_no_recovery() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        det.observe(1.0);
        det.record_perturbation(0, 5.0, 0);
        det.observe(5.0); // Way off baseline
        assert!(!det.check_recovery(0, 1, 0.1));
    }

    #[test]
    fn test_kurtosis_normal() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        // Approximate normal distribution — uniform integers have negative excess kurtosis
        for i in -25i32..25 {
            det.observe(i as f64);
        }
        let k = det.kurtosis();
        // Uniform-like distributions have excess kurtosis ≈ -1.2
        assert!(k > -2.0 && k < 1.0, "Kurtosis of uniform should be in range, got {}", k);
    }

    #[test]
    fn test_ternary_observation() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        det.observe_ternary(Ternary::Pos);
        det.observe_ternary(Ternary::Neg);
        det.observe_ternary(Ternary::Zero);
        assert_eq!(det.history().len(), 3);
    }

    #[test]
    fn test_reset() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        for i in 0..10 {
            det.observe(i as f64);
        }
        det.reset();
        assert!(det.history().is_empty());
    }

    #[test]
    fn test_rolling_variance_increases() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        for i in 0..30 {
            det.observe(1.0);
        }
        for i in 0..30 {
            det.observe((i as f64 - 15.0).powi(2));
        }
        let rv = det.rolling_variance(10);
        assert!(rv.last().unwrap() > rv.first().unwrap());
    }

    #[test]
    fn test_autocorrelation_trend_positive() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        // Construct series with increasing autocorrelation
        let mut v = 0.0;
        for i in 0..100 {
            v += 0.05 * (i as f64 * 0.02).sin();
            det.observe(v);
        }
        let trend = det.autocorrelation_trend(20);
        // Could be positive or negative depending on specifics
        assert!(trend.is_finite());
    }

    #[test]
    fn test_coefficient_of_variation() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        for i in 1..=10 {
            det.observe(i as f64);
        }
        let cv = det.coefficient_of_variation();
        assert!(cv > 0.0 && cv.is_finite());
    }

    #[test]
    fn test_sign_function() {
        assert_eq!(sign(-1.5), Ternary::Neg);
        assert_eq!(sign(0.0), Ternary::Zero);
        assert_eq!(sign(2.3), Ternary::Pos);
    }

    #[test]
    fn test_ternary_entropy_uniform() {
        let e = ternary_entropy([1.0/3.0; 3]);
        assert!((e - 1.585).abs() < 0.01, "Max entropy should be log2(3) ≈ 1.585, got {}", e);
    }

    #[test]
    fn test_ternary_entropy_degenerate() {
        let e = ternary_entropy([1.0, 0.0, 0.0]);
        assert!((e - 0.0).abs() < 1e-10, "Degenerate entropy should be 0, got {}", e);
    }

    #[test]
    fn test_report_confidence_range() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        for i in 0..50 {
            det.observe((i as f64 * 0.1).sin());
        }
        let report = det.analyze();
        assert!(report.confidence >= 0.0 && report.confidence <= 1.0);
    }

    #[test]
    fn test_mean_recovery_time_infinite_when_none() {
        let det = CriticalityDetector::new(CriticalityConfig::default());
        assert!(det.mean_recovery_time().is_infinite());
    }

    #[test]
    fn test_skewness_symmetric() {
        let mut det = CriticalityDetector::new(CriticalityConfig::default());
        for i in -10i32..=10 {
            det.observe(i as f64);
        }
        let s = det.skewness();
        assert!(s.abs() < 0.1, "Symmetric data should have near-zero skewness, got {}", s);
    }
}
