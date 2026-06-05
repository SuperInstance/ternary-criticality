# ternary-criticality

Critical slowing down detector for ternary systems.

Monitors recovery time from perturbations, diverging autocorrelation, and variance changes as early warning signals of system collapse. Built for ternary-state (-1, 0, +1) systems but works with any continuous time series.

## Features

- **Autocorrelation tracking**: Lag-k autocorrelation with rolling windows
- **Variance monitoring**: Detect variance tipping points via sliding windows
- **Recovery time measurement**: Track system recovery from perturbations
- **Kurtosis & skewness**: Higher-order moment analysis
- **Criticality reports**: Integrated analysis with confidence scoring
- **Early warning trends**: Autocorrelation trend detection via regression

## Usage

```rust
use ternary_criticality::{CriticalityDetector, CriticalityConfig, Ternary};

let config = CriticalityConfig::default();
let mut detector = CriticalityDetector::new(config);

// Observe system states
detector.observe_ternary(Ternary::Pos);
detector.observe_ternary(Ternary::Zero);

// Or continuous values
detector.observe(0.42);

// Record and check perturbations
detector.record_perturbation(0, 1.0, 0);
detector.check_recovery(0, 10, 0.1);

// Get analysis
let report = detector.analyze();
println!("Critical: {} (confidence: {})", report.is_critical, report.confidence);
```

## Test Coverage

23 tests covering stable systems, near-collapse detection, variance tipping points, autocorrelation trends, recovery tracking, and edge cases.

## Known Limitations

- No actual randomness — users must provide their own time series data
- Autocorrelation uses biased estimator (divides by N, not N-k)
- Recovery detection uses simple threshold comparison, not statistical significance
- Kurtosis uses excess kurtosis (normal = 0), which may be surprising for platykurtic distributions
- Rolling window operations allocate new vectors each call
- No multivariate criticality detection

## License

MIT
