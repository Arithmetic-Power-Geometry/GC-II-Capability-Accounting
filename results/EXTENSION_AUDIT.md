# GC-II extension audit

- Scalar baselines: exact Shapley, two-level factorial/Walsh ANOVA coefficients, and Sobol-style variance shares computed from the same 16-condition scalar set functions.
- Important interpretation: scalar baselines and scalar Mobius accounting receive the same aggregate information; the task-level EAS advantage comes from preserving the whole-envelope object, not from claiming a superior scalar decomposition formula.
- Held-out fully augmented digits prediction: singleton absolute error = 0.079491; best retained-order error = 0.016216 (third_order_truncated).
- Finite exact representation audit at n=12: both envelopes are balanced and have all coordinate projections through order k=2 full; parity uses 23 ROBDD nodes and the deterministic matched sample uses 729 nodes (ratio 31.70).
- The asymptotic whole-envelope separation theorem is a mathematical proof using Shannon's classical circuit-counting lower bound as an imported ingredient; the ROBDD experiment is only a finite exact representation audit.
