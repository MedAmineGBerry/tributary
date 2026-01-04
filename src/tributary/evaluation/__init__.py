"""
Evaluation and diagnostics utilities for VOLTA music marketing MMM.

This module provides tools for:
- MCMC diagnostics (convergence, divergences, ESS)
- ROAS estimation with uncertainty
- Shrinkage analysis for hierarchical models
- Model comparison (LOO-CV, WAIC)
- Posterior predictive checks
"""

from tributary.evaluation.diagnostics import (
    DiagnosticsReport,
    run_mcmc_diagnostics,
    posterior_predictive_check,
    compare_models_loo,
    compute_waic,
    format_diagnostics_report,
    energy_diagnostic,
    check_divergences_by_parameter,
)
from tributary.evaluation.roas import (
    ROASResult,
    compute_roas_from_trace,
    compute_shrinkage,
    roas_stability_comparison,
    compute_channel_contribution,
    format_roas_report,
    compute_optimal_allocation,
    compare_to_ground_truth,
)

__all__ = [
    # Diagnostics
    "DiagnosticsReport",
    "run_mcmc_diagnostics",
    "posterior_predictive_check",
    "compare_models_loo",
    "compute_waic",
    "format_diagnostics_report",
    "energy_diagnostic",
    "check_divergences_by_parameter",
    # ROAS
    "ROASResult",
    "compute_roas_from_trace",
    "compute_shrinkage",
    "roas_stability_comparison",
    "compute_channel_contribution",
    "format_roas_report",
    "compute_optimal_allocation",
    "compare_to_ground_truth",
]
