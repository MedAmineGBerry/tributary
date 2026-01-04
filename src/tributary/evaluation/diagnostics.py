from dataclasses import dataclass
from typing import Literal, Optional

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray


@dataclass
class DiagnosticsReport:
    rhat_summary: pd.DataFrame
    ess_summary: pd.DataFrame
    divergences: int
    max_treedepth_warnings: int
    problematic_params: list[str]
    overall_status: Literal["good", "warning", "bad"]


def run_mcmc_diagnostics(
    trace: az.InferenceData,
    rhat_threshold: float = 1.01,
    ess_threshold: int = 400,
) -> DiagnosticsReport:
    rhat = az.rhat(trace)
    rhat_df = _xarray_to_flat_df(rhat, "rhat")

    ess_bulk = az.ess(trace, method="bulk")
    ess_tail = az.ess(trace, method="tail")
    ess_bulk_df = _xarray_to_flat_df(ess_bulk, "ess_bulk")
    ess_tail_df = _xarray_to_flat_df(ess_tail, "ess_tail")
    ess_df = ess_bulk_df.merge(ess_tail_df, on="parameter")

    divergences = 0
    if "sample_stats" in trace.groups():
        diverging = trace.sample_stats.get("diverging", None)
        if diverging is not None:
            divergences = int(diverging.sum().values)

    max_treedepth_warnings = 0
    if "sample_stats" in trace.groups():
        reached_max = trace.sample_stats.get("reached_max_treedepth", None)
        if reached_max is not None:
            max_treedepth_warnings = int(reached_max.sum().values)

    problematic = []

    high_rhat_mask = rhat_df["rhat"] > rhat_threshold
    high_rhat = rhat_df[high_rhat_mask]["parameter"].tolist()
    for p in high_rhat:
        rhat_val = rhat_df.loc[rhat_df["parameter"] == p, "rhat"].values[0]
        problematic.append(f"{p} (R-hat={rhat_val:.3f})")

    low_ess_mask = (ess_df["ess_bulk"] < ess_threshold) | (
        ess_df["ess_tail"] < ess_threshold
    )
    low_ess = ess_df[low_ess_mask]["parameter"].tolist()
    for p in low_ess:
        if p not in high_rhat:
            row = ess_df.loc[ess_df["parameter"] == p].iloc[0]
            problematic.append(
                f"{p} (ESS_bulk={row['ess_bulk']:.0f}, ESS_tail={row['ess_tail']:.0f})"
            )

    if divergences > 0 or len(high_rhat) > 0:
        status = "bad"
    elif max_treedepth_warnings > 10 or len(low_ess) > 0:
        status = "warning"
    else:
        status = "good"

    return DiagnosticsReport(
        rhat_summary=rhat_df,
        ess_summary=ess_df,
        divergences=divergences,
        max_treedepth_warnings=max_treedepth_warnings,
        problematic_params=problematic,
        overall_status=status,
    )


def _xarray_to_flat_df(ds: xr.Dataset, value_name: str) -> pd.DataFrame:
    records = []

    for var in ds.data_vars:
        data = ds[var]

        if data.dims:
            for idx in np.ndindex(*data.shape):
                coord_parts = []
                for dim, i in zip(data.dims, idx):
                    coord_parts.append(str(data.coords[dim].values[i]))
                param_name = f"{var}[{'_'.join(coord_parts)}]"
                records.append(
                    {"parameter": param_name, value_name: float(data.values[idx])}
                )
        else:
            records.append({"parameter": var, value_name: float(data.values)})

    return pd.DataFrame(records)


def check_divergences_by_parameter(
    trace: az.InferenceData,
    var_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    if "sample_stats" not in trace.groups():
        return pd.DataFrame(
            columns=["parameter", "mean_divergent", "mean_ok", "abs_diff"]
        )

    diverging = trace.sample_stats.get("diverging", None)
    if diverging is None or diverging.sum() == 0:
        return pd.DataFrame(
            columns=["parameter", "mean_divergent", "mean_ok", "abs_diff"]
        )

    posterior = trace.posterior
    diverging_flat = diverging.values.flatten()

    if var_names is None:
        var_names = list(posterior.data_vars)

    records = []

    for var in var_names:
        if var not in posterior:
            continue

        data = posterior[var]
        flat_shape = (-1,) + data.shape[2:]
        data_flat = data.values.reshape(flat_shape)

        mean_div = data_flat[diverging_flat].mean()
        mean_ok = data_flat[~diverging_flat].mean()

        name = var if data_flat.ndim == 1 else f"{var} (all)"
        records.append(
            {
                "parameter": name,
                "mean_divergent": mean_div,
                "mean_ok": mean_ok,
                "abs_diff": abs(mean_div - mean_ok),
            }
        )

    return pd.DataFrame(records).sort_values("abs_diff", ascending=False)


def posterior_predictive_check(
    trace: az.InferenceData,
    y_obs: NDArray[np.floating],
    var_name: str = "streaming_revenue_obs",
) -> dict[str, float]:
    if "posterior_predictive" not in trace.groups():
        raise ValueError("Trace must contain posterior_predictive group.")

    ppc = trace.posterior_predictive[var_name]
    ppc_flat = ppc.stack(sample=["chain", "draw"])

    ppc_mean = ppc_flat.mean(dim="sample").values
    ppc_hdi = az.hdi(ppc, hdi_prob=0.94)[var_name]

    hdi_low = ppc_hdi.sel(hdi="lower").values
    hdi_high = ppc_hdi.sel(hdi="higher").values

    rmse = np.sqrt(np.mean((y_obs - ppc_mean) ** 2))
    mae = np.mean(np.abs(y_obs - ppc_mean))
    coverage = np.mean((y_obs >= hdi_low) & (y_obs <= hdi_high))
    calibration_error = abs(coverage - 0.94)
    sharpness = np.mean(hdi_high - hdi_low)
    mean_bias = np.mean(ppc_mean - y_obs)
    nrmse = rmse / np.mean(y_obs) if np.mean(y_obs) > 0 else np.nan

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "nrmse": float(nrmse),
        "coverage_94": float(coverage),
        "calibration_error": float(calibration_error),
        "interval_sharpness": float(sharpness),
        "mean_bias": float(mean_bias),
    }


def posterior_predictive_check_by_country(
    trace: az.InferenceData,
    df: pd.DataFrame,
    var_name: str = "streaming_revenue_obs",
    country_col: str = "country",
    target_col: str = "streaming_revenue",
) -> pd.DataFrame:
    if "posterior_predictive" not in trace.groups():
        raise ValueError("Trace must contain posterior_predictive group.")

    ppc = trace.posterior_predictive[var_name]
    ppc_flat = ppc.stack(sample=["chain", "draw"])
    ppc_mean = ppc_flat.mean(dim="sample").values

    ppc_hdi = az.hdi(ppc, hdi_prob=0.94)[var_name]
    hdi_low = ppc_hdi.sel(hdi="lower").values
    hdi_high = ppc_hdi.sel(hdi="higher").values

    y_obs = df[target_col].values
    countries = df[country_col].values

    records = []

    for country in df[country_col].unique():
        mask = countries == country

        rmse = np.sqrt(np.mean((y_obs[mask] - ppc_mean[mask]) ** 2))
        mae = np.mean(np.abs(y_obs[mask] - ppc_mean[mask]))
        coverage = np.mean(
            (y_obs[mask] >= hdi_low[mask]) & (y_obs[mask] <= hdi_high[mask])
        )

        records.append(
            {
                "country": country,
                "n_obs": mask.sum(),
                "rmse": rmse,
                "mae": mae,
                "coverage_94": coverage,
                "calibration_error": abs(coverage - 0.94),
                "mean_bias": np.mean(ppc_mean[mask] - y_obs[mask]),
            }
        )

    return pd.DataFrame(records).sort_values("country")


def compare_models_loo(
    traces: dict[str, az.InferenceData],
    scale: str = "log",
) -> pd.DataFrame:
    return az.compare(traces, ic="loo", scale=scale)


def compute_waic(trace: az.InferenceData) -> dict[str, float]:
    waic_result = az.waic(trace)
    return {
        "waic": float(waic_result.elpd_waic),
        "waic_se": float(waic_result.se),
        "p_waic": float(waic_result.p_waic),
    }


def energy_diagnostic(trace: az.InferenceData) -> dict[str, float]:
    if "sample_stats" not in trace.groups():
        return {"energy_fmi": None, "energy_warning": "No sample_stats in trace"}

    energy = trace.sample_stats.get("energy", None)
    if energy is None:
        return {"energy_fmi": None, "energy_warning": "No energy data in trace"}

    energy_vals = energy.values
    energy_diff = np.diff(energy_vals, axis=1)

    var_energy = np.var(energy_vals)
    var_diff = np.var(energy_diff) / 2

    e_fmi = var_diff / var_energy if var_energy > 0 else 0

    warning = "OK"
    if e_fmi < 0.2:
        warning = "Very low E-FMI"
    elif e_fmi < 0.3:
        warning = "Low E-FMI"

    return {"energy_fmi": float(e_fmi), "energy_warning": warning}


def format_diagnostics_report(report: DiagnosticsReport) -> str:
    status_emoji = {"good": "✅", "warning": "⚠️", "bad": "❌"}
    status_meaning = {
        "good": "All checks passed — estimates are reliable",
        "warning": "Minor issues — interpret with caution",
        "bad": "Major issues — DO NOT USE these estimates",
    }

    lines = [
        "=" * 60,
        f"MCMC DIAGNOSTICS REPORT  {status_emoji[report.overall_status]}",
        "=" * 60,
        "",
        f"Overall Status: {report.overall_status.upper()}",
        f"  → {status_meaning[report.overall_status]}",
        "",
        f"Divergent Transitions: {report.divergences}",
        f"Max Treedepth Warnings: {report.max_treedepth_warnings}",
        "",
        f"Max R-hat: {report.rhat_summary['rhat'].max():.4f}",
        f"Min Bulk ESS: {report.ess_summary['ess_bulk'].min():.0f}",
        f"Min Tail ESS: {report.ess_summary['ess_tail'].min():.0f}",
    ]

    if report.problematic_params:
        lines.append("")
        lines.append("Problematic parameters:")
        for p in report.problematic_params[:10]:
            lines.append(f"  • {p}")

    lines.append("=" * 60)
    return "\n".join(lines)


def quick_diagnostics(trace: az.InferenceData) -> None:
    report = run_mcmc_diagnostics(trace)
    emoji = {"good": "✅", "warning": "⚠️", "bad": "❌"}[report.overall_status]

    print(f"{emoji} Diagnostics: {report.overall_status.upper()}")
    print(f"   Divergences: {report.divergences}")
    print(f"   Max R-hat: {report.rhat_summary['rhat'].max():.3f}")
    print(f"   Min ESS: {report.ess_summary['ess_bulk'].min():.0f}")

    if report.problematic_params:
        print(f"   Issues: {report.problematic_params[:3]}")
