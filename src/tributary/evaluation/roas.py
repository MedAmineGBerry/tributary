from dataclasses import dataclass
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class ROASResult:
    mean: pd.DataFrame
    std: pd.DataFrame
    hdi_low: pd.DataFrame
    hdi_high: pd.DataFrame
    stability_score: pd.Series
    shrinkage: Optional[pd.DataFrame] = None


def compute_roas_from_trace(
    trace: az.InferenceData,
    spend_data: pd.DataFrame,
    channel_cols: list[str],
    country_col: str = "country",
    hdi_prob: float = 0.94,
) -> ROASResult:
    posterior = trace.posterior

    if "roas" in posterior:
        roas_samples = posterior["roas"]
    elif "beta" in posterior:
        roas_samples = posterior["beta"]
    else:
        raise ValueError("Trace must contain 'roas' or 'beta' variable")

    if "country" in roas_samples.dims:
        countries = roas_samples.coords["country"].values
        channels = roas_samples.coords["channel"].values

        roas_mean = roas_samples.mean(dim=["chain", "draw"]).values
        roas_std = roas_samples.std(dim=["chain", "draw"]).values

        hdi = az.hdi(roas_samples, hdi_prob=hdi_prob)
        hdi_low = hdi.sel(hdi="lower").values
        hdi_high = hdi.sel(hdi="higher").values
    else:
        channels = roas_samples.coords["channel"].values
        countries = spend_data[country_col].unique()

        roas_mean_1d = roas_samples.mean(dim=["chain", "draw"]).values
        roas_std_1d = roas_samples.std(dim=["chain", "draw"]).values

        hdi = az.hdi(roas_samples, hdi_prob=hdi_prob)
        hdi_low_1d = hdi.sel(hdi="lower").values
        hdi_high_1d = hdi.sel(hdi="higher").values

        roas_mean = np.tile(roas_mean_1d, (len(countries), 1))
        roas_std = np.tile(roas_std_1d, (len(countries), 1))
        hdi_low = np.tile(hdi_low_1d, (len(countries), 1))
        hdi_high = np.tile(hdi_high_1d, (len(countries), 1))

    mean_df = pd.DataFrame(roas_mean, index=countries, columns=channels)
    std_df = pd.DataFrame(roas_std, index=countries, columns=channels)
    hdi_low_df = pd.DataFrame(hdi_low, index=countries, columns=channels)
    hdi_high_df = pd.DataFrame(hdi_high, index=countries, columns=channels)

    stability = std_df.mean(axis=0) / mean_df.mean(axis=0).abs().clip(lower=1e-6)
    stability.name = "stability_cv"

    return ROASResult(
        mean=mean_df,
        std=std_df,
        hdi_low=hdi_low_df,
        hdi_high=hdi_high_df,
        stability_score=stability,
    )


def compute_shrinkage(
    trace: az.InferenceData,
    unpooled_trace: Optional[az.InferenceData] = None,
) -> pd.DataFrame:
    posterior = trace.posterior

    if "beta" not in posterior:
        raise ValueError("Trace must contain 'beta' for shrinkage calculation")

    beta = posterior["beta"]

    if "country" not in beta.dims:
        raise ValueError("Trace must be from hierarchical model with country dimension")

    countries = beta.coords["country"].values
    channels = beta.coords["channel"].values

    if unpooled_trace is not None and "beta" in unpooled_trace.posterior:
        unpooled_beta = unpooled_trace.posterior["beta"]

        hier_var = beta.var(dim=["chain", "draw"]).values
        unpooled_var = unpooled_beta.var(dim=["chain", "draw"]).values

        shrinkage = 1 - (hier_var / (unpooled_var + 1e-8))
        shrinkage = np.clip(shrinkage, 0, 1)
    else:
        beta_mean = beta.mean(dim=["chain", "draw"])
        global_mean = beta_mean.mean(dim="country")

        posterior_var = beta.var(dim=["chain", "draw"])
        between_var = beta_mean.var(dim="country")

        total_var = between_var + posterior_var.mean(dim="country")

        shrinkage_raw = 1 - (posterior_var.values / (total_var.values + 1e-8))
        shrinkage = np.clip(shrinkage_raw, 0, 1)

    return pd.DataFrame(shrinkage, index=countries, columns=channels)


def roas_stability_comparison(
    traces: dict[str, az.InferenceData],
    channel_cols: list[str],
) -> pd.DataFrame:
    results = []

    for model_name, trace in traces.items():
        posterior = trace.posterior
        if "beta" not in posterior:
            continue

        beta = posterior["beta"]

        if "country" in beta.dims:
            beta_means = beta.mean(dim=["chain", "draw"])
            cross_country_std = beta_means.std(dim="country")
            posterior_std = beta.std(dim=["chain", "draw"]).mean(dim="country")
        else:
            cross_country_std = xr.DataArray(
                np.zeros(len(channel_cols)),
                dims=["channel"],
                coords={"channel": channel_cols},
            )
            posterior_std = beta.std(dim=["chain", "draw"])

        for ch_idx, channel in enumerate(channel_cols):
            results.append(
                {
                    "model": model_name,
                    "channel": channel,
                    "cross_country_std": float(
                        cross_country_std.isel(channel=ch_idx).values
                    ),
                    "posterior_std": float(posterior_std.isel(channel=ch_idx).values),
                }
            )

    df = pd.DataFrame(results)
    pivot = df.pivot(index="channel", columns="model", values="cross_country_std")

    desired_order = ["pooled", "unpooled", "hierarchical"]
    available = [c for c in desired_order if c in pivot.columns]
    return pivot[available]


def compute_channel_contribution(
    trace: az.InferenceData,
    df: pd.DataFrame,
    channel_cols: list[str],
    country_col: str = "country",
    target_col: str = "streaming_revenue",
) -> pd.DataFrame:
    posterior = trace.posterior

    if "beta" not in posterior:
        raise ValueError("Trace must contain 'beta'")

    beta = posterior["beta"]
    beta_mean = beta.mean(dim=["chain", "draw"])

    countries = df[country_col].unique()
    contributions = np.zeros((len(countries), len(channel_cols)))

    for c_idx, country in enumerate(countries):
        mask = df[country_col] == country
        country_df = df.loc[mask]

        for ch_idx, channel in enumerate(channel_cols):
            total_spend = float(country_df[channel].sum())

            if "country" in beta_mean.dims:
                beta_val = float(beta_mean.sel(country=country, channel=channel).values)
            else:
                beta_val = float(beta_mean.sel(channel=channel).values)

            contributions[c_idx, ch_idx] = beta_val * total_spend

    return pd.DataFrame(contributions, index=countries, columns=channel_cols)


def compute_optimal_allocation(
    roas_result: ROASResult,
    total_budget: float,
    min_per_channel: float = 0.0,
    max_per_channel: Optional[float] = None,
) -> pd.DataFrame:
    roas_mean = roas_result.mean.copy()
    roas_positive = roas_mean.clip(lower=0.01)

    weights = roas_positive / roas_positive.sum().sum()
    allocation = weights * float(total_budget)

    if min_per_channel > 0:
        allocation = allocation.clip(lower=float(min_per_channel))

    if max_per_channel is not None:
        allocation = allocation.clip(upper=float(max_per_channel))

    allocation = allocation * (float(total_budget) / allocation.sum().sum())
    return allocation.round(2)


def compare_to_ground_truth(
    trace: az.InferenceData,
    true_params: "TrueParameters",
    param_name: str = "beta_mu",
) -> pd.DataFrame:
    posterior = trace.posterior

    if param_name not in posterior:
        raise ValueError(f"Parameter '{param_name}' not in trace")

    param = posterior[param_name]
    param_mean = param.mean(dim=["chain", "draw"]).values
    param_std = param.std(dim=["chain", "draw"]).values

    hdi = az.hdi(param, hdi_prob=0.94)[param_name]
    hdi_low = hdi.sel(hdi="lower").values
    hdi_high = hdi.sel(hdi="higher").values

    true_values = getattr(true_params, param_name)
    covered = (true_values >= hdi_low) & (true_values <= hdi_high)

    channels = true_params.channel_names

    return pd.DataFrame(
        {
            "channel": channels,
            "true_value": true_values,
            "recovered_mean": param_mean,
            "recovered_std": param_std,
            "hdi_low": hdi_low,
            "hdi_high": hdi_high,
            "covered": covered,
            "error": param_mean - true_values,
            "abs_error": np.abs(param_mean - true_values),
        }
    )


def format_roas_report(
    roas_result: ROASResult,
    decimals: int = 3,
    include_recommendations: bool = True,
) -> str:
    lines = [
        "=" * 70,
        "VOLTA MUSIC GROUP — ROAS ANALYSIS REPORT",
        "=" * 70,
        "",
        "MEAN ROAS BY MARKET × CHANNEL",
        "(Streaming revenue per EUR spent)",
        "-" * 50,
        roas_result.mean.round(decimals).to_string(),
        "",
        "ROAS UNCERTAINTY (Standard Deviation)",
        "-" * 50,
        roas_result.std.round(decimals).to_string(),
        "",
        "94% HDI BOUNDS",
        "-" * 50,
        "Lower bound:",
        roas_result.hdi_low.round(decimals).to_string(),
        "",
        "Upper bound:",
        roas_result.hdi_high.round(decimals).to_string(),
        "",
        "CHANNEL STABILITY SCORE",
        "(Cross-country CV — lower = more consistent across markets)",
        "-" * 50,
        roas_result.stability_score.round(decimals).to_string(),
        "",
    ]

    if include_recommendations:
        mean_roas = roas_result.mean.mean()
        best_channel = mean_roas.idxmax()
        worst_channel = mean_roas.idxmin()

        most_stable = roas_result.stability_score.idxmin()
        least_stable = roas_result.stability_score.idxmax()

        lines.extend(
            [
                "─" * 50,
                "KEY INSIGHTS",
                "─" * 50,
                f"• Highest average ROAS: {best_channel.replace('_spend', '')} ({mean_roas[best_channel]:.3f})",
                f"• Lowest average ROAS: {worst_channel.replace('_spend', '')} ({mean_roas[worst_channel]:.3f})",
                f"• Most stable across markets: {most_stable.replace('_spend', '')}",
                f"• Least stable (highest variance): {least_stable.replace('_spend', '')}",
                "",
                "RECOMMENDATIONS",
                "-" * 50,
                "• Prioritize budget toward high-ROAS, high-stability channels",
                "• For sparse markets, prefer hierarchical estimates over market-only fits",
                "• Consider uncertainty intervals when setting targets",
                "",
            ]
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def quick_roas_summary(
    trace: az.InferenceData,
    df: pd.DataFrame,
    channel_cols: list[str],
) -> None:
    roas = compute_roas_from_trace(trace, df, channel_cols)

    print("📊 ROAS Summary (Mean ± Std across markets)")
    print()

    for channel in channel_cols:
        mean_val = float(roas.mean[channel].mean())
        std_val = float(roas.std[channel].mean())

        cv = std_val / abs(mean_val) if abs(mean_val) > 0.01 else float("inf")
        flag = "  ⚠️ High uncertainty" if cv > 0.5 else ""

        channel_short = channel.replace("_spend", "")
        print(f"  {channel_short:18s}: {mean_val:.3f} ± {std_val:.3f}{flag}")
