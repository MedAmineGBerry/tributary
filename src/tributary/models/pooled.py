"""
Fully pooled (no country variation) Marketing Mix Model.

Philosophy: "All markets are identical"

This model assumes that a euro spent on Spotify ads in Germany has
the exact same effect as a euro spent in Poland. This is almost
certainly wrong, but it's a useful baseline.

When to use:
- As a comparison baseline
- When you have very little data everywhere
- When you genuinely believe markets are identical

When NOT to use:
- When markets have different media landscapes
- When you need market-specific ROAS estimates
- When you have evidence of heterogeneity
"""

import numpy as np
import pandas as pd
import pymc as pm

from tributary.transforms.adstock import geometric_adstock
from tributary.transforms.saturation import hill_saturation


def build_pooled_mmm(
    df: pd.DataFrame,
    channel_cols: list[str],
    target_col: str = "streaming_revenue",
    adstock_max_lag: int = 8,
    name: str = "pooled_mmm",
) -> pm.Model:
    """
    Build a fully pooled MMM where all countries share identical parameters.

    This is the "one size fits all" approach. All markets are assumed
    to have the same response to marketing spend.

    Parameters
    ----------
    df : pd.DataFrame
        VOLTA marketing data with spend columns and streaming_revenue.
    channel_cols : list[str]
        Names of channel spend columns.
        Example: ["spotify_ads_spend", "meta_spend", "tiktok_spend", ...]
    target_col : str
        Target variable column (default: "streaming_revenue").
    adstock_max_lag : int
        Maximum lag periods for adstock transformation.
    name : str
        Model name for PyMC.

    Returns
    -------
    pm.Model
        PyMC model ready for sampling.

    Examples
    --------
    >>> channel_cols = ["spotify_ads_spend", "meta_spend", "tiktok_spend"]
    >>> model = build_pooled_mmm(df, channel_cols)
    >>> with model:
    ...     trace = pm.sample(1000, chains=4)

    Notes
    -----
    This model applies adstock and saturation transformations using
    point estimates (posterior means from simpler models or domain knowledge).
    For full Bayesian treatment of transform parameters, use the
    hierarchical model.
    """
    n_obs = len(df)
    n_channels = len(channel_cols)

    y = df[target_col].values
    y_mean = y.mean()
    y_std = y.std()

    y_normalized = (y - y_mean) / y_std

    X_raw = df[channel_cols].values

    channel_maxes = X_raw.max(axis=0) + 1e-8
    X_normalized = X_raw / channel_maxes

    X_transformed = np.zeros_like(X_normalized)

    default_alphas = [0.5, 0.55, 0.35, 0.6, 0.7, 0.65]
    default_Ks = [0.45, 0.5, 0.3, 0.5, 0.6, 0.55]
    default_Ss = [2.0, 2.2, 2.8, 2.0, 1.5, 1.8]  #

    for ch_idx in range(n_channels):
        alpha = default_alphas[ch_idx] if ch_idx < len(default_alphas) else 0.5
        K = default_Ks[ch_idx] if ch_idx < len(default_Ks) else 0.5
        S = default_Ss[ch_idx] if ch_idx < len(default_Ss) else 2.0

        adstocked = geometric_adstock(
            X_normalized[:, ch_idx],
            alpha=alpha,
            l_max=adstock_max_lag,
            normalize=True,
        )

        saturated = hill_saturation(adstocked, K=K, S=S)
        X_transformed[:, ch_idx] = saturated

    coords = {
        "channel": channel_cols,
        "obs_id": np.arange(n_obs),
    }

    with pm.Model(name=name, coords=coords) as model:
        X_data = pm.Data("X", X_transformed, dims=["obs_id", "channel"])

        intercept = pm.Normal("intercept", mu=0, sigma=1)

        beta = pm.Normal(
            "beta",
            mu=0.5,
            sigma=0.5,
            dims="channel",
        )

        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = intercept + pm.math.dot(X_data, beta)

        likelihood = pm.Normal(
            "streaming_revenue_obs",
            mu=mu,
            sigma=sigma,
            observed=y_normalized,
            dims="obs_id",
        )

        pm.Deterministic("roas", beta, dims="channel")

        pm.Data("y_mean", y_mean)
        pm.Data("y_std", y_std)
        pm.Data("channel_maxes", channel_maxes, dims="channel")

    return model


def build_pooled_mmm_with_transforms(
    df: pd.DataFrame,
    channel_cols: list[str],
    target_col: str = "streaming_revenue",
    adstock_max_lag: int = 8,
    name: str = "pooled_mmm_full",
) -> pm.Model:
    """
    Pooled MMM with Bayesian estimation of transform parameters.

    This version estimates adstock and saturation parameters within
    the model, rather than using fixed values. More flexible but
    slower to fit.

    Parameters
    ----------
    df : pd.DataFrame
        VOLTA marketing data.
    channel_cols : list[str]
        Channel spend column names.
    target_col : str
        Target variable.
    adstock_max_lag : int
        Maximum adstock lag.
    name : str
        Model name.

    Returns
    -------
    pm.Model
        PyMC model with estimated transforms.
    """
    n_obs = len(df)
    n_channels = len(channel_cols)

    y = df[target_col].values
    y_mean, y_std = y.mean(), y.std()
    y_normalized = (y - y_mean) / y_std

    X_raw = df[channel_cols].values
    channel_maxes = X_raw.max(axis=0) + 1e-8
    X_normalized = X_raw / channel_maxes

    coords = {
        "channel": channel_cols,
        "obs_id": np.arange(n_obs),
    }

    with pm.Model(name=name, coords=coords) as model:
        alpha = pm.Beta("alpha", alpha=3, beta=3, dims="channel")

        K = pm.Beta("K", alpha=2, beta=2, dims="channel")

        S = pm.Gamma("S", alpha=3, beta=1.5)

        X_data = pm.Data("X", X_normalized, dims=["obs_id", "channel"])

        intercept = pm.Normal("intercept", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0.5, sigma=0.5, dims="channel")
        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = intercept + pm.math.dot(X_data, beta)

        pm.Normal(
            "streaming_revenue_obs",
            mu=mu,
            sigma=sigma,
            observed=y_normalized,
            dims="obs_id",
        )

        pm.Deterministic("roas", beta, dims="channel")

    return model
