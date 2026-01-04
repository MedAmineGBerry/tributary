import numpy as np
import pandas as pd
import pymc as pm

from tributary.transforms.adstock import geometric_adstock
from tributary.transforms.saturation import hill_saturation


def build_unpooled_mmm(
    df: pd.DataFrame,
    channel_cols: list[str],
    country_col: str = "country",
    target_col: str = "streaming_revenue",
    adstock_max_lag: int = 8,
    name: str = "unpooled_mmm",
) -> pm.Model:
    countries = df[country_col].unique().tolist()
    n_obs = len(df)

    country_to_idx = {c: i for i, c in enumerate(countries)}
    country_idx = df[country_col].map(country_to_idx).values

    y = df[target_col].values
    y_mean, y_std = y.mean(), y.std()
    y_normalized = (y - y_mean) / y_std

    X_raw = df[channel_cols].values
    channel_maxes = X_raw.max(axis=0) + 1e-8
    X_normalized = X_raw / channel_maxes

    transform_params = {
        "spotify_ads_spend": {"alpha": 0.50, "K": 0.45, "S": 2.0},
        "meta_spend": {"alpha": 0.55, "K": 0.50, "S": 2.2},
        "tiktok_spend": {"alpha": 0.35, "K": 0.30, "S": 2.8},
        "youtube_spend": {"alpha": 0.60, "K": 0.50, "S": 2.0},
        "radio_spend": {"alpha": 0.70, "K": 0.60, "S": 1.5},
        "playlist_spend": {"alpha": 0.65, "K": 0.55, "S": 1.8},
    }

    X_transformed = np.zeros_like(X_normalized)
    for ch_idx, channel in enumerate(channel_cols):
        params = transform_params.get(channel, {"alpha": 0.5, "K": 0.5, "S": 2.0})
        adstocked = geometric_adstock(
            X_normalized[:, ch_idx],
            alpha=params["alpha"],
            l_max=adstock_max_lag,
            normalize=True,
        )
        X_transformed[:, ch_idx] = hill_saturation(
            adstocked, K=params["K"], S=params["S"]
        )

    coords = {
        "country": countries,
        "channel": channel_cols,
        "obs_id": np.arange(n_obs),
    }

    with pm.Model(name=name, coords=coords) as model:
        X_data = pm.Data("X", X_transformed, dims=["obs_id", "channel"])
        country_idx_data = pm.Data("country_idx", country_idx, dims="obs_id")

        intercept = pm.Normal("intercept", mu=0, sigma=1, dims="country")

        beta = pm.Normal("beta", mu=0.5, sigma=0.5, dims=("country", "channel"))

        sigma = pm.HalfNormal("sigma", sigma=1, dims="country")

        intercept_obs = intercept[country_idx_data]
        sigma_obs = sigma[country_idx_data]
        beta_obs = beta[country_idx_data, :]

        channel_contribution = (X_data * beta_obs).sum(axis=1)
        mu = intercept_obs + channel_contribution

        pm.Normal(
            "streaming_revenue_obs",
            mu=mu,
            sigma=sigma_obs,
            observed=y_normalized,
            dims="obs_id",
        )

        pm.Deterministic("roas", beta, dims=("country", "channel"))

        pm.Data("y_mean", y_mean)
        pm.Data("y_std", y_std)

    return model


def build_unpooled_mmm_with_transforms(
    df: pd.DataFrame,
    channel_cols: list[str],
    country_col: str = "country",
    target_col: str = "streaming_revenue",
    adstock_max_lag: int = 8,
    name: str = "unpooled_mmm_full",
) -> pm.Model:
    countries = df[country_col].unique().tolist()
    n_obs = len(df)

    country_to_idx = {c: i for i, c in enumerate(countries)}
    country_idx = df[country_col].map(country_to_idx).values

    y = df[target_col].values
    y_mean, y_std = y.mean(), y.std()
    y_normalized = (y - y_mean) / y_std

    X_raw = df[channel_cols].values
    channel_maxes = X_raw.max(axis=0) + 1e-8
    X_normalized = X_raw / channel_maxes

    transform_params = {
        "spotify_ads_spend": {"alpha": 0.50, "K": 0.45, "S": 2.0},
        "meta_spend": {"alpha": 0.55, "K": 0.50, "S": 2.2},
        "tiktok_spend": {"alpha": 0.35, "K": 0.30, "S": 2.8},
        "youtube_spend": {"alpha": 0.60, "K": 0.50, "S": 2.0},
        "radio_spend": {"alpha": 0.70, "K": 0.60, "S": 1.5},
        "playlist_spend": {"alpha": 0.65, "K": 0.55, "S": 1.8},
    }

    X_transformed = np.zeros_like(X_normalized)
    for ch_idx, channel in enumerate(channel_cols):
        params = transform_params.get(channel, {"alpha": 0.5, "K": 0.5, "S": 2.0})
        adstocked = geometric_adstock(
            X_normalized[:, ch_idx],
            alpha=params["alpha"],
            l_max=adstock_max_lag,
            normalize=True,
        )
        X_transformed[:, ch_idx] = hill_saturation(
            adstocked, K=params["K"], S=params["S"]
        )

    coords = {
        "country": countries,
        "channel": channel_cols,
        "obs_id": np.arange(n_obs),
    }

    with pm.Model(name=name, coords=coords) as model:
        X_data = pm.Data("X", X_transformed, dims=["obs_id", "channel"])
        country_idx_data = pm.Data("country_idx", country_idx, dims="obs_id")

        intercept = pm.Normal("intercept", mu=0, sigma=1, dims="country")
        beta = pm.Normal("beta", mu=0.5, sigma=0.5, dims=("country", "channel"))
        sigma = pm.HalfNormal("sigma", sigma=1, dims="country")

        intercept_obs = intercept[country_idx_data]
        sigma_obs = sigma[country_idx_data]
        beta_obs = beta[country_idx_data, :]

        channel_contribution = (X_data * beta_obs).sum(axis=1)
        mu = intercept_obs + channel_contribution

        pm.Normal(
            "streaming_revenue_obs",
            mu=mu,
            sigma=sigma_obs,
            observed=y_normalized,
            dims="obs_id",
        )

        pm.Deterministic("roas", beta, dims=("country", "channel"))

        pm.Data("y_mean", y_mean)
        pm.Data("y_std", y_std)

    return model
