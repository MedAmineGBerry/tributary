import numpy as np
import pandas as pd
import pymc as pm
from typing import Optional

from tributary.transforms.adstock import geometric_adstock
from tributary.transforms.saturation import hill_saturation


def build_hierarchical_mmm(
    df: pd.DataFrame,
    channel_cols: list[str],
    country_col: str = "country",
    target_col: str = "streaming_revenue",
    adstock_max_lag: int = 8,
    centered: bool = False,
    name: str = "hierarchical_mmm",
) -> pm.Model:
    countries = df[country_col].unique().tolist()
    n_obs = len(df)

    country_to_idx = {c: i for i, c in enumerate(countries)}
    country_idx = df[country_col].map(country_to_idx).values

    obs_per_country = df.groupby(country_col).size().to_dict()

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

        beta_mu = pm.Normal("beta_mu", mu=0.5, sigma=0.5, dims="channel")
        beta_sigma = pm.HalfNormal("beta_sigma", sigma=0.3, dims="channel")

        intercept_mu = pm.Normal("intercept_mu", mu=0, sigma=1)
        intercept_sigma = pm.HalfNormal("intercept_sigma", sigma=0.5)

        if centered:
            beta = pm.Normal(
                "beta", mu=beta_mu, sigma=beta_sigma, dims=("country", "channel")
            )
            intercept = pm.Normal(
                "intercept", mu=intercept_mu, sigma=intercept_sigma, dims="country"
            )
        else:
            beta_offset = pm.Normal(
                "beta_offset",
                mu=0,
                sigma=1,
                dims=("country", "channel"),
            )
            beta = pm.Deterministic(
                "beta",
                beta_mu + beta_sigma * beta_offset,
                dims=("country", "channel"),
            )

            intercept_offset = pm.Normal(
                "intercept_offset", mu=0, sigma=1, dims="country"
            )
            intercept = pm.Deterministic(
                "intercept",
                intercept_mu + intercept_sigma * intercept_offset,
                dims="country",
            )

        sigma = pm.HalfNormal("sigma", sigma=1)

        intercept_obs = intercept[country_idx_data]
        beta_obs = beta[country_idx_data, :]

        channel_contribution = (X_data * beta_obs).sum(axis=1)
        mu = intercept_obs + channel_contribution

        pm.Normal(
            "streaming_revenue_obs",
            mu=mu,
            sigma=sigma,
            observed=y_normalized,
            dims="obs_id",
        )

        pm.Deterministic("roas", beta, dims=("country", "channel"))
        pm.Deterministic("roas_global", beta_mu, dims="channel")

        pm.Deterministic(
            "shrinkage_factor",
            beta_sigma**2 / (beta_sigma**2 + sigma**2),
            dims="channel",
        )

        pm.Data("y_mean", y_mean)
        pm.Data("y_std", y_std)
        pm.Data(
            "n_obs_per_country",
            np.array([obs_per_country[c] for c in countries]),
            dims="country",
        )

    return model


def build_hierarchical_mmm_full(
    df: pd.DataFrame,
    channel_cols: list[str],
    country_col: str = "country",
    target_col: str = "streaming_revenue",
    adstock_max_lag: int = 8,
    name: str = "hierarchical_mmm_full",
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

    coords = {
        "country": countries,
        "channel": channel_cols,
        "obs_id": np.arange(n_obs),
    }

    with pm.Model(name=name, coords=coords) as model:
        X_data = pm.Data("X", X_normalized, dims=["obs_id", "channel"])
        country_idx_data = pm.Data("country_idx", country_idx, dims="obs_id")

        alpha_mu = pm.Beta("alpha_mu", alpha=3, beta=3, dims="channel")
        alpha_kappa = pm.HalfNormal("alpha_kappa", sigma=10, dims="channel")

        alpha = pm.Beta(
            "alpha",
            alpha=alpha_mu * alpha_kappa,
            beta=(1 - alpha_mu) * alpha_kappa,
            dims=("country", "channel"),
        )

        K_mu = pm.Beta("K_mu", alpha=2, beta=2, dims="channel")
        K_kappa = pm.HalfNormal("K_kappa", sigma=10, dims="channel")

        K = pm.Beta(
            "K",
            alpha=K_mu * K_kappa,
            beta=(1 - K_mu) * K_kappa,
            dims=("country", "channel"),
        )

        S = pm.Gamma("S", alpha=3, beta=1.5, dims="channel")

        beta_mu = pm.Normal("beta_mu", mu=0.5, sigma=0.5, dims="channel")
        beta_sigma = pm.HalfNormal("beta_sigma", sigma=0.3, dims="channel")

        beta_offset = pm.Normal(
            "beta_offset", mu=0, sigma=1, dims=("country", "channel")
        )
        beta = pm.Deterministic(
            "beta",
            beta_mu + beta_sigma * beta_offset,
            dims=("country", "channel"),
        )

        intercept_mu = pm.Normal("intercept_mu", mu=0, sigma=1)
        intercept_sigma = pm.HalfNormal("intercept_sigma", sigma=0.5)

        intercept_offset = pm.Normal("intercept_offset", mu=0, sigma=1, dims="country")
        intercept = pm.Deterministic(
            "intercept",
            intercept_mu + intercept_sigma * intercept_offset,
            dims="country",
        )

        sigma = pm.HalfNormal("sigma", sigma=1)

        intercept_obs = intercept[country_idx_data]
        beta_obs = beta[country_idx_data, :]

        channel_contribution = (X_data * beta_obs).sum(axis=1)
        mu = intercept_obs + channel_contribution

        pm.Normal(
            "streaming_revenue_obs",
            mu=mu,
            sigma=sigma,
            observed=y_normalized,
            dims="obs_id",
        )

        pm.Deterministic("roas", beta, dims=("country", "channel"))
        pm.Deterministic("roas_global", beta_mu, dims="channel")
        pm.Deterministic("alpha_global", alpha_mu, dims="channel")
        pm.Deterministic("K_global", K_mu, dims="channel")

        pm.Data("y_mean", y_mean)
        pm.Data("y_std", y_std)

    return model


def build_hierarchical_mmm_regional(
    df: pd.DataFrame,
    channel_cols: list[str],
    country_col: str = "country",
    target_col: str = "streaming_revenue",
    region_mapping: Optional[dict[str, str]] = None,
    adstock_max_lag: int = 8,
    name: str = "hierarchical_mmm_regional",
) -> pm.Model:
    if region_mapping is None:
        region_mapping = {
            "DE": "DACH",
            "FR": "Western",
            "UK": "Western",
            "NL": "Western",
            "ES": "Southern",
            "IT": "Southern",
            "PL": "Nordic_CEE",
            "SE": "Nordic_CEE",
        }

    countries = df[country_col].unique().tolist()
    regions = list(set(region_mapping[c] for c in countries))
    n_obs = len(df)

    country_to_idx = {c: i for i, c in enumerate(countries)}
    region_to_idx = {r: i for i, r in enumerate(regions)}

    country_idx = df[country_col].map(country_to_idx).values
    country_to_region_idx = np.array(
        [region_to_idx[region_mapping[c]] for c in countries]
    )

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
        "region": regions,
        "country": countries,
        "channel": channel_cols,
        "obs_id": np.arange(n_obs),
    }

    with pm.Model(name=name, coords=coords) as model:
        X_data = pm.Data("X", X_transformed, dims=["obs_id", "channel"])
        country_idx_data = pm.Data("country_idx", country_idx, dims="obs_id")
        country_region_idx = pm.Data(
            "country_region_idx", country_to_region_idx, dims="country"
        )

        beta_global = pm.Normal("beta_global", mu=0.5, sigma=0.5, dims="channel")
        beta_region_sigma = pm.HalfNormal(
            "beta_region_sigma", sigma=0.3, dims="channel"
        )

        beta_region_offset = pm.Normal(
            "beta_region_offset",
            mu=0,
            sigma=1,
            dims=("region", "channel"),
        )
        beta_region = pm.Deterministic(
            "beta_region",
            beta_global + beta_region_sigma * beta_region_offset,
            dims=("region", "channel"),
        )

        beta_country_sigma = pm.HalfNormal(
            "beta_country_sigma", sigma=0.2, dims="channel"
        )

        beta_country_offset = pm.Normal(
            "beta_country_offset",
            mu=0,
            sigma=1,
            dims=("country", "channel"),
        )

        beta = pm.Deterministic(
            "beta",
            beta_region[country_region_idx, :]
            + beta_country_sigma * beta_country_offset,
            dims=("country", "channel"),
        )

        intercept_mu = pm.Normal("intercept_mu", mu=0, sigma=1)
        intercept_sigma = pm.HalfNormal("intercept_sigma", sigma=0.5)
        intercept_offset = pm.Normal("intercept_offset", mu=0, sigma=1, dims="country")
        intercept = pm.Deterministic(
            "intercept",
            intercept_mu + intercept_sigma * intercept_offset,
            dims="country",
        )

        sigma = pm.HalfNormal("sigma", sigma=1)

        intercept_obs = intercept[country_idx_data]
        beta_obs = beta[country_idx_data, :]

        channel_contribution = (X_data * beta_obs).sum(axis=1)
        mu = intercept_obs + channel_contribution

        pm.Normal(
            "streaming_revenue_obs",
            mu=mu,
            sigma=sigma,
            observed=y_normalized,
            dims="obs_id",
        )

        pm.Deterministic("roas", beta, dims=("country", "channel"))
        pm.Deterministic("roas_global", beta_global, dims="channel")
        pm.Deterministic("roas_region", beta_region, dims=("region", "channel"))

        pm.Data("y_mean", y_mean)
        pm.Data("y_std", y_std)

    return model


def sample_model(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
    **kwargs,
) -> "az.InferenceData":
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            **kwargs,
        )
    return trace


def sample_posterior_predictive(
    model: pm.Model,
    trace: "az.InferenceData",
    random_seed: int = 42,
) -> "az.InferenceData":
    with model:
        ppc = pm.sample_posterior_predictive(
            trace,
            random_seed=random_seed,
        )
    trace.extend(ppc)
    return trace
