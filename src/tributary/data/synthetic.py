"""
Synthetic data generation for VOLTA Music Group MMM validation.

Generates realistic music marketing data with known ground truth parameters,
allowing us to validate model recovery before applying to real data.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tributary.transforms.adstock import geometric_adstock
from tributary.transforms.saturation import hill_saturation


@dataclass
class TrueParameters:
    """
    Ground truth parameters for synthetic data generation.

    These are the 'true' values we're trying to recover with our models.
    By generating data from known parameters, we can validate our pipeline.

    Attributes
    ----------
    channel_names : list[str]
        Names of marketing channels.
    beta_mu : NDArray
        Global mean channel effectiveness (what hierarchical model should recover).
    alpha_mu : NDArray
        Global mean adstock decay rates.
    K_mu : NDArray
        Global mean saturation half-points.
    S : NDArray
        Saturation slope parameters (shared across countries).
    beta_sigma : NDArray
        Between-country variation in channel effectiveness.
    beta : NDArray
        Country-specific channel effectiveness (n_countries × n_channels).
    alpha : NDArray
        Country-specific adstock decay.
    K : NDArray
        Country-specific saturation points.
    intercepts : NDArray
        Country-specific baseline streaming revenue.
    noise_sigma : float
        Observation noise standard deviation.
    """

    channel_names: list[str]

    beta_mu: NDArray[np.floating]
    alpha_mu: NDArray[np.floating]
    K_mu: NDArray[np.floating]
    S: NDArray[np.floating]

    beta_sigma: NDArray[np.floating]
    alpha_sigma: NDArray[np.floating]
    K_sigma: NDArray[np.floating]

    beta: Optional[NDArray[np.floating]] = None
    alpha: Optional[NDArray[np.floating]] = None
    K: Optional[NDArray[np.floating]] = None
    intercepts: Optional[NDArray[np.floating]] = None

    noise_sigma: float = 0.1

    country_names: Optional[list[str]] = None


@dataclass
class SyntheticDataConfig:
    """
    Configuration for VOLTA synthetic data generation.

    This defines the structure of the synthetic dataset:
    - Which markets (countries)
    - Which channels
    - How much data per market
    - Spend patterns and gaps

    The defaults create the 'VOLTA Music Group' scenario with 8 European
    markets at varying data maturity levels.
    """

    countries: list[str] = field(
        default_factory=lambda: [
            "DE",  # Germany - mature market
            "FR",  # France - mature market
            "UK",  # United Kingdom - mature market
            "NL",  # Netherlands - growing
            "ES",  # Spain - growing
            "IT",  # Italy - growing
            "PL",  # Poland - NEW MARKET (sparse data!)
            "SE",  # Sweden - NEW MARKET (sparse data!)
        ]
    )

    channels: list[str] = field(
        default_factory=lambda: [
            "spotify_ads_spend",
            "meta_spend",
            "tiktok_spend",
            "youtube_spend",
            "radio_spend",
            "playlist_spend",
        ]
    )

    # Data length per market (weeks) - THE KEY ASYMMETRY
    # Mature markets have 2 years, new markets have only 6 months
    weeks_per_country: dict[str, int] = field(
        default_factory=lambda: {
            "DE": 104,  # 2 years - Germany is VOLTA's home market
            "UK": 104,  # 2 years - UK launched with Germany
            "FR": 78,  # 1.5 years
            "NL": 78,  # 1.5 years
            "ES": 52,  # 1 year
            "IT": 52,  # 1 year
            "PL": 26,  # 6 months - just launched! (SPARSE)
            "SE": 26,  # 6 months - just launched! (SPARSE)
        }
    )

    start_date: date = field(default_factory=lambda: date(2023, 1, 2))  # Monday

    mean_weekly_spend: dict[str, float] = field(
        default_factory=lambda: {
            "spotify_ads_spend": 25000,
            "meta_spend": 35000,
            "tiktok_spend": 20000,
            "youtube_spend": 30000,
            "radio_spend": 45000,
            "playlist_spend": 12000,
        }
    )

    spend_volatility: float = 0.35

    channel_gaps: dict[str, list[tuple[str, int, int]]] = field(
        default_factory=lambda: {
            "tiktok_spend": [
                ("PL", 0, 10),  # Poland: no TikTok first 10 weeks
                ("SE", 0, 8),  # Sweden: no TikTok first 8 weeks
            ],
            # Radio promo takes time to set up in new markets
            "radio_spend": [
                ("PL", 0, 16),  # Poland: no radio first 4 months
                ("SE", 0, 12),  # Sweden: no radio first 3 months
            ],
            # Budget cuts in Italy during summer
            "meta_spend": [
                ("IT", 26, 4),  # Italy: Meta paused for 4 weeks in summer
            ],
        }
    )

    market_size_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "DE": 1.5,  # Germany - largest market
            "UK": 1.4,  # UK - second largest
            "FR": 1.2,  # France
            "NL": 0.6,  # Netherlands - smaller
            "ES": 0.9,  # Spain
            "IT": 1.0,  # Italy
            "PL": 0.5,  # Poland - new, small
            "SE": 0.5,  # Sweden - new, small
        }
    )

    random_seed: int = 42


def generate_true_parameters(
    config: SyntheticDataConfig,
    random_seed: Optional[int] = None,
) -> TrueParameters:
    """
    Generate ground truth parameters for synthetic music marketing data.

    These parameters reflect realistic music industry dynamics:
    - Spotify and YouTube have moderate, consistent effects
    - TikTok has high variance (viral potential)
    - Radio has slower decay (longer carryover)
    - Playlist pitching has gradual, sustained effect

    Parameters
    ----------
    config : SyntheticDataConfig
        Data generation configuration.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    TrueParameters
        Complete set of true parameters for data generation.
    """
    rng = np.random.default_rng(random_seed or config.random_seed)

    n_countries = len(config.countries)
    n_channels = len(config.channels)
    beta_mu = np.array(
        [
            0.65,  # spotify_ads - moderate, consistent
            0.55,  # meta - good for awareness
            0.45,  # tiktok - lower average but high variance
            0.70,  # youtube - strong for music (video content)
            0.80,  # radio - still powerful in Europe
            0.35,  # playlist - smaller but efficient
        ]
    )

    alpha_mu = np.array(
        [
            0.50,  # spotify_ads - moderate decay
            0.55,  # meta - slightly longer (retargeting)
            0.35,  # tiktok - very fast decay (short attention)
            0.60,  # youtube - longer (video rewatches)
            0.70,  # radio - long carryover (repeated plays)
            0.65,  # playlist - sustained (stays on playlist)
        ]
    )

    K_mu = np.array(
        [
            0.45,  # spotify_ads
            0.50,  # meta
            0.30,  # tiktok - saturates fast
            0.50,  # youtube
            0.60,  # radio - harder to saturate
            0.55,  # playlist
        ]
    )

    S = np.array(
        [
            2.0,  # spotify_ads
            2.2,  # meta
            2.8,  # tiktok - steep saturation
            2.0,  # youtube
            1.5,  # radio - gradual
            1.8,  # playlist
        ]
    )

    beta_sigma = np.array(
        [
            0.12,  # spotify_ads - moderate variation
            0.15,  # meta - more variation
            0.20,  # tiktok - HIGH variation (some markets love it)
            0.12,  # youtube - moderate
            0.18,  # radio - varies by market culture
            0.10,  # playlist - relatively consistent
        ]
    )

    alpha_sigma = np.array(
        [
            0.08,  # spotify_ads
            0.08,  # meta
            0.10,  # tiktok
            0.07,  # youtube
            0.10,  # radio
            0.08,  # playlist
        ]
    )

    K_sigma = np.array(
        [
            0.08,
            0.10,
            0.12,
            0.08,
            0.10,
            0.08,
        ]
    )

    beta = np.zeros((n_countries, n_channels))
    alpha = np.zeros((n_countries, n_channels))
    K = np.zeros((n_countries, n_channels))

    for c in range(n_countries):
        # Draw country parameters from group distribution
        beta[c] = rng.normal(beta_mu, beta_sigma)
        alpha[c] = np.clip(rng.normal(alpha_mu, alpha_sigma), 0.05, 0.95)
        K[c] = np.clip(rng.normal(K_mu, K_sigma), 0.05, 0.95)

    market_sizes = np.array(
        [config.market_size_multipliers[c] for c in config.countries]
    )
    intercepts = rng.normal(10 + np.log(market_sizes), 0.3)

    return TrueParameters(
        channel_names=config.channels,
        beta_mu=beta_mu,
        alpha_mu=alpha_mu,
        K_mu=K_mu,
        S=S,
        beta_sigma=beta_sigma,
        alpha_sigma=alpha_sigma,
        K_sigma=K_sigma,
        beta=beta,
        alpha=alpha,
        K=K,
        intercepts=intercepts,
        noise_sigma=0.4,
        country_names=config.countries,
    )


def generate_spend_data(
    config: SyntheticDataConfig,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate realistic music marketing spend patterns.

    Includes:
    - Seasonality (Q4 boost for holiday playlists, summer festivals)
    - Market size scaling
    - Random weekly variation
    - Channel gaps/pauses
    - Release cycles (spend spikes around album drops)

    Parameters
    ----------
    config : SyntheticDataConfig
        Data generation configuration.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Spend data with date, country, and channel columns.
    """
    rng = np.random.default_rng(random_seed or config.random_seed)

    records = []

    for country in config.countries:
        n_weeks = config.weeks_per_country[country]
        market_scale = config.market_size_multipliers[country]

        max_weeks = max(config.weeks_per_country.values())
        weeks_offset = max_weeks - n_weeks
        country_start = config.start_date + timedelta(weeks=weeks_offset)

        for week in range(n_weeks):
            current_date = country_start + timedelta(weeks=week)
            week_of_year = current_date.isocalendar()[1]

            row = {
                "date": current_date,
                "country": country,
            }

            for channel in config.channels:
                base_spend = config.mean_weekly_spend[channel]

                # === Seasonality ===
                # Q4 boost (holiday playlists, end-of-year charts)
                q4_boost = 1.3 if current_date.month in [11, 12] else 1.0

                # Summer festival season (June-August)
                summer_boost = 1.15 if current_date.month in [6, 7, 8] else 1.0

                # January dip (post-holiday)
                jan_dip = 0.8 if current_date.month == 1 else 1.0

                seasonality = q4_boost * summer_boost * jan_dip

                # Occasional spend spikes (single/album drops)
                release_spike = 1.0
                if rng.random() < 0.05:  # 5% chance of release week
                    release_spike = rng.uniform(1.5, 2.5)

                noise = rng.lognormal(0, config.spend_volatility)

                spend = base_spend * market_scale * seasonality * release_spike * noise

                if channel in config.channel_gaps:
                    for gap_country, gap_start, gap_duration in config.channel_gaps[
                        channel
                    ]:
                        if (
                            country == gap_country
                            and gap_start <= week < gap_start + gap_duration
                        ):
                            spend = 0.0
                            break

                row[channel] = max(0, round(spend, 2))

            records.append(row)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    return df


def generate_revenue(
    df: pd.DataFrame,
    true_params: TrueParameters,
    config: SyntheticDataConfig,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate streaming revenue based on true parameters.

    Applies the full MMM data generating process:
    1. Normalize spend
    2. Apply adstock transformation
    3. Apply saturation transformation
    4. Compute channel contributions
    5. Add baseline + noise

    Parameters
    ----------
    df : pd.DataFrame
        Spend data (output of generate_spend_data).
    true_params : TrueParameters
        Ground truth parameters.
    config : SyntheticDataConfig
        Configuration.
    random_seed : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        Input df with streaming_revenue column added.
    """
    rng = np.random.default_rng(random_seed or config.random_seed)

    df = df.copy()
    df["streaming_revenue"] = 0.0

    for c_idx, country in enumerate(config.countries):
        mask = df["country"] == country
        country_df = df[mask]
        n_obs = len(country_df)

        if n_obs == 0:
            continue

        # Start with baseline revenue (exp of intercept)
        baseline = np.exp(true_params.intercepts[c_idx])
        revenue = np.full(n_obs, baseline)

        # Add channel contributions
        for ch_idx, channel in enumerate(config.channels):
            spend = country_df[channel].values

            if spend.max() == 0:
                continue

            spend_normalized = spend / (spend.max() + 1e-8)

            adstocked = geometric_adstock(
                spend_normalized,
                alpha=true_params.alpha[c_idx, ch_idx],
                l_max=8,
                normalize=True,
            )

            saturated = hill_saturation(
                adstocked,
                K=true_params.K[c_idx, ch_idx],
                S=true_params.S[ch_idx],
            )

            contribution = true_params.beta[c_idx, ch_idx] * saturated * baseline
            revenue = revenue + contribution

        noise = rng.normal(0, true_params.noise_sigma * baseline, size=n_obs)
        revenue = revenue + noise

        revenue = np.maximum(revenue, baseline * 0.1)

        df.loc[mask, "streaming_revenue"] = revenue.round(2)

    return df


def generate_synthetic_mmm_data(
    config: Optional[SyntheticDataConfig] = None,
    true_params: Optional[TrueParameters] = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, TrueParameters]:
    """
    Generate complete synthetic VOLTA music marketing dataset.

    This is the main entry point for creating validation data with
    known ground truth parameters.

    Parameters
    ----------
    config : SyntheticDataConfig, optional
        Configuration. Uses VOLTA defaults if None.
    true_params : TrueParameters, optional
        Pre-specified parameters. Generates new ones if None.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, TrueParameters]
        (data, true_parameters) for model validation.

    Examples
    --------
    >>> df, truth = generate_synthetic_mmm_data(random_seed=42)
    >>> print(df.shape)
    (572, 8)
    >>> print(df.columns.tolist())
    ['date', 'country', 'spotify_ads_spend', ..., 'streaming_revenue']
    >>> print(truth.beta_mu)  # True global channel effects
    array([0.65, 0.55, 0.45, 0.70, 0.80, 0.35])
    """
    if config is None:
        config = SyntheticDataConfig(random_seed=random_seed)

    if true_params is None:
        true_params = generate_true_parameters(config, random_seed)

    df = generate_spend_data(config, random_seed)

    df = generate_revenue(df, true_params, config, random_seed)

    col_order = ["date", "country", "streaming_revenue"] + config.channels
    df = df[col_order]

    return df, true_params


def save_synthetic_data(
    df: pd.DataFrame,
    true_params: TrueParameters,
    output_dir: str = "data/",
    filename: str = "volta_marketing.csv",
) -> None:
    """
    Save synthetic data and ground truth to files.

    Parameters
    ----------
    df : pd.DataFrame
        Marketing data.
    true_params : TrueParameters
        Ground truth parameters.
    output_dir : str
        Output directory path.
    filename : str
        CSV filename for marketing data.
    """
    import json
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Save data
    df.to_csv(output_path / filename, index=False)
    print(f"✅ Saved marketing data to {output_path / filename}")

    # Save ground truth
    truth = {
        "channel_names": true_params.channel_names,
        "country_names": true_params.country_names,
        "beta_mu": true_params.beta_mu.tolist(),
        "alpha_mu": true_params.alpha_mu.tolist(),
        "K_mu": true_params.K_mu.tolist(),
        "S": true_params.S.tolist(),
        "beta_sigma": true_params.beta_sigma.tolist(),
        "alpha_sigma": true_params.alpha_sigma.tolist(),
        "K_sigma": true_params.K_sigma.tolist(),
        "beta": true_params.beta.tolist(),
        "alpha": true_params.alpha.tolist(),
        "K": true_params.K.tolist(),
        "intercepts": true_params.intercepts.tolist(),
        "noise_sigma": true_params.noise_sigma,
    }

    with open(output_path / "ground_truth.json", "w") as f:
        json.dump(truth, f, indent=2)
    print(f"✅ Saved ground truth to {output_path / 'ground_truth.json'}")


def load_ground_truth(path: str = "data/ground_truth.json") -> TrueParameters:
    """
    Load ground truth parameters from JSON file.

    Parameters
    ----------
    path : str
        Path to ground_truth.json file.

    Returns
    -------
    TrueParameters
        Loaded ground truth parameters.
    """
    import json

    with open(path) as f:
        data = json.load(f)

    return TrueParameters(
        channel_names=data["channel_names"],
        country_names=data.get("country_names"),
        beta_mu=np.array(data["beta_mu"]),
        alpha_mu=np.array(data["alpha_mu"]),
        K_mu=np.array(data["K_mu"]),
        S=np.array(data["S"]),
        beta_sigma=np.array(data["beta_sigma"]),
        alpha_sigma=np.array(
            data.get("alpha_sigma", [0.1] * len(data["channel_names"]))
        ),
        K_sigma=np.array(data.get("K_sigma", [0.1] * len(data["channel_names"]))),
        beta=np.array(data["beta"]),
        alpha=np.array(data["alpha"]),
        K=np.array(data["K"]),
        intercepts=np.array(data["intercepts"]),
        noise_sigma=data["noise_sigma"],
    )


def summarize_dataset(df: pd.DataFrame) -> None:
    """
    Print a summary of the generated dataset.

    Parameters
    ----------
    df : pd.DataFrame
        VOLTA marketing data.
    """
    print("=" * 60)
    print("VOLTA MUSIC GROUP - SYNTHETIC DATA SUMMARY")
    print("=" * 60)

    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"📅 Date Range: {df['date'].min().date()} to {df['date'].max().date()}")

    print("\n🌍 Data per Market:")
    market_summary = (
        df.groupby("country")
        .agg(
            weeks=("date", "count"),
            avg_revenue=("streaming_revenue", "mean"),
            total_spend=("spotify_ads_spend", "sum"),  # Just one channel for brevity
        )
        .round(0)
    )
    print(market_summary.to_string())

    print("\n📢 Channel Spend Totals (€):")
    channel_cols = [c for c in df.columns if c.endswith("_spend")]
    for col in channel_cols:
        total = df[col].sum()
        print(f"   {col.replace('_spend', ''):20s}: €{total:,.0f}")

    print(
        "\n💰 Total Streaming Revenue: €{:,.0f}".format(df["streaming_revenue"].sum())
    )
    print("=" * 60)


if __name__ == "__main__":
    """Generate and save synthetic data when run as script."""
    print("Generating VOLTA Music Group synthetic data...")

    df, truth = generate_synthetic_mmm_data(random_seed=42)
    summarize_dataset(df)

    save_synthetic_data(df, truth)
    print("\n✅ Done! Data ready for modeling.")
