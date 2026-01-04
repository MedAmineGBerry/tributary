"""
Pytest configuration and shared fixtures for Tributary tests.

Provides reusable fixtures for:
- Random number generators
- Sample spend data
- Marketing DataFrames
- Fitted model traces (session-scoped for speed)
- Ground truth parameters
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta


# =============================================================================
# BASIC FIXTURES
# =============================================================================


@pytest.fixture
def random_seed() -> int:
    """Consistent random seed for reproducible tests."""
    return 42


@pytest.fixture
def rng(random_seed: int) -> np.random.Generator:
    """NumPy random generator."""
    return np.random.default_rng(random_seed)


# =============================================================================
# VOLTA SCENARIO FIXTURES
# =============================================================================


@pytest.fixture
def volta_countries() -> list[str]:
    """VOLTA's European markets."""
    return ["DE", "FR", "UK", "NL", "ES", "IT", "PL", "SE"]


@pytest.fixture
def volta_channels() -> list[str]:
    """VOLTA's music marketing channels."""
    return [
        "spotify_ads_spend",
        "meta_spend",
        "tiktok_spend",
        "youtube_spend",
        "radio_spend",
        "playlist_spend",
    ]


@pytest.fixture
def sparse_countries() -> list[str]:
    """Countries with sparse data (new markets)."""
    return ["PL", "SE"]


@pytest.fixture
def rich_countries() -> list[str]:
    """Countries with rich data (mature markets)."""
    return ["DE", "UK"]


# =============================================================================
# SPEND DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_spend_series(rng: np.random.Generator) -> np.ndarray:
    """Sample weekly spend data (52 weeks)."""
    base = rng.lognormal(mean=10, sigma=0.5, size=52)
    # Add some zeros (paused weeks)
    base[10:12] = 0
    base[30:32] = 0
    return base


@pytest.fixture
def sample_spend_series_short() -> np.ndarray:
    """Short spend series for edge case testing."""
    return np.array([100.0, 50.0, 0.0, 75.0, 25.0])


@pytest.fixture
def sample_spend_series_sparse() -> np.ndarray:
    """Very sparse spend (like Poland's TikTok)."""
    x = np.zeros(26)
    x[10:15] = [50000, 60000, 40000, 30000, 20000]  # Only 5 weeks of spend
    return x


@pytest.fixture
def normalized_spend_series(sample_spend_series: np.ndarray) -> np.ndarray:
    """Normalized spend series [0, 1]."""
    return sample_spend_series / (sample_spend_series.max() + 1e-8)


# =============================================================================
# DATAFRAME FIXTURES
# =============================================================================


@pytest.fixture
def sample_marketing_df(
    volta_channels: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sample VOLTA marketing DataFrame for testing.

    Creates a small dataset with 2 countries (DE, PL), different lengths.
    - DE: 52 weeks (rich)
    - PL: 26 weeks (sparse)
    """
    records = []
    start_date = date(2024, 1, 1)

    country_weeks = {"DE": 52, "PL": 26}

    for country, n_weeks in country_weeks.items():
        for week in range(n_weeks):
            row = {
                "date": start_date + timedelta(weeks=week),
                "country": country,
                "streaming_revenue": float(rng.lognormal(10, 0.3)),
            }
            for channel in volta_channels:
                row[channel] = float(rng.lognormal(8, 0.5))
            records.append(row)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def full_marketing_df(
    volta_countries: list[str],
    volta_channels: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Full VOLTA marketing DataFrame with all 8 countries.

    Reflects realistic data availability:
    - DE, UK: 104 weeks
    - FR, NL: 78 weeks
    - ES, IT: 52 weeks
    - PL, SE: 26 weeks
    """
    records = []
    start_date = date(2022, 1, 3)  # Monday

    weeks_per_country = {
        "DE": 104,
        "UK": 104,
        "FR": 78,
        "NL": 78,
        "ES": 52,
        "IT": 52,
        "PL": 26,
        "SE": 26,
    }

    max_weeks = max(weeks_per_country.values())

    for country in volta_countries:
        n_weeks = weeks_per_country[country]
        offset = max_weeks - n_weeks

        for week in range(n_weeks):
            row = {
                "date": start_date + timedelta(weeks=offset + week),
                "country": country,
                "streaming_revenue": float(rng.lognormal(10, 0.3)),
            }
            for channel in volta_channels:
                # Add some channel gaps for realism
                if channel == "tiktok_spend" and country == "PL" and week < 10:
                    row[channel] = 0.0
                elif channel == "radio_spend" and country == "SE" and week < 12:
                    row[channel] = 0.0
                else:
                    row[channel] = float(rng.lognormal(8, 0.5))
            records.append(row)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def synthetic_data_with_truth(random_seed: int):
    """
    Generate synthetic data with known ground truth.

    Returns (DataFrame, TrueParameters) tuple.
    """
    from tributary.data.synthetic import (
        generate_synthetic_mmm_data,
        SyntheticDataConfig,
    )

    config = SyntheticDataConfig(
        countries=["DE", "PL"],  # Just 2 countries for speed
        weeks_per_country={"DE": 52, "PL": 26},
        random_seed=random_seed,
    )

    return generate_synthetic_mmm_data(config, random_seed=random_seed)


# =============================================================================
# TRANSFORM PARAMETER FIXTURES
# =============================================================================


@pytest.fixture
def adstock_params() -> dict:
    """Standard adstock parameters for testing."""
    return {
        "alpha": 0.6,
        "l_max": 8,
        "normalize": True,
    }


@pytest.fixture
def saturation_params() -> dict:
    """Standard saturation parameters for testing."""
    return {
        "K": 0.5,
        "S": 2.0,
    }


@pytest.fixture
def channel_transform_params() -> dict[str, dict]:
    """Transform parameters for all channels."""
    return {
        "spotify_ads_spend": {"alpha": 0.50, "K": 0.45, "S": 2.0},
        "meta_spend": {"alpha": 0.55, "K": 0.50, "S": 2.2},
        "tiktok_spend": {"alpha": 0.35, "K": 0.30, "S": 2.8},
        "youtube_spend": {"alpha": 0.60, "K": 0.50, "S": 2.0},
        "radio_spend": {"alpha": 0.70, "K": 0.60, "S": 1.5},
        "playlist_spend": {"alpha": 0.65, "K": 0.55, "S": 1.8},
    }


# =============================================================================
# MODEL FIXTURES (Session-scoped for speed)
# =============================================================================


@pytest.fixture(scope="session")
def fitted_pooled_trace():
    """
    Pre-fitted pooled model trace.

    Session-scoped to avoid refitting for every test.
    Uses minimal sampling for speed.
    """
    pytest.importorskip("pymc")

    import pymc as pm

    from tributary.data.synthetic import (
        generate_synthetic_mmm_data,
        SyntheticDataConfig,
    )
    from tributary.models import build_pooled_mmm

    # Minimal config for fast tests
    config = SyntheticDataConfig(
        countries=["DE", "PL"],
        weeks_per_country={"DE": 30, "PL": 15},
        random_seed=42,
    )

    df, _ = generate_synthetic_mmm_data(config, random_seed=42)
    channel_cols = [c for c in df.columns if c.endswith("_spend")]

    model = build_pooled_mmm(df, channel_cols)

    with model:
        trace = pm.sample(
            draws=100,
            tune=100,
            chains=2,
            random_seed=42,
            return_inferencedata=True,
            progressbar=False,
        )

    return trace


@pytest.fixture(scope="session")
def fitted_unpooled_trace():
    """Pre-fitted unpooled model trace."""
    pytest.importorskip("pymc")

    import pymc as pm

    from tributary.data.synthetic import (
        generate_synthetic_mmm_data,
        SyntheticDataConfig,
    )
    from tributary.models import build_unpooled_mmm

    config = SyntheticDataConfig(
        countries=["DE", "PL"],
        weeks_per_country={"DE": 30, "PL": 15},
        random_seed=42,
    )

    df, _ = generate_synthetic_mmm_data(config, random_seed=42)
    channel_cols = [c for c in df.columns if c.endswith("_spend")]

    model = build_unpooled_mmm(df, channel_cols)

    with model:
        trace = pm.sample(
            draws=100,
            tune=100,
            chains=2,
            random_seed=42,
            return_inferencedata=True,
            progressbar=False,
        )

    return trace


@pytest.fixture(scope="session")
def fitted_hierarchical_trace():
    """Pre-fitted hierarchical model trace."""
    pytest.importorskip("pymc")

    import pymc as pm

    from tributary.data.synthetic import (
        generate_synthetic_mmm_data,
        SyntheticDataConfig,
    )
    from tributary.models import build_hierarchical_mmm

    config = SyntheticDataConfig(
        countries=["DE", "PL"],
        weeks_per_country={"DE": 30, "PL": 15},
        random_seed=42,
    )

    df, _ = generate_synthetic_mmm_data(config, random_seed=42)
    channel_cols = [c for c in df.columns if c.endswith("_spend")]

    model = build_hierarchical_mmm(df, channel_cols)

    with model:
        trace = pm.sample(
            draws=100,
            tune=100,
            chains=2,
            target_accept=0.9,
            random_seed=42,
            return_inferencedata=True,
            progressbar=False,
        )

    return trace


@pytest.fixture(scope="session")
def all_traces(fitted_pooled_trace, fitted_unpooled_trace, fitted_hierarchical_trace):
    """Dictionary of all fitted traces."""
    return {
        "pooled": fitted_pooled_trace,
        "unpooled": fitted_unpooled_trace,
        "hierarchical": fitted_hierarchical_trace,
    }


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring PyMC sampling"
    )
    config.addinivalue_line("markers", "pymc: marks tests requiring PyMC")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their requirements."""
    for item in items:
        # Mark tests that import pymc
        if "pymc" in item.nodeid or "trace" in item.nodeid.lower():
            item.add_marker(pytest.mark.pymc)

        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)


# =============================================================================
# HYPOTHESIS PROFILES
# =============================================================================

from hypothesis import settings, Verbosity

settings.register_profile("ci", max_examples=50, deadline=None)
settings.register_profile("dev", max_examples=10, deadline=None)
settings.register_profile(
    "debug", max_examples=5, verbosity=Verbosity.verbose, deadline=None
)
