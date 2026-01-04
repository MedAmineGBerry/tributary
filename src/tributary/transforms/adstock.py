from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray


MUSIC_CHANNEL_ADSTOCK_DEFAULTS: dict[str, dict[str, float]] = {
    "spotify_ads_spend": {
        "alpha": 0.50,
        "theta": 0,
        "l_max": 8,
        "description": "Moderate carryover.",
    },
    "meta_spend": {
        "alpha": 0.55,
        "theta": 1,
        "l_max": 8,
        "description": "Moderate carryover with slight delay.",
    },
    "tiktok_spend": {
        "alpha": 0.35,
        "theta": 0,
        "l_max": 6,
        "description": "Fast decay.",
    },
    "youtube_spend": {
        "alpha": 0.60,
        "theta": 1,
        "l_max": 10,
        "description": "Longer carryover.",
    },
    "radio_spend": {
        "alpha": 0.70,
        "theta": 2,
        "l_max": 12,
        "description": "Slow decay.",
    },
    "playlist_spend": {
        "alpha": 0.65,
        "theta": 3,
        "l_max": 12,
        "description": "Long tail.",
    },
}


def geometric_adstock(
    x: NDArray[np.floating],
    alpha: float,
    l_max: int = 8,
    normalize: bool = True,
) -> NDArray[np.floating]:
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if l_max < 1:
        raise ValueError(f"l_max must be >= 1, got {l_max}")
    if len(x) == 0:
        return x.copy()
    if alpha == 0:
        return x.copy()

    weights = np.array([alpha**i for i in range(l_max)], dtype=np.float64)
    if normalize:
        s = weights.sum()
        if s > 0:
            weights = weights / s

    convolved = np.convolve(x, weights, mode="full")[: len(x)]
    return convolved.astype(x.dtype)


def delayed_adstock(
    x: NDArray[np.floating],
    alpha: float,
    theta: int,
    l_max: int = 8,
    normalize: bool = True,
) -> NDArray[np.floating]:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if theta < 0:
        raise ValueError(f"theta must be >= 0, got {theta}")
    if l_max < 1:
        raise ValueError(f"l_max must be >= 1, got {l_max}")
    if len(x) == 0:
        return x.copy()

    weights = np.array(
        [alpha ** ((i - theta) ** 2) for i in range(l_max)], dtype=np.float64
    )
    if normalize:
        s = weights.sum()
        if s > 0:
            weights = weights / s

    convolved = np.convolve(x, weights, mode="full")[: len(x)]
    return convolved.astype(x.dtype)


def weibull_adstock(
    x: NDArray[np.floating],
    shape: float,
    scale: float,
    l_max: int = 8,
    adstock_type: Literal["cdf", "pdf"] = "cdf",
    normalize: bool = True,
) -> NDArray[np.floating]:
    from scipy.stats import weibull_min

    if shape <= 0:
        raise ValueError(f"shape must be > 0, got {shape}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if l_max < 1:
        raise ValueError(f"l_max must be >= 1, got {l_max}")
    if len(x) == 0:
        return x.copy()
    if adstock_type not in ("cdf", "pdf"):
        raise ValueError(f"adstock_type must be 'cdf' or 'pdf', got {adstock_type}")

    t = np.arange(l_max) + 1

    if adstock_type == "cdf":
        weights = 1 - weibull_min.cdf(t, c=shape, scale=scale)
    else:
        weights = weibull_min.pdf(t, c=shape, scale=scale)

    weights = np.maximum(weights, 0)

    if normalize:
        s = weights.sum()
        if s > 0:
            weights = weights / s

    convolved = np.convolve(x, weights, mode="full")[: len(x)]
    return convolved.astype(x.dtype)


def adstock_weights(
    alpha: float,
    l_max: int = 8,
    normalize: bool = True,
) -> NDArray[np.floating]:
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if l_max < 1:
        raise ValueError(f"l_max must be >= 1, got {l_max}")

    weights = np.array([alpha**i for i in range(l_max)], dtype=np.float64)

    if normalize:
        s = weights.sum()
        if s > 0:
            weights = weights / s

    return weights


def half_life_to_alpha(half_life: float) -> float:
    if half_life <= 0:
        raise ValueError(f"half_life must be > 0, got {half_life}")
    return 0.5 ** (1 / half_life)


def alpha_to_half_life(alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(np.log(0.5) / np.log(alpha))


def get_effective_spend_period(
    alpha: float,
    threshold: float = 0.95,
) -> int:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    n = np.ceil(np.log(1 - threshold) / np.log(alpha))
    return int(max(1, n))


def plot_adstock_decay(
    alphas: dict[str, float],
    l_max: int = 12,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Adstock Decay by Channel",
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    weeks = np.arange(l_max)

    for name, alpha in alphas.items():
        weights = adstock_weights(alpha, l_max=l_max, normalize=True)
        ax.plot(weeks, weights, marker="o", label=name, linewidth=2, markersize=6)

    ax.set_xlabel("Weeks After Spend")
    ax.set_ylabel("Effect Weight (normalized)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(weeks)

    plt.tight_layout()
    return fig


def apply_channel_adstock(
    df: "pd.DataFrame",
    channel_col: str,
    custom_params: Optional[dict] = None,
) -> NDArray[np.floating]:
    import pandas as pd

    if channel_col not in df.columns:
        raise KeyError(f"'{channel_col}' not found in df columns")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    x = df[channel_col].to_numpy()

    defaults = MUSIC_CHANNEL_ADSTOCK_DEFAULTS.get(
        channel_col, {"alpha": 0.5, "theta": 0, "l_max": 8}
    )
    params = {**defaults, **(custom_params or {})}

    alpha = float(params.get("alpha", 0.5))
    theta = int(params.get("theta", 0))
    l_max = int(params.get("l_max", 8))

    if theta > 0:
        return delayed_adstock(x, alpha=alpha, theta=theta, l_max=l_max, normalize=True)
    return geometric_adstock(x, alpha=alpha, l_max=l_max, normalize=True)


def compute_carryover_percentage(
    alpha: float,
    periods: int = 1,
) -> float:
    if periods < 0:
        raise ValueError(f"periods must be >= 0, got {periods}")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return float(alpha**periods)
