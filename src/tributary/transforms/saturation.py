import numpy as np
from numpy.typing import NDArray
from typing import Optional


MUSIC_CHANNEL_SATURATION_DEFAULTS: dict[str, dict[str, float]] = {
    "spotify_ads_spend": {"K": 0.45, "S": 2.0, "description": "Moderate saturation."},
    "meta_spend": {"K": 0.50, "S": 2.2, "description": "Standard digital saturation."},
    "tiktok_spend": {"K": 0.30, "S": 2.8, "description": "Quick saturation."},
    "youtube_spend": {"K": 0.50, "S": 2.0, "description": "Moderate saturation."},
    "radio_spend": {"K": 0.60, "S": 1.5, "description": "Slow saturation."},
    "playlist_spend": {"K": 0.55, "S": 1.8, "description": "Moderate-slow saturation."},
}


def hill_saturation(
    x: NDArray[np.floating],
    K: float,
    S: float,
) -> NDArray[np.floating]:
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if S <= 0:
        raise ValueError(f"S must be > 0, got {S}")

    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.size == 0:
        return x_arr.copy()

    x_safe = np.maximum(x_arr, 0.0)

    K_S = K**S
    x_S = x_safe**S

    result = 1.0 - K_S / (K_S + x_S + 1e-10)
    return np.clip(result, 0.0, 1.0)


def logistic_saturation(
    x: NDArray[np.floating],
    lam: float,
    x0: float = 0.0,
) -> NDArray[np.floating]:
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}")

    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.size == 0:
        return x_arr.copy()

    z = np.clip(lam * (x_arr - x0), -100, 100)
    result = (1 - np.exp(-z)) / (1 + np.exp(-z))
    return (result + 1) / 2


def exponential_saturation(
    x: NDArray[np.floating],
    lam: float,
) -> NDArray[np.floating]:
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}")

    x_arr = np.asarray(x, dtype=np.float64)
    x_safe = np.maximum(x_arr, 0.0)
    return 1.0 - np.exp(-lam * x_safe)


def michaelis_menten_saturation(
    x: NDArray[np.floating],
    Vmax: float,
    Km: float,
) -> NDArray[np.floating]:
    if Vmax <= 0:
        raise ValueError(f"Vmax must be > 0, got {Vmax}")
    if Km <= 0:
        raise ValueError(f"Km must be > 0, got {Km}")

    x_arr = np.asarray(x, dtype=np.float64)
    x_safe = np.maximum(x_arr, 0.0)
    return Vmax * x_safe / (Km + x_safe + 1e-10)


def tanh_saturation(
    x: NDArray[np.floating],
    scale: float = 1.0,
) -> NDArray[np.floating]:
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    x_arr = np.asarray(x, dtype=np.float64)
    x_safe = np.maximum(x_arr, 0.0)
    return np.tanh(scale * x_safe)


def compute_marginal_return(
    x: float,
    K: float,
    S: float,
) -> float:
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if S <= 0:
        raise ValueError(f"S must be > 0, got {S}")

    if x <= 0:
        return float(S / K)

    K_S = K**S
    x_S = x**S

    numerator = S * K_S * (x ** (S - 1))
    denominator = (K_S + x_S) ** 2
    return float(numerator / (denominator + 1e-10))


def find_saturation_threshold(
    K: float,
    S: float,
    threshold: float = 0.9,
) -> float:
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if S <= 0:
        raise ValueError(f"S must be > 0, got {S}")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    return float(K * (threshold / (1 - threshold)) ** (1 / S))


def compute_efficiency_score(
    x: NDArray[np.floating],
    K: float,
    S: float,
) -> NDArray[np.floating]:
    return hill_saturation(x, K, S)


def plot_saturation_curve(
    saturation_params: dict[str, dict[str, float]],
    x_max: float = 1.0,
    n_points: int = 100,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Saturation Curves by Channel",
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    x = np.linspace(0, x_max, n_points)

    for name, params in saturation_params.items():
        y = hill_saturation(x, K=float(params["K"]), S=float(params["S"]))
        ax.plot(x, y, label=name, linewidth=2)

        K = float(params["K"])
        if 0 <= K <= x_max:
            ax.scatter([K], [0.5], marker="o", s=50, zorder=5)

    ax.set_xlabel("Normalized Spend")
    ax.set_ylabel("Effect (fraction of maximum)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


def apply_channel_saturation(
    x: NDArray[np.floating],
    channel_name: str,
    custom_params: Optional[dict[str, float]] = None,
) -> NDArray[np.floating]:
    defaults = MUSIC_CHANNEL_SATURATION_DEFAULTS.get(channel_name, {"K": 0.5, "S": 2.0})
    params = {**defaults, **(custom_params or {})}
    return hill_saturation(x, K=float(params["K"]), S=float(params["S"]))


def compare_saturation_at_spend(
    spend_levels: list[float],
    channels: Optional[list[str]] = None,
) -> "pd.DataFrame":
    import pandas as pd

    if channels is None:
        channels = list(MUSIC_CHANNEL_SATURATION_DEFAULTS.keys())

    data: dict[str, list[float]] = {}
    for channel in channels:
        params = MUSIC_CHANNEL_SATURATION_DEFAULTS.get(channel, {"K": 0.5, "S": 2.0})
        vals = [
            float(
                hill_saturation(
                    np.array([lvl], dtype=np.float64),
                    K=float(params["K"]),
                    S=float(params["S"]),
                )[0]
            )
            for lvl in spend_levels
        ]
        data[channel.replace("_spend", "")] = vals

    return pd.DataFrame(data, index=[f"x={lvl:.2f}" for lvl in spend_levels])
