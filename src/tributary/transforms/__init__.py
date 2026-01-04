"""
Media transformation functions for VOLTA Music Marketing Mix Models.

Marketing spend doesn't translate directly to streaming revenue. Two key
phenomena must be modeled:

1. **Adstock (Carryover)**: When someone sees a Spotify ad today, they might
   not stream the song until next week. The effect "carries over" in time.

   - TikTok: Fast decay (short attention span, α ≈ 0.35)
   - Radio: Slow decay (repeated plays build familiarity, α ≈ 0.70)

2. **Saturation (Diminishing Returns)**: The first €10K on Meta ads has more
   impact than the tenth €10K. Eventually, you've reached everyone who cares.

   - TikTok: Quick saturation (viral or nothing, K ≈ 0.30)
   - Radio: Gradual saturation (broad reach takes time, K ≈ 0.60)

These transforms are applied BEFORE modeling:

    raw_spend → adstock → saturation → model

All functions are pure (no side effects), tested, and designed to work
with both NumPy arrays and PyTensor tensors.
"""

from tributary.transforms.adstock import (
    geometric_adstock,
    delayed_adstock,
    weibull_adstock,
    adstock_weights,
    plot_adstock_decay,
    MUSIC_CHANNEL_ADSTOCK_DEFAULTS,
)
from tributary.transforms.saturation import (
    hill_saturation,
    logistic_saturation,
    exponential_saturation,
    michaelis_menten_saturation,
    plot_saturation_curve,
    MUSIC_CHANNEL_SATURATION_DEFAULTS,
)

__all__ = [
    # Adstock
    "geometric_adstock",
    "delayed_adstock",
    "weibull_adstock",
    "adstock_weights",
    "plot_adstock_decay",
    "MUSIC_CHANNEL_ADSTOCK_DEFAULTS",
    # Saturation
    "hill_saturation",
    "logistic_saturation",
    "exponential_saturation",
    "michaelis_menten_saturation",
    "plot_saturation_curve",
    "MUSIC_CHANNEL_SATURATION_DEFAULTS",
]
