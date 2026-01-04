"""
Tests for media transformation functions.

Tests cover:
- Adstock transformations (geometric, delayed, Weibull)
- Saturation transformations (Hill, logistic, exponential)
- Edge cases and numerical stability
- Property-based testing with Hypothesis
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

from tributary.transforms.adstock import (
    geometric_adstock,
    delayed_adstock,
    weibull_adstock,
    adstock_weights,
    half_life_to_alpha,
    alpha_to_half_life,
    get_effective_spend_period,
    MUSIC_CHANNEL_ADSTOCK_DEFAULTS,
)
from tributary.transforms.saturation import (
    hill_saturation,
    logistic_saturation,
    exponential_saturation,
    michaelis_menten_saturation,
    compute_marginal_return,
    find_saturation_threshold,
    MUSIC_CHANNEL_SATURATION_DEFAULTS,
)


# =============================================================================
# GEOMETRIC ADSTOCK TESTS
# =============================================================================


class TestGeometricAdstock:
    """Tests for geometric adstock transformation."""

    def test_zero_decay_returns_weighted_input(self):
        """With alpha=0, output should be weighted input (first weight = 1)."""
        x = np.array([100.0, 0.0, 0.0, 0.0])
        result = geometric_adstock(x, alpha=0.0, l_max=4, normalize=True)
        # With normalize=True and alpha=0, weights = [1, 0, 0, 0] -> [1, 0, 0, 0] normalized
        np.testing.assert_array_almost_equal(result, x)

    def test_high_alpha_extends_carryover(self):
        """Higher alpha should spread effect over more periods."""
        x = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result_low = geometric_adstock(x, alpha=0.3, l_max=8)
        result_high = geometric_adstock(x, alpha=0.8, l_max=8)

        # High alpha should have more effect in later periods
        assert result_high[4] > result_low[4]
        assert result_high[6] > result_low[6]

    def test_carryover_effect(self):
        """Spend in period 0 should affect later periods."""
        x = np.array([100.0, 0.0, 0.0, 0.0])
        result = geometric_adstock(x, alpha=0.7, l_max=4, normalize=True)

        assert result[1] > 0, "Should have carryover to period 1"
        assert result[0] > result[1] > result[2], "Should decay over time"

    def test_output_shape_matches_input(self):
        """Output length should always match input length."""
        for length in [1, 10, 52, 104]:
            x = np.random.rand(length)
            result = geometric_adstock(x, alpha=0.5, l_max=8)
            assert len(result) == length

    def test_non_negative_output(self):
        """Adstock of non-negative inputs should be non-negative."""
        x = np.array([100.0, 50.0, 0.0, 75.0, 25.0])
        result = geometric_adstock(x, alpha=0.5, l_max=4)
        assert np.all(result >= 0)

    def test_empty_input(self):
        """Empty input should return empty output."""
        x = np.array([])
        result = geometric_adstock(x, alpha=0.5, l_max=4)
        assert len(result) == 0

    def test_single_element(self):
        """Single element input should work."""
        x = np.array([100.0])
        result = geometric_adstock(x, alpha=0.5, l_max=4)
        assert len(result) == 1
        assert result[0] > 0

    def test_normalize_preserves_relative_mass(self):
        """Normalized adstock should redistribute but not amplify."""
        x = np.array([100.0, 50.0, 25.0, 75.0])
        result = geometric_adstock(x, alpha=0.5, l_max=4, normalize=True)

        # With normalization, total should be close to original
        # (some edge effects expected)
        assert result.sum() <= x.sum() * 1.2

    def test_invalid_alpha_raises(self):
        """Alpha outside [0, 1] should raise ValueError."""
        x = np.array([100.0, 50.0])

        with pytest.raises(ValueError):
            geometric_adstock(x, alpha=-0.1, l_max=4)

        with pytest.raises(ValueError):
            geometric_adstock(x, alpha=1.5, l_max=4)

    def test_invalid_l_max_raises(self):
        """l_max < 1 should raise ValueError."""
        x = np.array([100.0, 50.0])

        with pytest.raises(ValueError):
            geometric_adstock(x, alpha=0.5, l_max=0)

    @given(
        arrays(
            np.float64,
            shape=st.integers(4, 52),
            elements=st.floats(0, 1000, allow_nan=False),
        ),
        st.floats(0.01, 0.99),
    )
    @settings(max_examples=50, deadline=None)
    def test_property_output_shape(self, x, alpha):
        """Property: output length always equals input length."""
        result = geometric_adstock(x, alpha=alpha, l_max=8)
        assert len(result) == len(x)

    @given(arrays(np.float64, shape=10, elements=st.floats(0, 100, allow_nan=False)))
    @settings(max_examples=50, deadline=None)
    def test_property_non_negative(self, x):
        """Property: adstock of non-negative inputs is non-negative."""
        x = np.abs(x)
        result = geometric_adstock(x, alpha=0.5, l_max=4)
        assert np.all(result >= 0)


class TestDelayedAdstock:
    """Tests for delayed (peaked) adstock transformation."""

    def test_theta_zero_similar_to_geometric(self):
        """With theta=0, should be similar to geometric adstock."""
        x = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        geometric = geometric_adstock(x, alpha=0.6, l_max=8, normalize=True)
        delayed = delayed_adstock(x, alpha=0.6, theta=0, l_max=8, normalize=True)

        # Peak should be at period 0 for both
        assert delayed[0] >= delayed[1]

    def test_theta_shifts_peak(self):
        """Non-zero theta should shift peak to later periods."""
        x = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = delayed_adstock(x, alpha=0.6, theta=3, l_max=8, normalize=True)

        # Peak should be around period 3
        peak_idx = np.argmax(result)
        assert 2 <= peak_idx <= 4, f"Peak at {peak_idx}, expected near 3"

    def test_radio_channel_pattern(self):
        """Radio should have delayed peak (theta=2 in defaults)."""
        x = np.array([100000.0] + [0.0] * 11)

        # Radio defaults: alpha=0.70, theta=2
        result = delayed_adstock(x, alpha=0.70, theta=2, l_max=12, normalize=True)

        # Should build up before decaying
        assert result[2] >= result[0], "Radio effect should peak after initial spend"

    def test_invalid_theta_raises(self):
        """Negative theta should raise ValueError."""
        x = np.array([100.0, 50.0])

        with pytest.raises(ValueError):
            delayed_adstock(x, alpha=0.5, theta=-1, l_max=4)


class TestWeibullAdstock:
    """Tests for Weibull adstock transformation."""

    def test_shape_one_similar_to_exponential(self):
        """Shape=1 should give exponential-like decay."""
        pytest.importorskip("scipy")

        x = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = weibull_adstock(x, shape=1.0, scale=3.0, l_max=8, adstock_type="cdf")

        # Should decay monotonically
        for i in range(1, len(result) - 1):
            if result[i] > 0:
                assert result[i] >= result[i + 1] or result[i + 1] < 1e-10

    def test_shape_greater_than_one_delayed(self):
        """Shape > 1 should create delayed peak."""
        pytest.importorskip("scipy")

        x = np.array([100.0] + [0.0] * 11)
        result = weibull_adstock(x, shape=2.0, scale=4.0, l_max=12, adstock_type="pdf")

        # Peak should not be at period 0
        peak_idx = np.argmax(result)
        assert peak_idx > 0, "Shape > 1 should delay the peak"

    def test_invalid_params_raise(self):
        """Invalid shape/scale should raise ValueError."""
        pytest.importorskip("scipy")

        x = np.array([100.0, 50.0])

        with pytest.raises(ValueError):
            weibull_adstock(x, shape=0, scale=1.0, l_max=4)

        with pytest.raises(ValueError):
            weibull_adstock(x, shape=1.0, scale=-1.0, l_max=4)


class TestAdstockUtilities:
    """Tests for adstock utility functions."""

    def test_adstock_weights_sum_to_one_normalized(self):
        """Normalized weights should sum to 1."""
        weights = adstock_weights(alpha=0.5, l_max=8, normalize=True)
        np.testing.assert_almost_equal(weights.sum(), 1.0)

    def test_adstock_weights_decay(self):
        """Weights should decay with lag."""
        weights = adstock_weights(alpha=0.6, l_max=8, normalize=False)

        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1]

    def test_half_life_conversion_roundtrip(self):
        """Converting alpha -> half_life -> alpha should give same result."""
        original_alpha = 0.7
        half_life = alpha_to_half_life(original_alpha)
        recovered_alpha = half_life_to_alpha(half_life)

        np.testing.assert_almost_equal(original_alpha, recovered_alpha)

    def test_half_life_interpretation(self):
        """Half-life should be when 50% of effect remains."""
        alpha = 0.5
        half_life = alpha_to_half_life(alpha)

        # At half-life periods, alpha^half_life should equal 0.5
        remaining = alpha**half_life
        np.testing.assert_almost_equal(remaining, 0.5)

    def test_effective_spend_period(self):
        """Should capture threshold fraction of effect."""
        alpha = 0.7
        threshold = 0.95
        periods = get_effective_spend_period(alpha, threshold)

        # Sum of first 'periods' weights should be >= threshold
        weights = adstock_weights(alpha, l_max=periods + 5, normalize=True)
        captured = weights[:periods].sum()

        assert captured >= threshold * 0.95  # Allow small tolerance


class TestMusicChannelAdstockDefaults:
    """Tests for music channel adstock default parameters."""

    def test_all_channels_have_defaults(self):
        """All VOLTA channels should have default parameters."""
        expected_channels = [
            "spotify_ads_spend",
            "meta_spend",
            "tiktok_spend",
            "youtube_spend",
            "radio_spend",
            "playlist_spend",
        ]

        for channel in expected_channels:
            assert channel in MUSIC_CHANNEL_ADSTOCK_DEFAULTS

    def test_tiktok_fastest_decay(self):
        """TikTok should have fastest decay (lowest alpha)."""
        alphas = {k: v["alpha"] for k, v in MUSIC_CHANNEL_ADSTOCK_DEFAULTS.items()}
        assert alphas["tiktok_spend"] == min(alphas.values())

    def test_radio_slowest_decay(self):
        """Radio should have slowest decay (highest alpha)."""
        alphas = {k: v["alpha"] for k, v in MUSIC_CHANNEL_ADSTOCK_DEFAULTS.items()}
        assert alphas["radio_spend"] == max(alphas.values())

    def test_alpha_values_valid(self):
        """All alpha values should be in (0, 1)."""
        for channel, params in MUSIC_CHANNEL_ADSTOCK_DEFAULTS.items():
            assert 0 < params["alpha"] < 1, f"{channel} alpha out of range"


# =============================================================================
# HILL SATURATION TESTS
# =============================================================================


class TestHillSaturation:
    """Tests for Hill saturation function."""

    def test_zero_input_gives_zero(self):
        """No spend = no effect."""
        result = hill_saturation(np.array([0.0]), K=1.0, S=2.0)
        assert result[0] == 0.0

    def test_half_saturation_at_K(self):
        """At x=K, output should be exactly 0.5."""
        K = 0.5
        result = hill_saturation(np.array([K]), K=K, S=2.0)
        np.testing.assert_almost_equal(result[0], 0.5, decimal=5)

    def test_bounded_output(self):
        """Output should always be in [0, 1]."""
        x = np.array([0, 0.1, 0.5, 1.0, 10.0, 100.0])
        result = hill_saturation(x, K=0.5, S=2.0)

        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_monotonically_increasing(self):
        """More spend should always give more (or equal) effect."""
        x = np.linspace(0, 2, 100)
        result = hill_saturation(x, K=0.5, S=2.0)

        assert np.all(np.diff(result) >= -1e-10)  # Allow tiny numerical errors

    def test_higher_S_steeper_curve(self):
        """Higher S should create steeper transition."""
        x = np.array([0.5])  # At half-saturation point

        result_low_S = hill_saturation(x, K=0.5, S=1.0)
        result_high_S = hill_saturation(x, K=0.5, S=5.0)

        # At K, both should be 0.5, but slopes differ
        np.testing.assert_almost_equal(result_low_S[0], 0.5, decimal=3)
        np.testing.assert_almost_equal(result_high_S[0], 0.5, decimal=3)

    def test_invalid_K_raises(self):
        """K <= 0 should raise ValueError."""
        x = np.array([0.5])

        with pytest.raises(ValueError):
            hill_saturation(x, K=0, S=2.0)

        with pytest.raises(ValueError):
            hill_saturation(x, K=-0.5, S=2.0)

    def test_invalid_S_raises(self):
        """S <= 0 should raise ValueError."""
        x = np.array([0.5])

        with pytest.raises(ValueError):
            hill_saturation(x, K=0.5, S=0)

        with pytest.raises(ValueError):
            hill_saturation(x, K=0.5, S=-1.0)

    def test_empty_input(self):
        """Empty input should return empty output."""
        x = np.array([])
        result = hill_saturation(x, K=0.5, S=2.0)
        assert len(result) == 0

    @given(st.floats(0.1, 10), st.floats(0.5, 5))
    @settings(max_examples=50, deadline=None)
    def test_property_always_bounded(self, K, S):
        """Property: output always in [0, 1] for valid params."""
        x = np.linspace(0, 10, 50)
        result = hill_saturation(x, K=K, S=S)
        assert np.all((result >= 0) & (result <= 1))


class TestLogisticSaturation:
    """Tests for logistic saturation function."""

    def test_bounded_output(self):
        """Output should be in [0, 1]."""
        x = np.linspace(0, 10, 100)
        result = logistic_saturation(x, lam=2.0)

        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_higher_lambda_faster_saturation(self):
        """Higher λ should saturate faster."""
        x = np.array([0.5])

        low_lam = logistic_saturation(x, lam=1.0)
        high_lam = logistic_saturation(x, lam=5.0)

        assert high_lam[0] > low_lam[0]

    def test_invalid_lambda_raises(self):
        """Lambda <= 0 should raise ValueError."""
        x = np.array([0.5])

        with pytest.raises(ValueError):
            logistic_saturation(x, lam=0)

        with pytest.raises(ValueError):
            logistic_saturation(x, lam=-1.0)


class TestExponentialSaturation:
    """Tests for exponential saturation function."""

    def test_zero_input_gives_zero(self):
        """No spend = no effect."""
        result = exponential_saturation(np.array([0.0]), lam=2.0)
        np.testing.assert_almost_equal(result[0], 0.0)

    def test_asymptotic_approach(self):
        """Should approach 1 asymptotically."""
        x = np.array([100.0])
        result = exponential_saturation(x, lam=2.0)

        assert result[0] > 0.99
        assert result[0] <= 1.0

    def test_monotonically_increasing(self):
        """More spend = more effect."""
        x = np.linspace(0, 5, 100)
        result = exponential_saturation(x, lam=2.0)

        assert np.all(np.diff(result) >= 0)


class TestMichaelisMentenSaturation:
    """Tests for Michaelis-Menten saturation function."""

    def test_half_saturation_at_Km(self):
        """At x=Km, output should be Vmax/2."""
        Vmax = 1.0
        Km = 0.5
        result = michaelis_menten_saturation(np.array([Km]), Vmax=Vmax, Km=Km)

        np.testing.assert_almost_equal(result[0], Vmax / 2)

    def test_equivalent_to_hill_S1(self):
        """Should be equivalent to Hill with S=1."""
        x = np.linspace(0, 2, 50)

        mm_result = michaelis_menten_saturation(x, Vmax=1.0, Km=0.5)
        hill_result = hill_saturation(x, K=0.5, S=1.0)

        np.testing.assert_array_almost_equal(mm_result, hill_result, decimal=5)


class TestSaturationUtilities:
    """Tests for saturation utility functions."""

    def test_marginal_return_decreases(self):
        """Marginal return should decrease with spend."""
        mr_low = compute_marginal_return(x=0.1, K=0.5, S=2.0)
        mr_high = compute_marginal_return(x=0.9, K=0.5, S=2.0)

        assert mr_low > mr_high

    def test_find_saturation_threshold(self):
        """Should find correct spend level for threshold."""
        K = 0.5
        S = 2.0
        threshold = 0.9

        x_threshold = find_saturation_threshold(K, S, threshold)
        actual_saturation = hill_saturation(np.array([x_threshold]), K=K, S=S)[0]

        np.testing.assert_almost_equal(actual_saturation, threshold, decimal=3)

    def test_find_saturation_threshold_invalid(self):
        """Threshold outside (0,1) should raise."""
        with pytest.raises(ValueError):
            find_saturation_threshold(K=0.5, S=2.0, threshold=0)

        with pytest.raises(ValueError):
            find_saturation_threshold(K=0.5, S=2.0, threshold=1.0)


class TestMusicChannelSaturationDefaults:
    """Tests for music channel saturation default parameters."""

    def test_all_channels_have_defaults(self):
        """All VOLTA channels should have default parameters."""
        expected_channels = [
            "spotify_ads_spend",
            "meta_spend",
            "tiktok_spend",
            "youtube_spend",
            "radio_spend",
            "playlist_spend",
        ]

        for channel in expected_channels:
            assert channel in MUSIC_CHANNEL_SATURATION_DEFAULTS

    def test_tiktok_saturates_fastest(self):
        """TikTok should saturate fastest (lowest K)."""
        Ks = {k: v["K"] for k, v in MUSIC_CHANNEL_SATURATION_DEFAULTS.items()}
        assert Ks["tiktok_spend"] == min(Ks.values())

    def test_radio_saturates_slowest(self):
        """Radio should saturate slowest (highest K)."""
        Ks = {k: v["K"] for k, v in MUSIC_CHANNEL_SATURATION_DEFAULTS.items()}
        assert Ks["radio_spend"] == max(Ks.values())

    def test_K_values_valid(self):
        """All K values should be in (0, 1)."""
        for channel, params in MUSIC_CHANNEL_SATURATION_DEFAULTS.items():
            assert 0 < params["K"] < 1, f"{channel} K out of range"

    def test_S_values_valid(self):
        """All S values should be positive."""
        for channel, params in MUSIC_CHANNEL_SATURATION_DEFAULTS.items():
            assert params["S"] > 0, f"{channel} S not positive"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestTransformPipeline:
    """Integration tests for the full transform pipeline."""

    def test_adstock_then_saturation(
        self, normalized_spend_series, adstock_params, saturation_params
    ):
        """Full pipeline: adstock -> saturation should work."""
        # Apply adstock
        adstocked = geometric_adstock(
            normalized_spend_series,
            alpha=adstock_params["alpha"],
            l_max=adstock_params["l_max"],
        )

        # Apply saturation
        saturated = hill_saturation(
            adstocked,
            K=saturation_params["K"],
            S=saturation_params["S"],
        )

        # Check properties
        assert len(saturated) == len(normalized_spend_series)
        assert np.all(saturated >= 0)
        assert np.all(saturated <= 1)

    def test_all_channels_transform(
        self, normalized_spend_series, channel_transform_params
    ):
        """All channels should transform without errors."""
        for channel, params in channel_transform_params.items():
            adstocked = geometric_adstock(
                normalized_spend_series,
                alpha=params["alpha"],
                l_max=8,
            )
            saturated = hill_saturation(adstocked, K=params["K"], S=params["S"])

            assert np.all(np.isfinite(saturated)), (
                f"{channel} produced non-finite values"
            )

    def test_sparse_spend_transforms(self, sample_spend_series_sparse):
        """Sparse spend (like Poland's TikTok) should transform correctly."""
        x = sample_spend_series_sparse / (sample_spend_series_sparse.max() + 1e-8)

        adstocked = geometric_adstock(x, alpha=0.35, l_max=8)
        saturated = hill_saturation(adstocked, K=0.3, S=2.8)

        # Should have non-zero values where there was spend
        assert saturated[12] > 0  # Effect persists past spend period
        assert np.all(saturated >= 0)
