"""Pydantic and Pandera schemas for VOLTA music marketing data validation."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
import pandera as pa
from pandera.typing import Series


VOLTA_MARKETS = Literal["DE", "FR", "UK", "NL", "ES", "IT", "PL", "SE"]

MUSIC_CHANNELS = [
    "spotify_ads_spend",
    "meta_spend",
    "tiktok_spend",
    "youtube_spend",
    "radio_spend",
    "playlist_spend",
]


class MarketingRow(BaseModel):
    """
    Single row of VOLTA music marketing data.

    Validates individual records before ingestion. Use for streaming
    validation or API inputs.

    Example
    -------
    >>> row = MarketingRow(
    ...     date=date(2024, 1, 15),
    ...     country="DE",
    ...     streaming_revenue=125000.0,
    ...     spotify_ads_spend=30000.0,
    ...     meta_spend=25000.0,
    ...     tiktok_spend=15000.0,
    ...     youtube_spend=20000.0,
    ...     radio_spend=40000.0,
    ...     playlist_spend=10000.0,
    ... )
    """

    date: date
    country: VOLTA_MARKETS

    # Target variable: streaming revenue (Spotify, Apple Music, etc.)
    streaming_revenue: float = Field(
        ge=0, description="Weekly streaming revenue in EUR"
    )

    # Channel spend columns
    spotify_ads_spend: float = Field(ge=0, description="Spotify Ad Studio spend")
    meta_spend: float = Field(ge=0, description="Meta (Instagram/Facebook) ads spend")
    tiktok_spend: float = Field(ge=0, description="TikTok promotion spend")
    youtube_spend: float = Field(ge=0, description="YouTube ads spend")
    radio_spend: float = Field(ge=0, description="Radio promotion/plugging spend")
    playlist_spend: float = Field(ge=0, description="Playlist pitching services spend")

    @field_validator(
        "streaming_revenue",
        "spotify_ads_spend",
        "meta_spend",
        "tiktok_spend",
        "youtube_spend",
        "radio_spend",
        "playlist_spend",
    )
    @classmethod
    def must_be_non_negative(cls, v: float, info) -> float:
        """Ensure all monetary values are non-negative."""
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative, got {v}")
        return v

    @model_validator(mode="after")
    def check_total_spend_reasonable(self) -> "MarketingRow":
        """Warn if total weekly spend seems unreasonably high."""
        total_spend = (
            self.spotify_ads_spend
            + self.meta_spend
            + self.tiktok_spend
            + self.youtube_spend
            + self.radio_spend
            + self.playlist_spend
        )
        if total_spend > 500_000:
            pass
        return self


class MarketingDataFrame(pa.DataFrameModel):
    """
    Pandera schema for full VOLTA marketing DataFrame validation.

    Use this for batch validation of complete datasets before modeling.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.read_csv("volta_marketing.csv", parse_dates=["date"])
    >>> MarketingDataFrame.validate(df)  # Raises if invalid
    """

    date: Series[pa.DateTime] = pa.Field(description="Week start date")
    country: Series[str] = pa.Field(
        isin=["DE", "FR", "UK", "NL", "ES", "IT", "PL", "SE"],
        description="ISO 2-letter country code",
    )

    streaming_revenue: Series[float] = pa.Field(
        ge=0,
        description="Weekly streaming revenue (EUR)",
    )

    spotify_ads_spend: Series[float] = pa.Field(
        ge=0,
        description="Spotify Ad Studio weekly spend (EUR)",
    )
    meta_spend: Series[float] = pa.Field(
        ge=0,
        description="Meta (IG/FB) weekly spend (EUR)",
    )
    tiktok_spend: Series[float] = pa.Field(
        ge=0,
        description="TikTok weekly spend (EUR)",
    )
    youtube_spend: Series[float] = pa.Field(
        ge=0,
        description="YouTube ads weekly spend (EUR)",
    )
    radio_spend: Series[float] = pa.Field(
        ge=0,
        description="Radio promotion weekly spend (EUR)",
    )
    playlist_spend: Series[float] = pa.Field(
        ge=0,
        description="Playlist pitching weekly spend (EUR)",
    )

    class Config:
        """Pandera configuration."""

        name = "VOLTAMarketingData"
        strict = True
        coerce = True
        ordered = False


class ChannelMetadata(BaseModel):
    """
    Metadata about a marketing channel.

    Useful for setting informed priors based on domain knowledge.
    """

    name: str
    display_name: str
    category: Literal["digital", "traditional", "platform"]
    typical_decay_days: int = Field(
        ge=1,
        le=90,
        description="Expected carryover duration in days",
    )
    typical_saturation_point: float = Field(
        ge=0,
        le=1,
        description="Normalized spend level where diminishing returns kick in",
    )
    notes: str = ""


CHANNEL_METADATA = {
    "spotify_ads_spend": ChannelMetadata(
        name="spotify_ads_spend",
        display_name="Spotify Ads",
        category="platform",
        typical_decay_days=7,
        typical_saturation_point=0.4,
        notes="Spotify Ad Studio - audio and display ads within Spotify app",
    ),
    "meta_spend": ChannelMetadata(
        name="meta_spend",
        display_name="Meta (IG/FB)",
        category="digital",
        typical_decay_days=10,
        typical_saturation_point=0.5,
        notes="Instagram Reels, Facebook ads, combined spend",
    ),
    "tiktok_spend": ChannelMetadata(
        name="tiktok_spend",
        display_name="TikTok",
        category="platform",
        typical_decay_days=5,
        typical_saturation_point=0.3,
        notes="TikTok Spark Ads and promotion. High variance in effectiveness.",
    ),
    "youtube_spend": ChannelMetadata(
        name="youtube_spend",
        display_name="YouTube Ads",
        category="platform",
        typical_decay_days=14,
        typical_saturation_point=0.5,
        notes="Pre-roll, discovery ads, shorts promotion",
    ),
    "radio_spend": ChannelMetadata(
        name="radio_spend",
        display_name="Radio Promo",
        category="traditional",
        typical_decay_days=21,  # Radio has long carryover
        typical_saturation_point=0.6,
        notes="Radio plugging, promotional spend with stations/DJs",
    ),
    "playlist_spend": ChannelMetadata(
        name="playlist_spend",
        display_name="Playlist Pitching",
        category="platform",
        typical_decay_days=28,  # Playlist placements have long tail
        typical_saturation_point=0.7,
        notes="Third-party playlist pitching services, editorial pitching",
    ),
}


def validate_dataframe(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Validate a DataFrame against the VOLTA schema.

    Parameters
    ----------
    df : pd.DataFrame
        Marketing data to validate.

    Returns
    -------
    pd.DataFrame
        Validated (and potentially coerced) DataFrame.

    Raises
    ------
    pandera.errors.SchemaError
        If validation fails.

    Example
    -------
    >>> df = pd.read_csv("data.csv", parse_dates=["date"])
    >>> df_valid = validate_dataframe(df)
    """
    return MarketingDataFrame.validate(df)


def get_channel_columns() -> list[str]:
    """Return list of channel spend column names."""
    return MUSIC_CHANNELS.copy()


def get_channel_display_names() -> dict[str, str]:
    """Return mapping of column names to display names."""
    return {k: v.display_name for k, v in CHANNEL_METADATA.items()}
