"""Central configuration for AutoStockAnalyzer."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"


class AlpacaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALPACA_")

    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"


class FredSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRED_")

    api_key: str = ""


class AlertSettings(BaseSettings):
    discord_webhook_url: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_email_to: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configs
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    fred: FredSettings = Field(default_factory=FredSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)

    # Trading
    trading_mode: str = "paper"  # "paper" or "live"

    # Data paths
    data_dir: Path = DATA_DIR
    raw_daily_dir: Path = DATA_DIR / "raw" / "daily"
    raw_quarterly_dir: Path = DATA_DIR / "raw" / "quarterly"
    raw_macro_dir: Path = DATA_DIR / "raw" / "macro"
    raw_dividends_dir: Path = DATA_DIR / "raw" / "dividends"
    processed_dir: Path = DATA_DIR / "processed"
    forecasts_dir: Path = DATA_DIR / "forecasts"

    # Ticker universe
    ticker_source: str = "sp500"  # "sp500" or "total_market"

    # Scraping
    backfill_start_year: int = 2015
    yfinance_batch_size: int = 50  # tickers per batch to avoid rate limits

    # ML train/test split
    ml_train_start: str = "2015-01-01"
    ml_train_end: str = "2019-12-31"
    ml_test_start: str = "2020-01-01"
    ml_test_end: str = "2026-12-31"

    # Forecasting
    forecast_horizons: int = 4  # quarters ahead
    holdout_periods: int = 8  # for AutoBest evaluation

    # Indicator thresholds
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    stochastic_overbought: float = 80.0
    stochastic_oversold: float = 20.0
    short_interest_high: float = 5.0  # days to cover threshold

    # Trading rules
    trailing_stop_pct: float = 5.0  # 5% trailing stop
    trailing_stop_days: int = 3  # activate after 3 profitable days
    max_position_pct: float = 5.0  # max 5% of portfolio per stock
    circuit_breaker_daily_pct: float = 10.0  # halt if portfolio drops 10% in a day
    circuit_breaker_single_stock_pct: float = 15.0  # force-sell at 15% loss

    # News & Sentiment (Phase 5)
    news_articles_dir: Path = DATA_DIR / "news" / "articles"
    news_short_interest_dir: Path = DATA_DIR / "news" / "short_interest"
    news_max_articles: int = 50          # max articles fetched per ticker per run
    news_sentiment_window_days: int = 7  # rolling window for sentiment summary
    finbert_model: str = "ProsusAI/finbert"

    # Ranking
    top_n_picks: int = 20

    # FRED macro series to track
    fred_series: list[str] = [
        "GDP",          # Gross Domestic Product
        "FEDFUNDS",     # Federal Funds Rate
        "CPIAUCSL",     # Consumer Price Index
        "UNRATE",       # Unemployment Rate
        "DGS10",        # 10-Year Treasury Rate
        "DGS2",         # 2-Year Treasury Rate
        "VIXCLS",       # VIX Volatility Index
        "T10Y2Y",       # 10Y-2Y Treasury Spread (yield curve)
        "UMCSENT",      # Consumer Sentiment
        "INDPRO",       # Industrial Production
    ]


def get_settings() -> Settings:
    """Load settings from environment/.env file."""
    return Settings()
