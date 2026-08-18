from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

"""
Purpose: Central Pydantic configuration — loads all app settings from .env / environment variables.

Connections:
  - Used by: every module calls get_settings() to access paths, API keys, and trading parameters
  - standalone — no project-internal imports

In:  .env file and environment variables (ALPACA_*, FRED_*, OLLAMA_*, SEARX_*, ML_MODEL_SECRET, etc.)
Out: typed Settings object with nested AlpacaSettings, FredSettings, AlertSettings, OllamaSettings,
     SearxSettings sub-configs, plus research_* paths/tunables consumed by src/research/*
"""


# Project root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"


_ENV_FILE = str(Path(__file__).parent.parent / ".env")


class AlpacaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ALPACA_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"


class FredSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FRED_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = ""


class AlertSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    discord_webhook_url: str = ""
    smtp_host: str = ""
    smtp_port: Optional[int] = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_email_to: str = ""

    @field_validator("smtp_port", mode="before")
    @classmethod
    def _coerce_smtp_port(cls, v):
        if v == "" or v is None:
            return 587
        return v


class OllamaSettings(BaseSettings):
    """Local Ollama LLM endpoint — used by the deep-research agent (src/research/)."""

    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = "http://localhost:11434"
    model: str = ""              # empty ⇒ auto-pick the first installed model
    request_timeout: int = 120   # seconds; LLM generations can be slow
    num_ctx: int = 8192          # context window passed as an Ollama option


class SearxSettings(BaseSettings):
    """Local SearXNG search endpoint — used by the deep-research agent (src/research/)."""

    model_config = SettingsConfigDict(
        env_prefix="SEARX_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = "http://localhost:8080"
    request_timeout: int = 20    # seconds


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
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    searx: SearxSettings = Field(default_factory=SearxSettings)

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

    # ML model integrity (HMAC-SHA256 signing of saved .pkl/.pt files)
    ml_model_secret: str = ""  # set via ML_MODEL_SECRET in .env

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
    high_conviction_score_threshold: float = 0.7
    broker_trailing_stop_pct: float = 5.0           # initial broker trailing-stop trail %
    broker_trailing_stop_tight_pct: float = 1.0     # tighter trail after horizon expires

    # News & Sentiment (Phase 5)
    news_articles_dir: Path = DATA_DIR / "news" / "articles"
    news_short_interest_dir: Path = DATA_DIR / "news" / "short_interest"
    news_max_articles: int = 50          # max articles fetched per ticker per run
    news_sentiment_window_days: int = 7  # rolling window for sentiment summary
    finbert_model: str = "ProsusAI/finbert"

    # Political & Insider Trading (Phase 6)
    political_congress_dir: Path = DATA_DIR / "political" / "congress"
    political_insider_dir: Path = DATA_DIR / "political" / "insider"
    quiver_base_url: str = "https://api.quiverquant.com/beta"
    congress_lookback_days: int = 90    # how far back to fetch congressional trades
    insider_lookback_days: int = 90     # how far back to fetch insider filings
    insider_max_filings: int = 40       # max Form 4 filings to parse per ticker
    edgar_rate_limit_delay: float = 0.15  # seconds between EDGAR requests (SEC: max 10/sec)

    # Options & Calendar (Phase 7)
    options_dir: Path = DATA_DIR / "options"
    calendar_dir: Path = DATA_DIR / "calendar"
    options_unusual_volume_multiplier: float = 2.0   # flag when volume > X * open_interest
    options_iv_spike_threshold: float = 0.20         # flag IV when > hist_vol + 20pp
    options_lookback_days: int = 30                  # hist vol window (trading days)
    earnings_lookback_days: int = 365 * 3            # years of beat/miss history
    earnings_upcoming_days: int = 30                 # how far ahead to flag earnings

    # Automated Options Trading (Phase 12)
    options_capital: float = 5000.0          # dedicated options budget in dollars
    options_max_dte: int = 45                # max days to expiry at entry
    options_min_dte: int = 30               # min days to expiry at entry
    options_exit_dte: int = 7               # close position at this DTE
    options_profit_target: float = 0.50     # close at 50% gain on premium paid
    options_stop_loss: float = 0.50         # close at 50% loss on premium paid
    options_min_flow_signal: float = 0.15   # minimum |flow_signal| to enter
    options_min_vol_signal: float = 0.20    # minimum |vol_signal| to enter

    # Trading — Phase 10 data paths
    portfolio_snapshots_dir: Path = DATA_DIR / "portfolio" / "snapshots"
    tax_lots_dir: Path = DATA_DIR / "tax_lots"
    trade_journal_dir: Path = DATA_DIR / "trade_journal"
    circuit_breaker_state_file: Path = DATA_DIR / "circuit_breaker_state.json"

    # Live quotes (Alpaca market data — Phase 13)
    live_quotes_dir: Path = DATA_DIR / "live_quotes"
    live_quotes_refresh_seconds: int = 300

    # Trading — Phase 10 rules
    kelly_criterion_enabled: bool = False
    kelly_fraction: float = 0.25          # fractional Kelly (safety factor)
    bear_regime_position_scale: float = 0.5  # reduce max_position_pct by this factor in Bear/High-Vol

    # Ranking
    top_n_picks: int = 20

    # Deep-research agent (Ollama + SearXNG supplier discovery — src/research/)
    research_dir: Path = DATA_DIR / "research"           # review-queue JSON per ticker
    research_cache_dir: Path = DATA_DIR / "research" / "cache"  # SHA-keyed LLM/search cache
    research_max_rounds: int = 6              # hard cap on agent rounds (Auto never exceeds this)
    research_results_per_query: int = 6       # SearXNG results pulled per query
    research_fetch_pages: int = 3             # top-K result pages fetched for full text per round
    research_request_delay: float = 1.0       # seconds between outbound search/LLM requests
    research_min_confidence: float = 0.5      # findings below this are flagged low-confidence in review

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
