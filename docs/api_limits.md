# API limits and client modules

The api-integration agent reads this file before touching any API code.
Fill in every value in angle brackets from each provider's official
documentation. If a value is truly unknown, write "unknown" instead of
guessing. Preflight flags any angle brackets left in this file.

## Client modules

The files that call an external API directly. The api-integration agent
is used for changes to these files.

| File | Provider |
|---|---|
| src/trading/alpaca_client.py | Alpaca |
| src/scraper/live_quotes.py | Alpaca |
| dashboard/pages/10_portfolio.py | Alpaca |
| dashboard/components/market_clock.py | Alpaca |
| src/scraper/price_scraper.py | yfinance |
| src/scraper/dividend_scraper.py | yfinance |
| src/news/scraper.py | yfinance |
| src/news/short_interest.py | yfinance |
| src/analytics/sector_analysis.py | yfinance |
| src/calendar/earnings.py | yfinance |
| src/options/options_data.py | yfinance |
| src/options/implied_vol.py | yfinance |
| src/indicators/fundamentals.py | yfinance |
| dashboard/pages/11_sector_analysis.py | yfinance |
| src/scraper/macro_scraper.py | FRED |
| src/calendar/economic.py | FRED |
| dashboard/app.py | FRED |

## Alpaca

- Rate limit: <requests per minute from Alpaca docs, and which plan you are on>
- Auth: API key and secret, read from env vars <ALPACA_KEY_NAME> and <ALPACA_SECRET_NAME>
- Key expiry: <does the key expire, and what error comes back if it is invalid>
- Notes: <paper vs live endpoint you use>

## FRED

- Rate limit: <requests per minute from FRED docs>
- Auth: API key read from env var <FRED_KEY_NAME>
- Key expiry: <expiry behavior>
- Notes: most series update daily or less often, so cache aggressively

## yfinance

- Rate limit: unknown. yfinance is an unofficial library that scrapes
  Yahoo Finance and has no documented limit.
- Auth: none
- Key expiry: not applicable
- Notes: treat as fragile. Cache historical data, add small delays between
  calls, and expect the response format to change without warning.
