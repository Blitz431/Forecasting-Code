from __future__ import annotations

"""
Purpose: Checked-in curated supplier-buyer relationships — the seed set the supply chain map starts from.

Connections:
  - src/analytics/supply_chain.py: SupplyChainGraph.load() falls back to this when
    data/supply_chain.json does not exist yet

In:  nothing — a static constant
Out: SEED_RELATIONSHIPS, a list of (supplier, buyer, type, dependency_pct) tuples

Why this lives in git rather than in data/:
  The whole of data/ is gitignored, so a curated seed stored only as
  data/supply_chain.json would disappear on a fresh clone and the map would open
  empty. Keeping the seed here mirrors how analytics/sector_analysis.py keeps
  _STATIC_SECTORS in code with data/sector_cache.json as the runtime overlay.

Direction:      supplier -> buyer ("TSM -> AAPL" = AAPL buys from TSM).
dependency_pct: estimated % of the SUPPLIER's revenue coming from that buyer.
                Rough public estimates, NOT figures pulled from filings.
"""

# (supplier, buyer, relationship_type, dependency_pct)
SEED_RELATIONSHIPS: list[tuple[str, str, str, float | None]] = [
    ('ABBV', 'CAH', 'Distribution', 14.0),
    ('ALB', 'TSLA', 'Raw Materials', 15.0),
    ('AMAT', 'INTC', 'Components', 8.0),
    ('AMAT', 'MU', 'Components', 10.0),
    ('AMAT', 'TSM', 'Components', 17.0),
    ('AMZN', 'NFLX', 'Cloud Services', 2.0),
    ('ANET', 'META', 'Components', 21.0),
    ('ANET', 'MSFT', 'Components', 15.0),
    ('AVGO', 'AAPL', 'Semiconductors', 20.0),
    ('CAH', 'CVS', 'Distribution', 24.0),
    ('CIEN', 'AMZN', 'Components', 8.0),
    ('CL', 'WMT', 'Distribution', 12.0),
    ('COHR', 'AAPL', 'Components', 10.0),
    ('COR', 'CVS', 'Distribution', 18.0),
    ('CSCO', 'AMZN', 'Components', 4.0),
    ('CSX', 'NUE', 'Logistics', 3.0),
    ('DHR', 'MRK', 'Components', 4.0),
    ('DOW', 'F', 'Raw Materials', 3.0),
    ('FDX', 'AMZN', 'Logistics', 3.0),
    ('FIS', 'BAC', 'Software', 6.0),
    ('FLEX', 'AAPL', 'Components', 5.0),
    ('GD', 'LMT', 'Components', 2.0),
    ('GE', 'BA', 'Components', 12.0),
    ('GIS', 'TGT', 'Distribution', 8.0),
    ('GIS', 'WMT', 'Distribution', 21.0),
    ('GLW', 'AAPL', 'Components', 20.0),
    ('GPN', 'WFC', 'Payment Processing', 5.0),
    ('HON', 'BA', 'Components', 9.0),
    ('HON', 'LMT', 'Components', 3.0),
    ('HSY', 'WMT', 'Distribution', 25.0),
    ('HWM', 'BA', 'Components', 15.0),
    ('HWM', 'GE', 'Components', 8.0),
    ('INTC', 'DELL', 'Semiconductors', 19.0),
    ('INTC', 'HPE', 'Semiconductors', 6.0),
    ('INTC', 'HPQ', 'Semiconductors', 17.0),
    ('JBHT', 'WMT', 'Logistics', 12.0),
    ('JBL', 'AAPL', 'Components', 20.0),
    ('KHC', 'KR', 'Distribution', 9.0),
    ('KHC', 'WMT', 'Distribution', 22.0),
    ('KLAC', 'MU', 'Components', 9.0),
    ('KLAC', 'TSM', 'Components', 20.0),
    ('KMB', 'WMT', 'Distribution', 14.0),
    ('KO', 'COST', 'Distribution', 5.0),
    ('KO', 'WMT', 'Distribution', 13.0),
    ('LLY', 'MCK', 'Distribution', 12.0),
    ('LRCX', 'MU', 'Components', 12.0),
    ('LRCX', 'TSM', 'Components', 18.0),
    ('MA', 'C', 'Payment Processing', 5.0),
    ('MA', 'JPM', 'Payment Processing', 7.0),
    ('MCHP', 'F', 'Semiconductors', 4.0),
    ('MCK', 'CVS', 'Distribution', 20.0),
    ('MCK', 'UNH', 'Distribution', 10.0),
    ('MDLZ', 'WMT', 'Distribution', 15.0),
    ('MRK', 'COR', 'Distribution', 12.0),
    ('MSFT', 'DELL', 'Software', 8.0),
    ('MSFT', 'HPQ', 'Software', 6.0),
    ('MSFT', 'ORCL', 'Cloud Services', 1.0),
    ('MU', 'AAPL', 'Components', 9.0),
    ('MU', 'NVDA', 'Components', 12.0),
    ('NUE', 'GM', 'Raw Materials', 5.0),
    ('NVDA', 'AMZN', 'Semiconductors', 10.0),
    ('NVDA', 'DELL', 'Semiconductors', 5.0),
    ('NVDA', 'GOOGL', 'Semiconductors', 6.0),
    ('NVDA', 'META', 'Semiconductors', 13.0),
    ('NVDA', 'MSFT', 'Semiconductors', 15.0),
    ('NVDA', 'ORCL', 'Semiconductors', 5.0),
    ('NVDA', 'SMCI', 'Semiconductors', 4.0),
    ('NVDA', 'TSLA', 'Semiconductors', 2.0),
    ('NXPI', 'F', 'Semiconductors', 6.0),
    ('NXPI', 'TSLA', 'Semiconductors', 5.0),
    ('ODFL', 'WMT', 'Logistics', 4.0),
    ('ON', 'TSLA', 'Semiconductors', 10.0),
    ('ORCL', 'JPM', 'Software', 2.0),
    ('PEP', 'COST', 'Distribution', 6.0),
    ('PEP', 'WMT', 'Distribution', 16.0),
    ('PFE', 'MCK', 'Distribution', 10.0),
    ('PG', 'COST', 'Distribution', 8.0),
    ('PG', 'WMT', 'Distribution', 21.0),
    ('PPG', 'F', 'Raw Materials', 5.0),
    ('PPG', 'GM', 'Raw Materials', 6.0),
    ('QCOM', 'AAPL', 'Semiconductors', 20.0),
    ('RTX', 'BA', 'Components', 8.0),
    ('RTX', 'LMT', 'Components', 4.0),
    ('SMCI', 'META', 'Components', 20.0),
    ('SMCI', 'MSFT', 'Components', 12.0),
    ('STLD', 'F', 'Raw Materials', 4.0),
    ('STX', 'DELL', 'Components', 12.0),
    ('STX', 'HPQ', 'Components', 8.0),
    ('SWKS', 'AAPL', 'Semiconductors', 65.0),
    ('SYY', 'CMG', 'Distribution', 4.0),
    ('SYY', 'MCD', 'Distribution', 10.0),
    ('TDG', 'BA', 'Components', 10.0),
    ('TMO', 'PFE', 'Components', 5.0),
    ('TSM', 'AAPL', 'Semiconductors', 23.0),
    ('TSM', 'NVDA', 'Semiconductors', 11.0),
    ('TXN', 'AAPL', 'Semiconductors', 5.0),
    ('TXT', 'BA', 'Components', 3.0),
    ('UNP', 'DE', 'Logistics', 2.0),
    ('UPS', 'AMZN', 'Logistics', 11.0),
    ('V', 'BAC', 'Payment Processing', 6.0),
    ('V', 'JPM', 'Payment Processing', 8.0),
    ('V', 'WMT', 'Payment Processing', 4.0),
    ('WDC', 'DELL', 'Components', 10.0),
]
