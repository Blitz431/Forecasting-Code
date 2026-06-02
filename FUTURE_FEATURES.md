# Future Features — Backlog

Ideas to build eventually. Not prioritized — just a running list.

---

## Visualizations

### Supply Chain / Company Relationship Map
- Interactive network/bubble graph showing buyer-supplier relationships between S&P 500 companies
- Directed arrows: arrow pointing FROM supplier TO buyer (e.g. TSMC → AAPL means AAPL buys from TSMC)
- Nodes sized by market cap, colored by sector
- Hover card shows company name, relationship type, and estimated revenue dependency %
- Start with a curated static JSON of ~80 known relationships for the tickers we have scraped
- Add a manual "Add relationship" form in the UI to grow the dataset over time
- Future: automate sourcing from SEC 10-K filings (companies mention major customers/suppliers)
- Tech: Plotly network graph + NetworkX for layout, fits existing dark theme

---

## UI

### Custom UI Template / Theme
- Apply a consistent visual template across all dashboard pages
- (Template to be provided by user)

---

## Data

- All tradeable iteam on the stock exchange

---

## Analysis

*(add ideas here)*

---

## Other

*(add ideas here)*
