"""Tests for src/scraper/storage.py — Parquet upsert storage layer.

Covers the fast metadata-only get_latest_date path (and its fallback),
upsert dedup/sort semantics, atomic write behavior, and concurrent-write
safety of the per-file locking.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from src.scraper.storage import (
    get_latest_date,
    get_latest_dates,
    get_ticker_filepath,
    load_dataframe,
    save_dataframe,
    upsert_dataframe,
)


def _ohlcv(dates: list[str], name: str = "Date") -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name=name)
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [float(i + 1) for i in range(n)],
            "High": [float(i + 2) for i in range(n)],
            "Low": [float(i) for i in range(n)],
            "Close": [float(i + 1.5) for i in range(n)],
            "Volume": [1000.0 * (i + 1) for i in range(n)],
        },
        index=idx,
    )


# ------------------------------------------------------------------ #
# get_latest_date
# ------------------------------------------------------------------ #

class TestGetLatestDate:
    def test_named_datetimeindex(self, tmp_path):
        p = tmp_path / "AAPL.parquet"
        df = _ohlcv(["2020-01-01", "2020-01-02", "2020-01-05"])
        df.to_parquet(p, engine="pyarrow")
        assert get_latest_date(p) == pd.Timestamp("2020-01-05")

    def test_missing_file_returns_none(self, tmp_path):
        p = tmp_path / "NOPE.parquet"
        assert get_latest_date(p) is None

    def test_unnamed_index_falls_back(self, tmp_path):
        # Simulate the __index_level_0__ case: index has no name, so pandas
        # metadata won't have a clean single named index column and the
        # fast row-group-statistics path must fall back to a real read.
        p = tmp_path / "UNNAMED.parquet"
        idx = pd.date_range("2020-01-01", periods=5, freq="D")  # no name
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        df.to_parquet(p, engine="pyarrow")

        # Confirm the fallback is actually exercised (metadata really is
        # __index_level_0__), not just correctness by accident.
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(p)
        index_cols = (pf.schema_arrow.pandas_metadata or {}).get("index_columns", [])
        assert index_cols == ["__index_level_0__"]

        assert get_latest_date(p) == pd.Timestamp("2020-01-05")

    def test_empty_dataframe_returns_none(self, tmp_path):
        p = tmp_path / "EMPTY.parquet"
        df = pd.DataFrame({"Close": pd.Series([], dtype=float)})
        df.index = pd.DatetimeIndex([], name="Date")
        df.to_parquet(p, engine="pyarrow")
        assert get_latest_date(p) is None


# ------------------------------------------------------------------ #
# get_latest_dates (batch)
# ------------------------------------------------------------------ #

class TestGetLatestDatesBatch:
    def test_mix_of_existing_and_missing(self, tmp_path):
        p1 = tmp_path / "AAA.parquet"
        p2 = tmp_path / "BBB.parquet"
        p3 = tmp_path / "CCC.parquet"  # never created

        _ohlcv(["2021-01-01", "2021-01-02"]).to_parquet(p1, engine="pyarrow")
        _ohlcv(["2022-06-01"]).to_parquet(p2, engine="pyarrow")

        filepaths = {"AAA": p1, "BBB": p2, "CCC": p3}
        result = get_latest_dates(filepaths, max_workers=4)

        assert set(result.keys()) == {"AAA", "BBB", "CCC"}
        assert result["AAA"] == pd.Timestamp("2021-01-02")
        assert result["BBB"] == pd.Timestamp("2022-06-01")
        assert result["CCC"] is None

    def test_empty_input(self):
        assert get_latest_dates({}) == {}


# ------------------------------------------------------------------ #
# upsert_dataframe
# ------------------------------------------------------------------ #

class TestUpsertDataframe:
    def test_merge_non_overlapping(self, tmp_path):
        p = tmp_path / "T.parquet"
        first = _ohlcv(["2020-01-01", "2020-01-02"])
        upsert_dataframe(first, p)

        second = _ohlcv(["2020-01-03", "2020-01-04"])
        merged = upsert_dataframe(second, p)

        assert len(merged) == 4
        assert list(merged.index) == sorted(merged.index)
        on_disk = load_dataframe(p)
        assert len(on_disk) == 4

    def test_overlapping_new_data_wins(self, tmp_path):
        """Dedup semantics: on overlapping index, the newly-upserted row
        must win (not the old stored one) — this is upsert, not insert."""
        p = tmp_path / "T.parquet"
        old = _ohlcv(["2020-01-01"])
        old["Close"] = [100.0]
        upsert_dataframe(old, p)

        new = _ohlcv(["2020-01-01"])
        new["Close"] = [999.0]
        merged = upsert_dataframe(new, p)

        assert len(merged) == 1
        assert merged["Close"].iloc[0] == 999.0

    def test_new_data_empty_returns_existing_unchanged(self, tmp_path):
        p = tmp_path / "T.parquet"
        first = _ohlcv(["2020-01-01", "2020-01-02"])
        upsert_dataframe(first, p)

        empty = pd.DataFrame()
        result = upsert_dataframe(empty, p)
        assert len(result) == 2

    def test_first_write_no_existing_file(self, tmp_path):
        p = tmp_path / "NEW.parquet"
        df = _ohlcv(["2020-01-01"])
        merged = upsert_dataframe(df, p)
        assert len(merged) == 1
        assert p.exists()

    def test_slow_path_fires_with_duplicates_and_unsorted_data(self, tmp_path):
        """The has_duplicates / is_monotonic_increasing checks are only an
        optimization to *skip* extra work when they don't apply — confirm
        that when merged data DOES have duplicates AND IS NOT sorted, the
        slow path (remove_duplicates + sort_index) still fires and produces
        a fully correct, sorted, deduplicated result."""
        p = tmp_path / "T.parquet"

        # Existing data out of order on disk (write directly, bypassing
        # upsert, so the pre-existing file itself is unsorted).
        existing = _ohlcv(["2020-01-05", "2020-01-01", "2020-01-03"])
        save_dataframe(existing, p)

        # New data overlaps one existing date (duplicate) and adds dates
        # both before and after the existing range, also out of order.
        new = _ohlcv(["2020-01-03", "2020-01-06", "2019-12-31"])
        new["Close"] = [-1.0, 6.0, 0.0]  # mark the overlapping row distinctly

        merged = upsert_dataframe(new, p)

        expected_index = pd.DatetimeIndex(
            ["2019-12-31", "2020-01-01", "2020-01-03", "2020-01-05", "2020-01-06"],
            name="Date",
        )
        assert list(merged.index) == list(expected_index)
        assert merged.index.is_monotonic_increasing
        assert not merged.index.has_duplicates
        # The overlapping 2020-01-03 row must reflect the NEW value (-1.0),
        # not the original existing value.
        assert merged.loc[pd.Timestamp("2020-01-03"), "Close"] == -1.0

        on_disk = load_dataframe(p)
        assert list(on_disk.index) == list(expected_index)
        assert on_disk.loc[pd.Timestamp("2020-01-03"), "Close"] == -1.0


# ------------------------------------------------------------------ #
# Atomic writes
# ------------------------------------------------------------------ #

class TestAtomicWrite:
    def test_no_tmp_file_left_on_success(self, tmp_path):
        p = tmp_path / "T.parquet"
        save_dataframe(_ohlcv(["2020-01-01"]), p)
        assert p.exists()
        tmp = p.with_name(p.name + ".tmp")
        assert not tmp.exists()

    def test_tmp_file_cleaned_up_on_replace_failure(self, tmp_path, monkeypatch):
        """Simulate a failure AFTER the tmp file is written but before/at
        os.replace — confirm the finally-block cleanup still removes the
        .tmp file and the exception propagates (doesn't corrupt the
        original file)."""
        p = tmp_path / "T.parquet"
        save_dataframe(_ohlcv(["2020-01-01"]), p)  # establish an original file

        import src.scraper.storage as storage_mod

        def _boom(*args, **kwargs):
            raise OSError("simulated os.replace failure")

        monkeypatch.setattr(storage_mod.os, "replace", _boom)

        with pytest.raises(OSError):
            save_dataframe(_ohlcv(["2020-01-02"]), p)

        tmp = p.with_name(p.name + ".tmp")
        assert not tmp.exists(), "tmp file must be cleaned up even when os.replace fails"
        # Original file must be untouched/still readable (replace never happened).
        on_disk = load_dataframe(p)
        assert list(on_disk.index) == [pd.Timestamp("2020-01-01")]

    def test_tmp_file_cleaned_up_on_serialize_failure(self, tmp_path, monkeypatch):
        """If to_parquet itself raises (e.g. unserializable data), the
        finally block must not blow up even though the tmp file was never
        created."""
        p = tmp_path / "T.parquet"

        import pandas as pd_mod

        def _boom(self, *args, **kwargs):
            raise ValueError("simulated serialize failure")

        monkeypatch.setattr(pd_mod.DataFrame, "to_parquet", _boom)

        with pytest.raises(ValueError):
            save_dataframe(_ohlcv(["2020-01-01"]), p)

        tmp = p.with_name(p.name + ".tmp")
        assert not tmp.exists()
        assert not p.exists()


# ------------------------------------------------------------------ #
# Concurrent-write safety
# ------------------------------------------------------------------ #

class TestConcurrentWrites:
    def test_concurrent_upserts_same_file_no_data_loss(self, tmp_path):
        """Several threads upsert non-overlapping rows into the SAME file
        concurrently. The per-file lock must fully serialize the
        read-modify-write cycle so no row is lost and the file never ends
        up corrupted/partially written."""
        p = tmp_path / "CONCURRENT.parquet"
        n_threads = 12
        errors: list[Exception] = []

        def _worker(i: int) -> None:
            try:
                # Small random-ish stagger to increase interleaving odds.
                time.sleep(0.001 * (i % 3))
                df = _ohlcv([f"2020-01-{i + 1:02d}"])
                df["Close"] = [float(i)]
                upsert_dataframe(df, p)
            except Exception as exc:  # pragma: no cover - surfaced via errors list
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"worker threads raised: {errors}"

        final = load_dataframe(p)
        assert len(final) == n_threads, (
            f"expected {n_threads} rows after concurrent upserts, got {len(final)} "
            "(indicates lost updates / race condition in per-file locking)"
        )
        assert final.index.is_monotonic_increasing
        assert not final.index.has_duplicates
        # Every thread's distinct Close value must be present exactly once.
        assert sorted(final["Close"].tolist()) == [float(i) for i in range(n_threads)]

    def test_concurrent_writes_never_leave_tmp_file(self, tmp_path):
        p = tmp_path / "CONCURRENT2.parquet"
        n_threads = 8

        def _worker(i: int) -> None:
            df = _ohlcv([f"2020-02-{i + 1:02d}"])
            upsert_dataframe(df, p)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        tmp = p.with_name(p.name + ".tmp")
        assert not tmp.exists()
        assert p.exists()


# ------------------------------------------------------------------ #
# get_ticker_filepath (sanity)
# ------------------------------------------------------------------ #

def test_get_ticker_filepath():
    from pathlib import Path as P
    p = get_ticker_filepath("AAPL", P("data/raw/daily"))
    assert str(p).replace("\\", "/") == "data/raw/daily/AAPL.parquet"
