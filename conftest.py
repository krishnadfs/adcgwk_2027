"""
Shared fixtures and data builders for the customer_reliability_analysis test-suite.

Design notes
------------
* No test touches the real 1M-row workbook. Every dataset is a tiny, hand-checkable frame
  built by the fixtures below, so expected values can be verified with a calculator.
* Anything that writes to disk uses pytest's ``tmp_path`` - the project's real ``data/`` and
  ``output/`` folders are never read or modified.
* Matplotlib runs on the Agg backend (the script already forces this on import) and every
  figure is closed after each test, so plotting tests are headless-safe and leak-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Make the script importable no matter where pytest is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import customer_reliability_analysis as _cra  # noqa: E402  (import after sys.path tweak)

RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
               "Price", "Customer ID", "Country"]

RAW_DEFAULTS = {
    "Invoice": "500001", "StockCode": "SKU-1", "Description": "Widget", "Quantity": 1,
    "InvoiceDate": "2010-01-05 10:00:00", "Price": 10.0, "Customer ID": 1001.0,
    "Country": "United Kingdom",
}


# ----------------------------------------------------------------------
# Housekeeping
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_all_figures():
    """Guarantee no matplotlib figure survives a test, whatever the test did."""
    yield
    plt.close("all")


# ----------------------------------------------------------------------
# Raw / clean transaction builders
# ----------------------------------------------------------------------

@pytest.fixture
def make_raw():
    """Build a *raw* (pre-cleaning) frame shaped like ``_read_year_sheets`` output.

    Each row is a dict of overrides applied on top of RAW_DEFAULTS, so a test only states
    the fields it cares about.
    """
    def _make(rows):
        return pd.DataFrame([{**RAW_DEFAULTS, **row} for row in rows], columns=RAW_COLUMNS)
    return _make


@pytest.fixture
def make_clean():
    """Build a *cleaned* frame (what ``load_transactions`` returns) for downstream functions.

    Rows are tuples: (invoice, quantity, date, unit_price, customer_id[, country]).
    """
    def _make(rows):
        norm = [tuple(r) + ("United Kingdom",) if len(r) == 5 else tuple(r) for r in rows]
        df = pd.DataFrame(norm, columns=["Invoice", "Quantity", "InvoiceDate", "Price",
                                         "Customer ID", "Country"])
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed")
        df["Customer ID"] = df["Customer ID"].astype("int64")
        df["Transaction_Value"] = df["Quantity"] * df["Price"]
        df["Gross_Value"] = df["Transaction_Value"].clip(lower=0)
        return df
    return _make


@pytest.fixture
def load_raw(monkeypatch, tmp_path):
    """Run ``load_transactions`` against an in-memory raw frame (no Excel I/O).

    Replaces only the disk-reading layer, so every cleaning rule in ``load_transactions``
    itself still executes for real.
    """
    def _load(raw_df, **kwargs):
        src = tmp_path / "transactions.xlsx"
        src.touch()  # load_transactions checks path.exists() before reading
        monkeypatch.setattr(_cra, "_load_and_cache_raw", lambda path, cache_dir: raw_df.copy())
        return _cra.load_transactions(src, **kwargs)
    return _load


@pytest.fixture
def write_workbook():
    """Write a real .xlsx file: ``sheets`` maps sheet name -> list of rows (header first)."""
    from openpyxl import Workbook

    def _write(path, sheets):
        wb = Workbook()
        wb.remove(wb.active)
        for name, rows in sheets.items():
            ws = wb.create_sheet(title=name)
            for row in rows:
                ws.append(list(row))
        wb.save(path)
        return Path(path)
    return _write


# ----------------------------------------------------------------------
# Segmentation fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def planted_summary():
    """90 customers in three deliberately well-separated behavioural groups (30 each).

    high : big spenders, frequent, very recent, very reliable
    mid  : moderate on every dimension
    low  : tiny spend, rare, long-lapsed, unreliable

    Separation is large enough that K-Means (k=3) must recover the planted groups, which
    lets the tests assert on *meaning* rather than just on shapes.
    """
    rng = np.random.default_rng(0)
    spec = {
        "high": dict(value=(20000, 30000), invoices=(60, 90), avg=(300, 400),
                     recency=(5, 20), cv=(0.2, 0.4), rel=(85, 100)),
        "mid": dict(value=(1500, 2500), invoices=(10, 20), avg=(100, 150),
                    recency=(100, 140), cv=(0.8, 1.0), rel=(55, 70)),
        "low": dict(value=(50, 150), invoices=(1, 3), avg=(30, 60),
                    recency=(280, 360), cv=(1.5, 2.0), rel=(5, 30)),
    }
    frames, next_id = [], 1
    for group, s in spec.items():
        n = 30
        frames.append(pd.DataFrame({
            "Customer_ID": range(next_id, next_id + n),
            "Total_Transaction_Value": rng.uniform(*s["value"], n),
            "Unique_Invoice_Count": rng.integers(s["invoices"][0], s["invoices"][1] + 1, n),
            "Average_Invoice_Value": rng.uniform(*s["avg"], n),
            "Recency_Days": rng.integers(s["recency"][0], s["recency"][1] + 1, n),
            "Monthly_Value_CV": rng.uniform(*s["cv"], n),
            "Reliability_Score": rng.uniform(*s["rel"], n),
            "Planted_Group": group,
        }))
        next_id += n
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def segmented(planted_summary):
    """A fully segmented portfolio (k=3) with Risk_Level attached, as ``main`` builds it."""
    seg = _cra.assign_segments(planted_summary, n_clusters=3, random_state=42)
    seg.summary["Risk_Level"] = seg.summary["Reliability_Score"].apply(_cra.assign_risk_level)
    return seg


@pytest.fixture
def captured_figures(monkeypatch):
    """Record every figure the plot functions close, so tests can inspect titles/labels/bars
    even though the functions close their figures before returning."""
    figures = []
    real_close = plt.close

    def spy(fig=None):
        if fig is not None and not isinstance(fig, (str, int)):
            figures.append(fig)
        return real_close() if fig is None else real_close(fig)

    monkeypatch.setattr(_cra.plt, "close", spy)
    return figures