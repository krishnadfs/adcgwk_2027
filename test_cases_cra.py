"""
Test-suite for ``customer_reliability_analysis.py``.

Organisation (one test class per unit of the script, in the script's own order)
------------------------------------------------------------------------------
    TestModuleContract            constants and documented design guarantees
    TestAnalysisConfig            validation + immutability of the config dataclass
    TestActivePeriodClassifier    workbook-compatible Active/Inactive logic (hand-computed golden data)
    TestReliabilityScoreCalculator  score maths and every band boundary
    TestRiskAndTrendHelpers       assign_risk_level / classify_trend boundaries
    TestDataQualityReport         report serialisation
    TestReadYearSheets            real .xlsx reading (sheet selection, headers)
    TestLoadAndCacheRaw           pickle cache: miss, hit, stale, directory creation
    TestLoadTransactions          every cleaning stage + every report counter
    TestBuildMonthlyAggregates    monthly bucketing, gap filling, growth columns
    TestComputeCustomerFeatures   RFM features: value, recency, CV, trend
    TestAnalyzeCustomer           per-customer pipeline (both layers)
    TestRunPortfolioAnalysis      portfolio pass + the unit-price vs real-value finding
    TestLabelClusters             profile-derived segment naming
    TestEvaluateKRange            k-selection diagnostics
    TestAssignSegments            K-Means segmentation + model persistence
    TestPlots                     every chart: valid PNG, correct content, no leaked figures
    TestPersistence               JSON/CSV report writers
    TestParseArgs                 CLI contract
    TestMain                      end-to-end runs through ``main``
    TestKnownLimitations          strict-xfail tests documenting defects found during review

Run:  pytest                      (everything)
      pytest -m "not integration" (fast unit tests only)
      pytest --cov=customer_reliability_analysis --cov-report=term-missing
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import customer_reliability_analysis as cra

integration = pytest.mark.integration


# ======================================================================
# Shared helpers
# ======================================================================

def build_classifier(months, **kwargs):
    """months: iterable of (period_label, monthly_price_sum, line_count)."""
    clf = cra.ActivePeriodClassifier(**kwargs)
    clf.load([{"period": p, "quantity": 1, "price": price, "invoices": n} for p, price, n in months])
    return clf.compute()


# Hand-computed golden scenario (threshold=100, window=4). Derivation, month by month:
#   avg_spent    : 100, 100, 73.75, 64.00, 59.0909, 55.00     (cumulative price / cumulative lines)
#   rel_incr     : 0,   0,  -26.25, -9.75, -4.9091, -4.0909   (change in running average)
#   spend_flag   : 1,   1,   0,     0,      0,       0        (price strictly > 100)
#   no_decline   : 1,   1,   1,     1,      1,       0        (last 4 rel_incr all negative only at month 6)
GOLDEN_MONTHS = [
    ("2010_1", 200, 2), ("2010_2", 300, 3), ("2010_3", 90, 3),
    ("2010_4", 50, 2), ("2010_5", 10, 1), ("2010_6", 10, 1),
]


# ======================================================================
# Module contract
# ======================================================================

class TestModuleContract:
    def test_cluster_features_exclude_redundant_workbook_columns(self):
        """The module docstring promises Active/Inactive/Total months are NOT clustered on."""
        assert {"Active_Months", "Inactive_Months", "Total_Months"}.isdisjoint(cra.CLUSTER_FEATURES)

    def test_cluster_features_are_unique(self):
        assert len(cra.CLUSTER_FEATURES) == len(set(cra.CLUSTER_FEATURES))

    def test_log_transformed_features_are_clustered_features(self):
        assert set(cra.LOG_TRANSFORM_FEATURES) <= set(cra.CLUSTER_FEATURES)

    def test_required_columns_cover_everything_the_pipeline_reads(self):
        assert cra.REQUIRED_COLUMNS == {"Invoice", "Quantity", "InvoiceDate", "Price",
                                        "Customer ID", "Country"}

    @pytest.mark.parametrize("name, expected", [
        ("Year 2009-2010", True),
        ("Year 2010-2011", True),
        ("year 2009-2010", False),          # case-sensitive
        ("Year 2009-10", False),            # needs two 4-digit years
        ("Year 2009-2010 copy", False),     # anchored at the end
        (" Year 2009-2010", False),         # anchored at the start
        ("Notes", False),
    ])
    def test_year_sheet_pattern(self, name, expected):
        assert bool(cra.YEAR_SHEET_PATTERN.match(name)) is expected

    def test_default_paths_are_anchored_to_the_script_location(self):
        assert cra.DEFAULT_DATA_PATH == cra.PROJECT_ROOT / "data" / "customer_transactions_sample_v3.xlsx"
        assert cra.DEFAULT_OUTPUT_DIR == cra.PROJECT_ROOT / "output"


# ======================================================================
# AnalysisConfig
# ======================================================================

class TestAnalysisConfig:
    def test_defaults_match_the_workbook_compatible_configuration(self):
        cfg = cra.AnalysisConfig()
        assert cfg.spend_threshold == 100.0
        assert cfg.max_decline_months == 4
        assert cfg.n_clusters == 4
        assert cfg.country is None
        assert cfg.exclude_cancellations is False   # workbook keeps cancellations
        assert cfg.drop_duplicates is False         # workbook keeps duplicates
        assert cfg.random_state == 42

    def test_config_is_immutable(self):
        cfg = cra.AnalysisConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.n_clusters = 9

    @pytest.mark.parametrize("kwargs, message", [
        ({"spend_threshold": -0.01}, "spend_threshold"),
        ({"max_decline_months": 0}, "max_decline_months"),
        ({"n_clusters": 0}, "n_clusters"),
    ])
    def test_invalid_values_are_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            cra.AnalysisConfig(**kwargs)

    def test_smallest_legal_values_are_accepted(self):
        cfg = cra.AnalysisConfig(spend_threshold=0, max_decline_months=1, n_clusters=1)
        assert (cfg.spend_threshold, cfg.max_decline_months, cfg.n_clusters) == (0, 1, 1)


# ======================================================================
# ActivePeriodClassifier  (workbook-compatible layer)
# ======================================================================

class TestActivePeriodClassifier:

    # ---- construction / loading ------------------------------------------------

    @pytest.mark.parametrize("kwargs, message", [
        ({"spend_threshold": -1}, "spend_threshold"),
        ({"max_decline_months": 0}, "max_decline_months"),
    ])
    def test_constructor_validates_parameters(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            cra.ActivePeriodClassifier(**kwargs)

    def test_load_and_compute_return_self_for_chaining(self):
        clf = cra.ActivePeriodClassifier()
        assert clf.load([{"period": "2010_1", "quantity": 1, "price": 1, "invoices": 1}]) is clf
        assert clf.compute() is clf

    def test_load_sorts_chronologically_not_lexicographically(self):
        """'2010_10' must come after '2010_2' - a plain string sort would get this wrong."""
        clf = cra.ActivePeriodClassifier().load([
            {"period": "2010_10", "quantity": 1, "price": 1, "invoices": 1},
            {"period": "2010_2", "quantity": 1, "price": 1, "invoices": 1},
            {"period": "2009_12", "quantity": 1, "price": 1, "invoices": 1},
        ])
        assert [r.period for r in clf.rows] == ["2009_12", "2010_2", "2010_10"]

    def test_load_coerces_numeric_types(self):
        clf = cra.ActivePeriodClassifier().load(
            [{"period": "2010_1", "quantity": "5", "price": "100.5", "invoices": "3"}])
        row = clf.rows[0]
        assert (row.quantity, row.price, row.invoices) == (5.0, 100.5, 3)
        assert isinstance(row.price, float) and isinstance(row.invoices, int)

    def test_load_replaces_previously_loaded_rows(self):
        clf = cra.ActivePeriodClassifier()
        clf.load([{"period": "2010_1", "quantity": 1, "price": 1, "invoices": 1}])
        clf.load([{"period": "2011_5", "quantity": 1, "price": 1, "invoices": 1}])
        assert [r.period for r in clf.rows] == ["2011_5"]

    @pytest.mark.parametrize("bad_label", ["2010-3", "2010_", "abc_def", "2010_3_1", "", "None"])
    def test_invalid_period_labels_raise_a_clear_error(self, bad_label):
        with pytest.raises(ValueError, match="Invalid period label"):
            cra.ActivePeriodClassifier().load(
                [{"period": bad_label, "quantity": 1, "price": 1, "invoices": 1}])

    def test_input_order_does_not_change_results(self):
        shuffled = list(reversed(GOLDEN_MONTHS))
        assert build_classifier(shuffled).to_records() == build_classifier(GOLDEN_MONTHS).to_records()

    # ---- golden-data verification of every derived column -----------------------

    def test_every_derived_column_matches_hand_calculation(self):
        df = build_classifier(GOLDEN_MONTHS).to_dataframe()
        assert df["period"].tolist() == [m[0] for m in GOLDEN_MONTHS]
        np.testing.assert_allclose(df["value_per_invoice"], [100, 100, 30, 25, 10, 10])
        np.testing.assert_allclose(df["till_date"], [200, 500, 590, 640, 650, 660])
        np.testing.assert_allclose(df["avg_spent"], [100, 100, 73.75, 64.0, 650 / 11, 55.0])
        np.testing.assert_allclose(
            df["rel_incr_increase"], [0, 0, -26.25, -9.75, 650 / 11 - 64.0, 55.0 - 650 / 11])
        assert df["spend_flag"].tolist() == [1, 1, 0, 0, 0, 0]
        assert df["no_decline_flag"].tolist() == [1, 1, 1, 1, 1, 0]
        assert df["active_period"].tolist() == ["Active", "Active"] + ["Inactive"] * 4

    def test_summary_matches_hand_calculation(self):
        assert build_classifier(GOLDEN_MONTHS).summary() == {
            "months": 6, "active_months": 2, "inactive_months": 4,
            "total_spend": 660.0, "final_avg_spent": 55.0,
        }

    def test_active_periods_are_listed_chronologically(self):
        assert build_classifier(GOLDEN_MONTHS).active_periods() == ["2010_1", "2010_2"]

    def test_to_records_exposes_every_monthrow_field(self):
        record = build_classifier(GOLDEN_MONTHS).to_records()[0]
        assert set(record) == {f.name for f in dataclasses.fields(cra.MonthRow)}

    def test_compute_is_idempotent(self):
        clf = build_classifier(GOLDEN_MONTHS)
        first = clf.to_records()
        clf.compute()
        assert clf.to_records() == first

    # ---- rule 1: spend flag -----------------------------------------------------

    @pytest.mark.parametrize("price, expected", [(99.99, "Inactive"), (100.0, "Inactive"),
                                                 (100.01, "Active")])
    def test_spend_threshold_is_strictly_greater_than(self, price, expected):
        clf = build_classifier([("2010_1", price, 1)], spend_threshold=100.0)
        assert clf.rows[0].active_period == expected

    def test_custom_spend_threshold_is_honoured(self):
        clf = build_classifier([("2010_1", 50, 1)], spend_threshold=10)
        assert clf.rows[0].spend_flag == 1

    # ---- rule 2: no-decline flag ------------------------------------------------

    @pytest.mark.parametrize("window, expected_flags", [
        # unit-price 1000,900,800,700,600 -> running avg 1000,950,900,850,800 -> rel_incr 0,-50,-50,-50,-50
        (4, ["Active"] * 4 + ["Inactive"]),   # first full window of 4 negatives ends at month 5
        (2, ["Active", "Active", "Inactive", "Inactive", "Inactive"]),
        (1, ["Active", "Inactive", "Inactive", "Inactive", "Inactive"]),
    ])
    def test_consecutive_decline_window_deactivates_otherwise_high_spend_months(self, window, expected_flags):
        months = [(f"2010_{i + 1}", 1000 - 100 * i, 1) for i in range(5)]
        clf = build_classifier(months, spend_threshold=100, max_decline_months=window)
        assert all(r.spend_flag == 1 for r in clf.rows)   # spend alone never disqualifies here
        assert [r.active_period for r in clf.rows] == expected_flags

    def test_short_history_is_never_penalised_for_decline(self):
        """Fewer months than the window -> no_decline_flag must be 1 by definition."""
        clf = build_classifier([("2010_1", 1000, 1), ("2010_2", 900, 1), ("2010_3", 800, 1)],
                               max_decline_months=4)
        assert [r.no_decline_flag for r in clf.rows] == [1, 1, 1]

    def test_first_month_has_zero_relative_increase(self):
        assert build_classifier([("2010_1", 500, 5)]).rows[0].rel_incr_increase == 0.0

    # ---- degenerate inputs --------------------------------------------------------

    def test_zero_line_month_does_not_divide_by_zero(self):
        row = build_classifier([("2010_1", 50, 0)]).rows[0]
        assert row.value_per_invoice == 0.0 and row.avg_spent == 0.0

    def test_empty_classifier_summary(self):
        clf = cra.ActivePeriodClassifier().compute()
        assert clf.summary() == {"months": 0, "active_months": 0, "inactive_months": 0,
                                 "total_spend": 0, "final_avg_spent": None}
        assert clf.active_periods() == []
        assert clf.to_dataframe().empty


# ======================================================================
# ReliabilityScoreCalculator
# ======================================================================

class _StubClassifier:
    """Duck-typed stand-in: ReliabilityScoreCalculator only ever calls .summary()."""
    def __init__(self, active, total):
        self._active, self._total = active, total

    def summary(self):
        return {"months": self._total, "active_months": self._active,
                "inactive_months": self._total - self._active}


class TestReliabilityScoreCalculator:
    @pytest.mark.parametrize("active, total, score, band", [
        (10, 10, 100.0, "Very Loyal"),
        (80, 100, 80.0, "Very Loyal"),          # lower edge of Very Loyal
        (79, 100, 79.0, "Likely to Stay"),
        (60, 100, 60.0, "Likely to Stay"),      # lower edge of Likely to Stay
        (59, 100, 59.0, "At Risk"),
        (40, 100, 40.0, "At Risk"),             # lower edge of At Risk
        (39, 100, 39.0, "Likely to Exit"),
        (0, 10, 0.0, "Likely to Exit"),
        (0, 0, 0.0, "Likely to Exit"),          # no history must not divide by zero
    ])
    def test_score_and_band_boundaries(self, active, total, score, band):
        result = cra.ReliabilityScoreCalculator().calculate(_StubClassifier(active, total))
        assert result["reliability_score"] == pytest.approx(score)
        assert result["reliability_band"] == band

    @pytest.mark.parametrize("active, total, expected", [(1, 3, 33.33), (2, 3, 66.67), (1, 7, 14.29)])
    def test_score_is_rounded_to_two_decimals(self, active, total, expected):
        result = cra.ReliabilityScoreCalculator().calculate(_StubClassifier(active, total))
        assert result["reliability_score"] == expected

    def test_counts_are_passed_through(self):
        result = cra.ReliabilityScoreCalculator().calculate(_StubClassifier(3, 8))
        assert (result["total_months"], result["active_months"], result["inactive_months"]) == (8, 3, 5)

    def test_works_with_a_real_classifier(self):
        result = cra.ReliabilityScoreCalculator().calculate(build_classifier(GOLDEN_MONTHS))
        assert result == {"total_months": 6, "active_months": 2, "inactive_months": 4,
                          "reliability_score": 33.33, "reliability_band": "Likely to Exit"}


# ======================================================================
# assign_risk_level / classify_trend
# ======================================================================

class TestRiskAndTrendHelpers:
    @pytest.mark.parametrize("score, expected", [
        (0, "High Risk"), (29.99, "High Risk"),
        (30, "Medium Risk"), (59.99, "Medium Risk"),
        (60, "Low Risk"), (100, "Low Risk"),
    ])
    def test_assign_risk_level_boundaries(self, score, expected):
        assert cra.assign_risk_level(score) == expected

    @pytest.mark.parametrize("trend, expected", [
        (None, "Insufficient History"),
        (float("nan"), "Insufficient History"),
        (np.nan, "Insufficient History"),
        (np.float64("nan"), "Insufficient History"),
        (5.0, "Increasing"), (1.1, "Increasing"),            # inclusive lower edge
        (1.0999, "Stable"), (1.0, "Stable"), (0.9001, "Stable"),
        (0.9, "Declining"), (0.0, "Declining"), (-2.0, "Declining"),   # inclusive upper edge
    ])
    def test_classify_trend_boundaries(self, trend, expected):
        assert cra.classify_trend(trend) == expected


# ======================================================================
# DataQualityReport
# ======================================================================

class TestDataQualityReport:
    def test_to_dict_is_json_serialisable_and_complete(self, load_raw, make_raw):
        _, report = load_raw(make_raw([{}]))
        as_dict = report.to_dict()
        assert set(as_dict) == {f.name for f in dataclasses.fields(cra.DataQualityReport)}
        assert json.loads(json.dumps(as_dict)) == as_dict


# ======================================================================
# _read_year_sheets   (real .xlsx files)
# ======================================================================

@integration
class TestReadYearSheets:
    HEADER = ["Invoice", "Quantity"]

    def test_concatenates_all_year_sheets_in_order_and_ignores_others(self, tmp_path, write_workbook):
        path = write_workbook(tmp_path / "book.xlsx", {
            "Year 2009-2010": [self.HEADER, ["A1", 1], ["A2", 2]],
            "Notes": [["not", "data"], ["ignore", "me"]],
            "Year 2010-2011": [self.HEADER, ["B1", 3]],
        })
        df = cra._read_year_sheets(path)
        assert df["Invoice"].tolist() == ["A1", "A2", "B1"]
        assert df.index.tolist() == [0, 1, 2]          # ignore_index=True

    def test_falls_back_to_the_active_sheet_when_no_year_sheet_exists(self, tmp_path, write_workbook):
        path = write_workbook(tmp_path / "book.xlsx", {"Sales": [self.HEADER, ["X1", 9]]})
        df = cra._read_year_sheets(path)
        assert df.to_dict("records") == [{"Invoice": "X1", "Quantity": 9}]

    def test_single_year_sheet_is_returned_unconcatenated(self, tmp_path, write_workbook):
        path = write_workbook(tmp_path / "book.xlsx", {"Year 2009-2010": [self.HEADER, ["A1", 1]]})
        assert len(cra._read_year_sheets(path)) == 1

    def test_blank_header_cells_get_positional_names(self, tmp_path, write_workbook):
        path = write_workbook(tmp_path / "book.xlsx", {"Year 2009-2010": [["A", None, "C"], [1, 2, 3]]})
        assert list(cra._read_year_sheets(path).columns) == ["A", "col_1", "C"]

    def test_header_only_sheet_yields_an_empty_frame_with_columns(self, tmp_path, write_workbook):
        path = write_workbook(tmp_path / "book.xlsx", {"Year 2009-2010": [self.HEADER]})
        df = cra._read_year_sheets(path)
        assert df.empty and list(df.columns) == self.HEADER


# ======================================================================
# _load_and_cache_raw
# ======================================================================

class TestLoadAndCacheRaw:
    @pytest.fixture
    def source_and_spy(self, tmp_path, monkeypatch):
        """A touched source file plus a spy replacing the slow Excel parser."""
        src = tmp_path / "book.xlsx"
        src.touch()
        frame = pd.DataFrame({"a": [1, 2, 3]})
        calls = []

        def fake_read(path):
            calls.append(path)
            return frame.copy()

        monkeypatch.setattr(cra, "_read_year_sheets", fake_read)
        return src, frame, calls

    def test_cache_miss_parses_source_and_writes_pickle(self, tmp_path, source_and_spy):
        src, frame, calls = source_and_spy
        out = cra._load_and_cache_raw(src, tmp_path)
        pd.testing.assert_frame_equal(out, frame)
        assert len(calls) == 1
        assert (tmp_path / "book.raw.pkl").exists()

    def test_fresh_cache_is_used_without_reparsing(self, tmp_path, source_and_spy):
        src, frame, calls = source_and_spy
        cra._load_and_cache_raw(src, tmp_path)
        again = cra._load_and_cache_raw(src, tmp_path)
        assert len(calls) == 1                       # second call never touched the parser
        pd.testing.assert_frame_equal(again, frame)

    def test_stale_cache_is_rebuilt_when_source_is_newer(self, tmp_path, source_and_spy):
        src, _, calls = source_and_spy
        cra._load_and_cache_raw(src, tmp_path)
        future = time.time() + 1000
        os.utime(src, (future, future))              # source edited after the cache was written
        cra._load_and_cache_raw(src, tmp_path)
        assert len(calls) == 2

    def test_missing_cache_directory_is_created(self, tmp_path, source_and_spy):
        src, _, _ = source_and_spy
        nested = tmp_path / "deep" / "er" / "cache"
        cra._load_and_cache_raw(src, nested)
        assert (nested / "book.raw.pkl").exists()


# ======================================================================
# load_transactions
# ======================================================================

class TestLoadTransactions:

    # ---- guard rails ------------------------------------------------------------

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cra.load_transactions(tmp_path / "does_not_exist.xlsx")

    def test_missing_required_columns_are_named_in_the_error(self, load_raw, make_raw):
        raw = make_raw([{}]).drop(columns=["Country", "Price"])
        with pytest.raises(ValueError, match=r"missing required column.*Country.*Price"):
            load_raw(raw)

    # ---- typing & derived columns -----------------------------------------------

    def test_output_types_and_derived_columns(self, load_raw, make_raw):
        df, _ = load_raw(make_raw([
            {"Invoice": "1", "Quantity": 3, "Price": 2.5, "Customer ID": 13078.0},
            {"Invoice": "2", "Quantity": -2, "Price": 4.0, "Customer ID": 13078.0},
        ]))
        assert str(df["Customer ID"].dtype) == "int64"          # float IDs from Excel -> int
        assert pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"])
        assert df["Transaction_Value"].tolist() == [7.5, -8.0]  # Quantity x unit Price
        assert df["Gross_Value"].tolist() == [7.5, 0.0]         # negatives clipped to 0

    def test_report_tracks_row_counts_ids_dates_and_countries(self, load_raw, make_raw):
        _, rep = load_raw(make_raw([
            {"Invoice": "1", "Customer ID": 1.0, "InvoiceDate": "2010-01-05 10:00", "Country": "United Kingdom"},
            {"Invoice": "2", "Customer ID": 2.0, "InvoiceDate": "2011-12-09 12:50", "Country": "France"},
            {"Invoice": "2", "Customer ID": 2.0, "InvoiceDate": "2011-12-09 12:50", "Country": "France",
             "StockCode": "SKU-2"},
        ]))
        assert rep.rows_loaded == 3 and rep.rows_after_cleaning == 3
        assert rep.unique_customers == 2
        assert rep.unique_invoices == 2          # two lines share invoice "2"
        assert rep.countries_seen == 2
        assert (rep.date_range_start, rep.date_range_end) == ("2010-01-05", "2011-12-09")

    # ---- always-applied cleaning ------------------------------------------------

    def test_each_cleaning_stage_is_removed_and_counted(self, load_raw, make_raw):
        blank = {col: None for col in cra.REQUIRED_COLUMNS | {"StockCode", "Description"}}
        header_row = {col: col for col in ("Invoice", "StockCode", "Description", "Quantity",
                                           "InvoiceDate", "Price", "Customer ID", "Country")}
        raw = make_raw([
            {"Invoice": "10001"},                                   # valid
            {"Invoice": "10002", "InvoiceDate": "2010-02-10 09:30:00"},  # valid
            blank,                                                  # fully blank row
            header_row,                                             # header repeated mid-sheet
            {"Invoice": "10003", "Customer ID": None},              # no customer
            {"Invoice": "10004", "Quantity": "abc", "Customer ID": 3.0},         # bad quantity
            {"Invoice": "10005", "InvoiceDate": "not-a-date", "Customer ID": 4.0},  # bad date
            {"Invoice": "10006", "Price": "free", "Customer ID": 5.0},           # bad price
        ])
        df, rep = load_raw(raw)
        assert rep.rows_loaded == 8
        assert rep.blank_rows_removed == 1
        assert rep.duplicate_header_rows_removed == 1
        assert rep.missing_customer_id_removed == 1
        assert rep.invalid_numeric_or_date_removed == 3
        assert rep.rows_after_cleaning == len(df) == 2
        assert df["Invoice"].tolist() == ["10001", "10002"]

    # ---- duplicates ---------------------------------------------------------------

    def test_duplicates_are_kept_by_default_to_match_the_workbook(self, load_raw, make_raw):
        raw = make_raw([{"Invoice": "1"}, {"Invoice": "1"}, {"Invoice": "2"}])
        df, rep = load_raw(raw)
        assert len(df) == 3
        assert rep.duplicates_dropped is False

    def test_duplicate_counter_counts_every_row_in_a_duplicate_group(self, load_raw, make_raw):
        """Characterisation: uses duplicated(keep=False), so the original row is counted too.
        Three identical rows therefore report 3 (not 2 extra copies)."""
        raw = make_raw([{"Invoice": "1"}] * 3 + [{"Invoice": "2"}])
        _, rep = load_raw(raw)
        assert rep.fully_duplicate_transaction_rows == 3

    def test_drop_duplicates_keeps_one_copy(self, load_raw, make_raw):
        raw = make_raw([{"Invoice": "1"}] * 3 + [{"Invoice": "2"}])
        df, rep = load_raw(raw, drop_duplicates=True)
        assert df["Invoice"].tolist() == ["1", "2"]
        assert rep.duplicates_dropped is True
        assert rep.fully_duplicate_transaction_rows == 3     # measured before dropping

    # ---- cancellations / returns --------------------------------------------------

    @pytest.fixture
    def cancellation_raw(self, make_raw):
        return make_raw([
            {"Invoice": "10001", "Quantity": 5, "Price": 2.0},      # normal sale
            {"Invoice": "C10002", "Quantity": -5, "Price": 2.0},    # cancellation
            {"Invoice": "10003", "Quantity": -1, "Price": 2.0},     # negative qty, not 'C'-prefixed
            {"Invoice": "10004", "Quantity": 1, "Price": 0.0},      # zero-price line
        ])

    def test_cancellations_and_returns_are_kept_by_default(self, load_raw, cancellation_raw):
        df, rep = load_raw(cancellation_raw)
        assert len(df) == 4
        assert rep.cancelled_invoice_rows == 1
        assert rep.negative_quantity_rows == 2
        assert rep.cancellations_excluded is False

    def test_exclude_cancellations_removes_c_invoices_negative_qty_and_zero_price(self, load_raw, cancellation_raw):
        df, rep = load_raw(cancellation_raw, exclude_cancellations=True)
        assert df["Invoice"].tolist() == ["10001"]
        assert rep.cancellations_excluded is True
        assert (rep.cancelled_invoice_rows, rep.negative_quantity_rows) == (1, 2)   # counted before removal
        assert rep.rows_after_cleaning == 1

    # ---- country filter -----------------------------------------------------------

    @pytest.fixture
    def multi_country_raw(self, make_raw):
        return make_raw([
            {"Invoice": "1", "Customer ID": 1.0, "Country": "United Kingdom"},
            {"Invoice": "2", "Customer ID": 2.0, "Country": "United Kingdom"},
            {"Invoice": "3", "Customer ID": 3.0, "Country": "France"},
            {"Invoice": "4", "Customer ID": 4.0, "Country": "Germany"},
        ])

    def test_country_filter_restricts_rows_but_reports_all_countries_seen(self, load_raw, multi_country_raw):
        df, rep = load_raw(multi_country_raw, country="France")
        assert df["Customer ID"].tolist() == [3]
        assert rep.countries_seen == 3              # measured before filtering
        assert rep.country_filter == "France"
        assert rep.unique_customers == 1            # measured after filtering

    def test_country_with_no_rows_yields_empty_frame_and_na_dates(self, load_raw, multi_country_raw):
        df, rep = load_raw(multi_country_raw, country="Atlantis")
        assert df.empty
        assert rep.rows_after_cleaning == 0 and rep.unique_customers == 0
        assert (rep.date_range_start, rep.date_range_end) == ("N/A", "N/A")

    # ---- real workbook, end to end --------------------------------------------------

    @integration
    def test_reads_a_real_two_sheet_workbook_and_caches_it_next_to_the_source(self, tmp_path, write_workbook):
        header = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
                  "Customer ID", "Country"]
        path = write_workbook(tmp_path / "transactions.xlsx", {
            "Year 2009-2010": [header,
                               ["489434", "A", "x", 12, datetime(2009, 12, 1, 7, 45), 6.95, 13085, "United Kingdom"],
                               ["489435", "B", "y", 2, datetime(2009, 12, 2, 9, 0), 3.5, 13085, "United Kingdom"]],
            "Year 2010-2011": [header,
                               ["C500000", "C", "z", -1, datetime(2010, 12, 5, 11, 0), 9.0, 12682, "France"]],
            "Notes": [["ignored"]],
        })
        df, rep = cra.load_transactions(path)
        assert len(df) == 3 and rep.rows_loaded == 3
        assert rep.cancelled_invoice_rows == 1
        assert (tmp_path / "transactions.raw.pkl").exists()


# ======================================================================
# build_monthly_aggregates
# ======================================================================

class TestBuildMonthlyAggregates:
    @pytest.fixture
    def customer(self, make_clean):
        return make_clean([
            ("A", 2, "2010-01-05", 10.0, 1),    # Jan: two lines on one invoice
            ("A", 3, "2010-01-20", 20.0, 1),
            ("B", 1, "2010-03-02", 40.0, 1),    # Mar: a sale and a return (no February at all)
            ("C", -1, "2010-03-15", 15.0, 1),
        ])

    def test_observed_months_are_aggregated_correctly(self, customer):
        m = cra.build_monthly_aggregates(customer)
        assert m["Period"].tolist() == [pd.Period("2010-01"), pd.Period("2010-03")]   # gap NOT filled
        jan, mar = m.iloc[0], m.iloc[1]
        assert (jan.quantity, jan.price, jan.invoices, jan.unique_invoices) == (5, 30.0, 2, 1)
        assert (jan.transaction_value, jan.gross_value) == (80.0, 80.0)
        assert (mar.quantity, mar.price, mar.invoices, mar.unique_invoices) == (0, 55.0, 2, 2)
        assert (mar.transaction_value, mar.gross_value) == (25.0, 40.0)   # return nets out, never grosses

    def test_period_labels_use_unpadded_months(self, customer):
        assert cra.build_monthly_aggregates(customer)["period_label"].tolist() == ["2010_1", "2010_3"]

    def test_growth_columns(self, customer):
        m = cra.build_monthly_aggregates(customer)
        assert np.isnan(m.loc[0, "Abs_Growth"]) and np.isnan(m.loc[0, "Pct_Growth"])
        assert m.loc[1, "Abs_Growth"] == pytest.approx(25.0)
        assert m.loc[1, "Pct_Growth"] == pytest.approx(25 / 30 * 100)

    def test_fill_gaps_inserts_zero_activity_months(self, customer):
        m = cra.build_monthly_aggregates(customer, fill_gaps=True)
        assert m["period_label"].tolist() == ["2010_1", "2010_2", "2010_3"]
        feb = m.iloc[1]
        assert (feb.quantity, feb.price, feb.invoices, feb.unique_invoices,
                feb.transaction_value, feb.gross_value) == (0, 0, 0, 0, 0, 0)
        assert m.loc[1, "Pct_Growth"] == pytest.approx(-100.0)

    def test_fill_gaps_extends_to_reference_end(self, customer):
        m = cra.build_monthly_aggregates(customer, fill_gaps=True, reference_end=pd.Period("2010-05", "M"))
        assert m["period_label"].tolist() == ["2010_1", "2010_2", "2010_3", "2010_4", "2010_5"]
        assert m["transaction_value"].tolist()[-2:] == [0, 0]

    def test_reference_end_is_ignored_without_fill_gaps(self, customer):
        m = cra.build_monthly_aggregates(customer, fill_gaps=False, reference_end=pd.Period("2010-12", "M"))
        assert len(m) == 2


# ======================================================================
# compute_customer_features
# ======================================================================

class TestComputeCustomerFeatures:
    END = pd.Period("2010-08", "M")

    @pytest.fixture
    def customer(self, make_clean):
        return make_clean([
            ("I1", 2, "2010-01-10 10:00", 10.0, 1),   # Jan: 20 + 5 = 25
            ("I1", 1, "2010-01-10 10:00", 5.0, 1),
            ("I2", 4, "2010-03-15", 25.0, 1),         # Mar: 100
            ("I3", 1, "2010-06-20 12:00", 50.0, 1),   # Jun: 50
        ])

    def test_full_feature_set_matches_hand_calculation(self, customer):
        f = cra.compute_customer_features(customer, self.END)
        monthly = np.array([25, 0, 100, 0, 0, 50, 0, 0], dtype=float)   # Jan..Aug, zero-filled
        assert f["Total_Transaction_Value"] == 175.0
        assert f["Unique_Invoice_Count"] == 3
        assert f["Average_Invoice_Value"] == pytest.approx(175 / 3, abs=0.01)
        assert f["Customer_Lifetime_Months"] == 8
        assert f["Purchase_Frequency"] == 0.375
        assert f["Monthly_Value_CV"] == pytest.approx(monthly.std() / monthly.mean(), abs=1e-4)
        assert f["Recency_Days"] == 72                       # 20 Jun 12:00 -> 31 Aug end-of-day
        assert f["First_Purchase_Date"] == "2010-01-10"
        assert f["Last_Purchase_Date"] == "2010-06-20"

    def test_spend_trend_compares_last_three_months_to_the_three_before(self, customer):
        f = cra.compute_customer_features(customer, self.END)
        # recent = mean(Jun 50, Jul 0, Aug 0) = 16.67 ; prior = mean(Mar 100, Apr 0, May 0) = 33.33
        assert f["Spend_Trend"] == 0.5
        assert f["Trend_Status"] == "Declining"

    def test_short_history_has_no_trend(self, make_clean):
        two_months = make_clean([("I1", 1, "2010-01-10", 50.0, 1), ("I2", 1, "2010-02-10", 50.0, 1)])
        f = cra.compute_customer_features(two_months, pd.Period("2010-03", "M"))
        assert f["Customer_Lifetime_Months"] == 3
        assert f["Spend_Trend"] is None
        assert f["Trend_Status"] == "Insufficient History"

    def test_zero_prior_spend_gives_no_trend_instead_of_dividing_by_zero(self, make_clean):
        df = make_clean([("I1", 1, "2010-01-10", 100.0, 1),
                         ("I2", 1, "2010-06-10", 100.0, 1),
                         ("I3", 1, "2010-07-10", 100.0, 1)])
        f = cra.compute_customer_features(df, pd.Period("2010-07", "M"))
        assert f["Customer_Lifetime_Months"] == 7          # Feb-Apr window is all zero
        assert f["Spend_Trend"] is None
        assert f["Trend_Status"] == "Insufficient History"

    @pytest.mark.parametrize("last_purchase, expected_days", [
        ("2010-12-31 12:00", 0),     # last day of the reference month -> 0 full days
        ("2010-12-01 00:00", 30),
        ("2010-11-30 00:00", 31),
    ])
    def test_recency_is_measured_to_the_end_of_the_reference_month(self, make_clean, last_purchase, expected_days):
        df = make_clean([("I1", 1, last_purchase, 10.0, 1)])
        f = cra.compute_customer_features(df, pd.Period("2010-12", "M"))
        assert f["Recency_Days"] == expected_days

    def test_volatility_uses_gross_value_so_returns_cannot_inflate_it(self, make_clean):
        """Jan: +1000 sale and -990 return (net 10). Feb: +1000. Net-based CV would be huge;
        gross-based CV is exactly 0 because both months have gross value 1000."""
        df = make_clean([("I1", 1, "2010-01-05", 1000.0, 1),
                         ("C1", -1, "2010-01-20", 990.0, 1),
                         ("I2", 1, "2010-02-05", 1000.0, 1)])
        f = cra.compute_customer_features(df, pd.Period("2010-02", "M"))
        assert f["Monthly_Value_CV"] == 0.0
        assert f["Total_Transaction_Value"] == 1010.0      # but true net value still reflects the return

    def test_customer_with_only_returns_is_handled(self, make_clean):
        df = make_clean([("C1", -2, "2010-02-01", 10.0, 1)])
        f = cra.compute_customer_features(df, pd.Period("2010-02", "M"))
        assert f["Total_Transaction_Value"] == -20.0
        assert f["Monthly_Value_CV"] == 0.0                # gross mean is 0 -> guarded, no NaN/inf
        assert f["Average_Invoice_Value"] == -20.0

    def test_single_purchase_customer(self, make_clean):
        df = make_clean([("I1", 1, "2010-02-01", 10.0, 1)])
        f = cra.compute_customer_features(df, pd.Period("2010-02", "M"))
        assert (f["Customer_Lifetime_Months"], f["Monthly_Value_CV"], f["Purchase_Frequency"]) == (1, 0.0, 1.0)


# ======================================================================
# analyze_customer / analyze_single_customer
# ======================================================================

class TestAnalyzeCustomer:
    END = pd.Period("2010-06", "M")

    @pytest.fixture
    def customer(self, make_clean):
        # Observed months: Jan(150) Feb(150) Jun(50).  Running avg: 150, 150, 116.67 -> Jun fails spend test.
        return make_clean([("I1", 1, "2010-01-10", 150.0, 7),
                           ("I2", 1, "2010-02-10", 150.0, 7),
                           ("I3", 1, "2010-06-10", 50.0, 7)])

    def test_empty_frame_returns_none(self, make_clean):
        empty = make_clean([("I1", 1, "2010-01-10", 150.0, 7)]).iloc[0:0]
        assert cra.analyze_customer(empty, 7, cra.AnalysisConfig(), self.END) is None

    def test_result_structure_and_internal_consistency(self, customer):
        r = cra.analyze_customer(customer, 7, cra.AnalysisConfig(), self.END)
        assert set(r) == {"Customer_ID", "Monthly_KPIs", "Summary", "Reliability", "Behavioral", "Details"}
        assert r["Customer_ID"] == 7
        assert r["Summary"]["months"] == r["Reliability"]["total_months"] == len(r["Details"]) == 3
        assert r["Reliability"]["active_months"] == 2
        assert r["Reliability"]["reliability_score"] == 66.67
        assert r["Reliability"]["reliability_band"] == "Likely to Stay"

    def test_workbook_layer_sees_observed_months_but_behavioral_layer_sees_the_gaps(self, customer):
        r = cra.analyze_customer(customer, 7, cra.AnalysisConfig(), self.END)
        assert len(r["Monthly_KPIs"]) == 3                       # Jan, Feb, Jun only
        assert r["Behavioral"]["Customer_Lifetime_Months"] == 6  # Jan..Jun inclusive

    def test_config_thresholds_flow_through(self, customer):
        strict = cra.AnalysisConfig(spend_threshold=1e9)
        r = cra.analyze_customer(customer, 7, strict, self.END)
        assert r["Reliability"]["active_months"] == 0
        assert r["Reliability"]["reliability_band"] == "Likely to Exit"

    def test_customer_id_is_cast_to_plain_int(self, customer):
        assert type(cra.analyze_customer(customer, np.int64(7), cra.AnalysisConfig(), self.END)["Customer_ID"]) is int

    def test_single_customer_wrapper_filters_by_id(self, make_clean):
        df = make_clean([("I1", 1, "2010-01-10", 150.0, 1), ("I2", 1, "2010-01-10", 999.0, 2)])
        r = cra.analyze_single_customer(df, 1, cra.AnalysisConfig(), self.END)
        assert r["Customer_ID"] == 1
        assert r["Summary"]["total_spend"] == 150.0               # customer 2's 999 is excluded

    def test_single_customer_wrapper_returns_none_for_unknown_id(self, make_clean):
        df = make_clean([("I1", 1, "2010-01-10", 150.0, 1)])
        assert cra.analyze_single_customer(df, 999, cra.AnalysisConfig(), self.END) is None


# ======================================================================
# run_portfolio_analysis
# ======================================================================

class TestRunPortfolioAnalysis:
    @pytest.fixture
    def portfolio(self, make_clean):
        return make_clean([
            ("I1", 10, "2010-03-05", 5.0, 1),    # customer 1: two lines, Qty x Price = 150, Sum(Price) = 10
            ("I1", 20, "2010-03-05", 5.0, 1),
            ("I2", 1, "2010-01-15", 100.0, 2),   # customer 2: lapsed since January
        ])

    def test_one_row_per_customer_sorted_by_id(self, portfolio):
        out = cra.run_portfolio_analysis(portfolio, cra.AnalysisConfig())
        assert out["Customer_ID"].tolist() == [1, 2]

    def test_output_contains_every_clustering_feature_and_workbook_metric(self, portfolio):
        out = cra.run_portfolio_analysis(portfolio, cra.AnalysisConfig())
        expected = set(cra.CLUSTER_FEATURES) | {
            "Total_Months", "Active_Months", "Inactive_Months", "Reliability_Band",
            "Total_Spend", "Final_Avg_Spent", "Spend_Trend", "Trend_Status", "Purchase_Frequency"}
        assert expected <= set(out.columns)

    def test_workbook_spend_is_unit_price_sum_while_value_is_quantity_times_price(self, portfolio):
        """The documented 'Price is per-unit' finding: the two money columns must differ."""
        row = cra.run_portfolio_analysis(portfolio, cra.AnalysisConfig()).set_index("Customer_ID").loc[1]
        assert row["Total_Spend"] == 10.0                  # 5 + 5   (workbook-compatible)
        assert row["Total_Transaction_Value"] == 150.0     # 10x5 + 20x5 (real money)

    def test_recency_is_measured_against_the_dataset_end_not_each_customers_own_end(self, portfolio):
        out = cra.run_portfolio_analysis(portfolio, cra.AnalysisConfig()).set_index("Customer_ID")
        assert out.loc[1, "Recency_Days"] == 26            # 5 Mar -> end of March
        assert out.loc[2, "Recency_Days"] == 75            # 15 Jan -> end of March
        assert out.loc[2, "Customer_Lifetime_Months"] == 3

    def test_config_is_applied_to_every_customer(self, portfolio):
        out = cra.run_portfolio_analysis(portfolio, cra.AnalysisConfig(spend_threshold=1e9))
        assert (out["Reliability_Score"] == 0).all()


# ======================================================================
# _label_clusters
# ======================================================================

class TestLabelClusters:
    @staticmethod
    def cluster_summary(values, reliability, recency):
        return pd.DataFrame({"Cluster": range(len(values)), "Customer_Count": 10,
                             "Avg_Value": values, "Avg_Reliability": reliability,
                             "Avg_Recency_Days": recency})

    def test_labels_combine_value_reliability_and_recency(self):
        cs = self.cluster_summary([250, 150, 50, 10], [90, 80, 20, 10], [10, 300, 20, 400])
        mapping = cra._label_clusters(cs, pd.Series(np.arange(1, 301)))   # value terciles ~ (100, 200)
        assert mapping == {
            0: "High-Value, Reliable, Recent",
            1: "Mid-Value, Reliable, Lapsed",
            2: "Low-Value, Unreliable, Recent",
            3: "Low-Value, Unreliable, Lapsed",
        }

    def test_value_tercile_edges_are_high_inclusive_and_low_exclusive(self):
        population = pd.Series(np.arange(0, 301, dtype=float))
        lo, hi = population.quantile([1 / 3, 2 / 3])
        cs = self.cluster_summary([hi, lo, lo - 0.01], [50, 50, 50], [100, 100, 100])
        mapping = cra._label_clusters(cs, population)
        assert mapping[0].startswith("High-Value")      # v >= hi
        assert mapping[1].startswith("Mid-Value")       # v == lo is NOT below lo
        assert mapping[2].startswith("Low-Value")       # v < lo

    def test_cluster_exactly_at_the_median_counts_as_reliable_and_recent(self):
        cs = self.cluster_summary([10, 20, 30], [10, 50, 90], [10, 50, 90])
        assert cra._label_clusters(cs, pd.Series([10, 20, 30]))[1].endswith("Reliable, Recent")

    def test_single_cluster_is_labelled_without_error(self):
        cs = self.cluster_summary([100], [50], [10])
        assert len(cra._label_clusters(cs, pd.Series([50, 100, 150]))) == 1


# ======================================================================
# evaluate_k_range
# ======================================================================

class TestEvaluateKRange:
    @pytest.fixture
    def three_blobs(self):
        rng = np.random.default_rng(1)
        centres = np.array([[0, 0], [10, 10], [-10, 10]])
        return np.vstack([c + rng.normal(0, 0.5, (30, 2)) for c in centres])

    def test_returns_one_row_per_k_with_all_diagnostics(self, three_blobs):
        out = cra.evaluate_k_range(three_blobs, range(2, 6))
        assert out["k"].tolist() == [2, 3, 4, 5]
        assert list(out.columns) == ["k", "inertia", "silhouette", "davies_bouldin", "calinski_harabasz"]
        assert out[["inertia", "silhouette", "davies_bouldin", "calinski_harabasz"]].notna().all().all()

    def test_k_equal_one_has_no_cluster_quality_metrics(self, three_blobs):
        row = cra.evaluate_k_range(three_blobs, range(1, 2)).iloc[0]
        assert row["inertia"] > 0
        assert np.isnan([row["silhouette"], row["davies_bouldin"], row["calinski_harabasz"]]).all()

    def test_inertia_never_increases_with_more_clusters(self, three_blobs):
        inertia = cra.evaluate_k_range(three_blobs, range(1, 7))["inertia"].to_numpy()
        assert np.all(np.diff(inertia) <= 1e-9)

    def test_diagnostics_agree_on_the_true_number_of_clusters(self, three_blobs):
        out = cra.evaluate_k_range(three_blobs, range(2, 7)).set_index("k")
        assert out["silhouette"].idxmax() == 3
        assert out["davies_bouldin"].idxmin() == 3
        assert out["calinski_harabasz"].idxmax() == 3

    def test_is_reproducible_for_a_fixed_random_state(self, three_blobs):
        a = cra.evaluate_k_range(three_blobs, range(2, 5), random_state=7)
        b = cra.evaluate_k_range(three_blobs, range(2, 5), random_state=7)
        pd.testing.assert_frame_equal(a, b)


# ======================================================================
# assign_segments
# ======================================================================

class TestAssignSegments:
    def test_rejects_more_clusters_than_customers(self, planted_summary):
        with pytest.raises(ValueError, match="at least 5 scored customers"):
            cra.assign_segments(planted_summary.head(4), n_clusters=5)

    def test_recovers_the_planted_groups_exactly(self, planted_summary):
        seg = cra.assign_segments(planted_summary, n_clusters=3)
        clusters_per_group = seg.summary.groupby("Planted_Group")["Cluster"].nunique()
        assert (clusters_per_group == 1).all()                         # no group split across clusters
        assert seg.summary.groupby("Planted_Group")["Cluster"].first().nunique() == 3   # no two groups merged

    def test_segment_labels_describe_each_planted_group(self, planted_summary):
        seg = cra.assign_segments(planted_summary, n_clusters=3)
        label_of = seg.summary.groupby("Planted_Group")["Customer_Segment"].first().to_dict()
        assert label_of == {
            "high": "High-Value, Reliable, Recent",
            "mid": "Mid-Value, Reliable, Recent",
            "low": "Low-Value, Unreliable, Lapsed",
        }

    def test_cluster_summary_is_consistent_with_the_customer_table(self, planted_summary):
        seg = cra.assign_segments(planted_summary, n_clusters=3)
        assert seg.cluster_summary["Customer_Count"].sum() == len(planted_summary)
        assert set(seg.cluster_summary["Customer_Segment"]) == set(seg.summary["Customer_Segment"])
        expected_avg = seg.summary.groupby("Cluster")["Total_Transaction_Value"].mean()
        np.testing.assert_allclose(seg.cluster_summary.set_index("Cluster")["Avg_Value"], expected_avg)

    def test_input_frame_is_not_mutated(self, planted_summary):
        before = planted_summary.copy(deep=True)
        cra.assign_segments(planted_summary, n_clusters=3)
        pd.testing.assert_frame_equal(planted_summary, before)

    def test_scaled_matrix_is_standardised(self, planted_summary):
        seg = cra.assign_segments(planted_summary, n_clusters=3)
        assert seg.X_scaled.shape == (len(planted_summary), len(cra.CLUSTER_FEATURES))
        np.testing.assert_allclose(seg.X_scaled.mean(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(seg.X_scaled.std(axis=0), 1, atol=1e-9)

    def test_same_random_state_gives_identical_assignments(self, planted_summary):
        a = cra.assign_segments(planted_summary, n_clusters=3, random_state=5).summary["Cluster"]
        b = cra.assign_segments(planted_summary, n_clusters=3, random_state=5).summary["Cluster"]
        assert a.tolist() == b.tolist()

    # ---- robustness to awkward real-world values ------------------------------------

    def test_nan_features_are_filled_not_propagated(self, planted_summary):
        planted_summary.loc[0, "Monthly_Value_CV"] = np.nan
        seg = cra.assign_segments(planted_summary, n_clusters=3)
        assert np.isfinite(seg.X_scaled).all()

    def test_net_negative_customer_value_is_clipped_before_the_log_transform(self, planted_summary):
        planted_summary.loc[0, "Total_Transaction_Value"] = -500.0     # a heavy returner
        seg = cra.assign_segments(planted_summary, n_clusters=3)
        assert np.isfinite(seg.X_scaled).all()                          # log1p(-500) would be NaN

    def test_a_constant_feature_does_not_break_scaling(self, planted_summary):
        planted_summary["Monthly_Value_CV"] = 0.0                      # e.g. every customer bought once
        seg = cra.assign_segments(planted_summary, n_clusters=3)
        col = cra.CLUSTER_FEATURES.index("Monthly_Value_CV")
        assert np.isfinite(seg.X_scaled).all() and (seg.X_scaled[:, col] == 0).all()

    def test_single_cluster_is_allowed_and_has_no_silhouette(self, planted_summary, tmp_path):
        seg = cra.assign_segments(planted_summary, n_clusters=1, output_dir=tmp_path)
        assert seg.summary["Cluster"].nunique() == 1
        assert json.loads((tmp_path / "model_metadata.json").read_text())["silhouette_score"] is None

    # ---- persistence ------------------------------------------------------------------

    def test_no_files_are_written_without_an_output_dir(self, planted_summary, tmp_path):
        cra.assign_segments(planted_summary, n_clusters=3)
        assert list(tmp_path.iterdir()) == []

    def test_metadata_documents_the_trained_model(self, planted_summary, tmp_path):
        cra.assign_segments(planted_summary, n_clusters=3, random_state=11, output_dir=tmp_path)
        meta = json.loads((tmp_path / "model_metadata.json").read_text())
        assert meta["model_type"] == "KMeans" and meta["scaler"] == "StandardScaler"
        assert meta["n_clusters"] == 3 and meta["random_state"] == 11 and meta["n_init"] == 20
        assert meta["n_customers"] == len(planted_summary)
        assert meta["features"] == cra.CLUSTER_FEATURES
        assert meta["log_transformed_features"] == cra.LOG_TRANSFORM_FEATURES
        assert -1 <= meta["silhouette_score"] <= 1
        assert datetime.fromisoformat(meta["trained_at"]).tzinfo is not None   # timezone-aware UTC

    def test_saved_model_reproduces_the_assignments_on_the_original_data(self, planted_summary, tmp_path):
        seg = cra.assign_segments(planted_summary, n_clusters=3, output_dir=tmp_path)
        bundle = joblib.load(tmp_path / "segmentation_model.joblib")
        assert set(bundle) == {"scaler", "kmeans", "features", "log_transform_features"}

        # Re-apply the saved preprocessing exactly as a downstream scorer would.
        X = planted_summary[bundle["features"]].fillna(0).copy()
        for col in bundle["log_transform_features"]:
            X[col] = np.log1p(X[col].clip(lower=0))
        labels = bundle["kmeans"].predict(bundle["scaler"].transform(X))
        assert labels.tolist() == seg.summary["Cluster"].tolist()

    def test_missing_joblib_skips_model_file_but_keeps_metadata(self, planted_summary, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "joblib", None)      # makes `import joblib` raise ImportError
        cra.assign_segments(planted_summary, n_clusters=3, output_dir=tmp_path)
        assert (tmp_path / "model_metadata.json").exists()
        assert not (tmp_path / "segmentation_model.joblib").exists()


# ======================================================================
# Plots
# ======================================================================

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

PLOT_CASES = [
    ("feature_correlation.png", lambda seg, out: cra.plot_feature_correlation(seg.summary, out)),
    ("cluster_k_selection.png",
     lambda seg, out: cra.plot_k_selection(cra.evaluate_k_range(seg.X_scaled, range(2, 5)), 3, out)),
    ("customer_clusters_pca.png", lambda seg, out: cra.plot_customer_segments_pca(seg.X_scaled, seg.summary, out)),
    ("customer_clusters.png", lambda seg, out: cra.plot_customer_segments(seg.summary, out)),
    ("cluster_profile.png", lambda seg, out: cra.plot_cluster_profile(seg.summary, out)),
    ("customer_risk_distribution.png", lambda seg, out: cra.plot_risk_distribution(seg.summary, out)),
    ("reliability_distribution.png", lambda seg, out: cra.plot_reliability_histogram(seg.summary, out)),
]


@integration
class TestPlots:
    @pytest.mark.parametrize("filename, draw", PLOT_CASES, ids=[c[0] for c in PLOT_CASES])
    def test_each_plot_writes_a_valid_png_and_leaks_no_figure(self, segmented, tmp_path, filename, draw):
        path = draw(segmented, tmp_path)
        assert path == tmp_path / filename
        assert path.read_bytes()[:8] == PNG_SIGNATURE
        assert plt.get_fignums() == []

    def test_correlation_plot_also_exports_the_matrix_as_csv(self, segmented, tmp_path, captured_figures):
        cra.plot_feature_correlation(segmented.summary, tmp_path)
        corr = pd.read_csv(tmp_path / "feature_correlation.csv", index_col=0)
        assert list(corr.columns) == cra.CLUSTER_FEATURES and list(corr.index) == cra.CLUSTER_FEATURES
        np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0)
        ax = captured_figures[0].axes[0]
        assert [t.get_text() for t in ax.get_xticklabels()] == cra.CLUSTER_FEATURES

    def test_k_selection_plot_has_four_panels_with_the_chosen_k_marked(self, segmented, tmp_path, captured_figures):
        cra.plot_k_selection(cra.evaluate_k_range(segmented.X_scaled, range(2, 6)), 3, tmp_path)
        fig = captured_figures[0]
        assert len(fig.axes) == 4
        assert "chosen k = 3" in fig._suptitle.get_text()
        for ax in fig.axes:
            marker = [ln for ln in ax.lines if len(ln.get_xdata()) == 2 and np.allclose(ln.get_xdata(), 3)]
            assert len(marker) == 1                       # the dashed axvline at the chosen k

    def test_pca_plot_has_one_legend_entry_per_segment_and_reports_variance(self, segmented, tmp_path, captured_figures):
        cra.plot_customer_segments_pca(segmented.X_scaled, segmented.summary, tmp_path)
        ax = captured_figures[0].axes[0]
        legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
        assert legend_labels == set(segmented.summary["Customer_Segment"])
        assert "PCA Projection" in ax.get_title() and "%" in ax.get_title()

    def test_segment_scatter_uses_real_spend_on_x_and_reliability_on_y(self, segmented, tmp_path, captured_figures):
        cra.plot_customer_segments(segmented.summary, tmp_path)
        ax = captured_figures[0].axes[0]
        assert "Total Transaction Value" in ax.get_xlabel() and ax.get_ylabel() == "Reliability Score"
        assert {t.get_text() for t in ax.get_legend().get_texts()} == set(segmented.summary["Customer_Segment"])
        plotted = sum(len(c.get_offsets()) for c in ax.collections)
        assert plotted == len(segmented.summary)           # every customer appears exactly once

    def test_cluster_profile_draws_one_bar_per_feature_per_segment(self, segmented, tmp_path, captured_figures):
        cra.plot_cluster_profile(segmented.summary, tmp_path)
        ax = captured_figures[0].axes[0]
        n_segments = segmented.summary["Customer_Segment"].nunique()
        assert len(ax.patches) == n_segments * len(cra.CLUSTER_FEATURES)
        assert all(np.isfinite(p.get_height()) for p in ax.patches)

    def test_cluster_profile_survives_a_feature_that_is_identical_across_segments(self, segmented, tmp_path, captured_figures):
        segmented.summary["Monthly_Value_CV"] = 1.0       # zero std across segments -> divide-by-zero guard
        cra.plot_cluster_profile(segmented.summary, tmp_path)
        assert all(np.isfinite(p.get_height()) for p in captured_figures[0].axes[0].patches)

    def test_risk_distribution_has_one_bar_per_observed_level(self, segmented, tmp_path, captured_figures):
        cra.plot_risk_distribution(segmented.summary, tmp_path)
        ax = captured_figures[0].axes[0]
        assert len(ax.patches) == segmented.summary["Risk_Level"].nunique()
        assert sum(p.get_height() for p in ax.patches) == len(segmented.summary)

    def test_reliability_histogram_has_20_bins_covering_every_customer(self, segmented, tmp_path, captured_figures):
        cra.plot_reliability_histogram(segmented.summary, tmp_path)
        ax = captured_figures[0].axes[0]
        assert f"n={len(segmented.summary)}" in ax.get_title()
        assert len(ax.patches) == 20
        assert sum(p.get_height() for p in ax.patches) == len(segmented.summary)


# ======================================================================
# Report writers
# ======================================================================

class TestPersistence:
    def test_single_customer_report_is_valid_json_with_full_detail(self, make_clean, tmp_path):
        df = make_clean([("I1", 1, "2010-01-10", 150.0, 42), ("I2", 1, "2010-02-10", 150.0, 42)])
        report = cra.analyze_customer(df, 42, cra.AnalysisConfig(), pd.Period("2010-02", "M"))
        cra._save_single_customer_report(report, tmp_path)

        payload = json.loads((tmp_path / "customer_42.json").read_text())
        assert set(payload) == {"Customer_ID", "Summary", "Reliability", "Behavioral", "Details"}
        assert payload["Customer_ID"] == 42
        assert len(payload["Details"]) == 2
        assert payload["Details"][0]["period"] == "2010_1"
        assert payload["Behavioral"]["First_Purchase_Date"] == "2010-01-10"

    def test_portfolio_outputs_write_four_consistent_files(self, segmented, tmp_path):
        cra._save_portfolio_outputs(segmented.summary, segmented.cluster_summary, tmp_path)
        names = {p.name for p in tmp_path.iterdir()}
        assert names == {"all_customers_summary.csv", "all_customers_summary.json",
                         "cluster_summary.csv", "cluster_summary.json"}

        csv = pd.read_csv(tmp_path / "all_customers_summary.csv")
        assert len(csv) == len(segmented.summary)
        assert {"Cluster", "Customer_Segment", "Risk_Level"} <= set(csv.columns)
        records = json.loads((tmp_path / "all_customers_summary.json").read_text())
        assert len(records) == len(segmented.summary)
        assert len(pd.read_csv(tmp_path / "cluster_summary.csv")) == 3


# ======================================================================
# CLI
# ======================================================================

class TestParseArgs:
    def test_defaults(self):
        args = cra.parse_args([])
        assert args.data_path == cra.DEFAULT_DATA_PATH
        assert args.output_dir == cra.DEFAULT_OUTPUT_DIR
        assert args.customer_id is None and args.country is None
        assert args.exclude_cancellations is False and args.drop_duplicates is False
        assert args.spend_threshold == 100.0 and args.max_decline_months == 4
        assert args.n_clusters == 4 and args.k_range == "2-8"
        assert args.skip_plots is False and args.log_level == "INFO"

    def test_every_option_is_parsed_with_the_right_type(self):
        args = cra.parse_args([
            "--data-path", "in.xlsx", "--output-dir", "out", "--customer-id", "13078",
            "--country", "United Kingdom", "--exclude-cancellations", "--drop-duplicates",
            "--spend-threshold", "250.5", "--max-decline-months", "3", "--n-clusters", "5",
            "--k-range", "3-6", "--skip-plots", "--log-level", "DEBUG"])
        assert args.data_path == Path("in.xlsx") and args.output_dir == Path("out")
        assert args.customer_id == 13078 and isinstance(args.customer_id, int)
        assert args.country == "United Kingdom"
        assert args.exclude_cancellations and args.drop_duplicates and args.skip_plots
        assert args.spend_threshold == 250.5 and args.max_decline_months == 3
        assert args.n_clusters == 5 and args.k_range == "3-6" and args.log_level == "DEBUG"

    @pytest.mark.parametrize("argv", [
        ["--log-level", "VERBOSE"],
        ["--customer-id", "abc"],
        ["--n-clusters", "many"],
        ["--spend-threshold", "lots"],
        ["--unknown-flag"],
    ])
    def test_invalid_arguments_exit_with_usage_error(self, argv):
        with pytest.raises(SystemExit) as exc:
            cra.parse_args(argv)
        assert exc.value.code == 2


# ======================================================================
# main()  -  end-to-end
# ======================================================================

HEADER = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
PLOT_FILES = {"feature_correlation.png", "cluster_k_selection.png", "customer_clusters_pca.png",
              "customer_clusters.png", "cluster_profile.png", "customer_risk_distribution.png",
              "reliability_distribution.png"}


@pytest.fixture
def workbook(tmp_path, write_workbook):
    """12 customers (IDs 20001-20012) over 15 months with varied engagement, split over two
    year sheets; every 3rd customer is French. Deterministic (seeded)."""
    rng = np.random.default_rng(7)
    months = pd.period_range("2010-01", "2011-03", freq="M")
    sheet_1, sheet_2 = [HEADER], [HEADER]
    for idx, cid in enumerate(range(20001, 20013)):
        country = "France" if idx % 3 == 0 else "United Kingdom"
        for m in months:
            if m == months[0] or rng.random() < 0.35 + 0.05 * idx:
                for _ in range(int(rng.integers(1, 4))):
                    row = [f"{m.year}{m.month:02d}{cid}", "SKU", "Item", int(rng.integers(1, 20)),
                           datetime(m.year, m.month, int(rng.integers(1, 28)), 10, 0),
                           float(rng.choice([2.5, 5.0, 12.0, 30.0])), float(cid), country]
                    (sheet_1 if m < pd.Period("2010-11") else sheet_2).append(row)
    # one cancellation and one exact duplicate row, to exercise the data-quality report
    sheet_1.append(["C201001", "SKU", "Item", -2, datetime(2010, 2, 3, 10, 0), 5.0, 20001.0, "France"])
    sheet_1.append(list(sheet_1[1]))
    return write_workbook(tmp_path / "transactions.xlsx",
                          {"Year 2009-2010": sheet_1, "Year 2010-2011": sheet_2})


def run_main(workbook, out_dir, *extra):
    cra.main(["--data-path", str(workbook), "--output-dir", str(out_dir),
              "--n-clusters", "2", "--k-range", "2-4", *extra])


@integration
class TestMain:
    def test_full_run_writes_every_artifact(self, workbook, tmp_path):
        out = tmp_path / "out"
        run_main(workbook, out, "--customer-id", "20001")

        produced = {p.name for p in out.iterdir()}
        assert PLOT_FILES <= produced
        assert {"data_quality_report.json", "customer_20001.json", "all_customers_summary.csv",
                "all_customers_summary.json", "cluster_summary.csv", "cluster_summary.json",
                "model_metadata.json", "segmentation_model.joblib", "feature_correlation.csv",
                "cluster_k_evaluation.csv"} <= produced

        summary = pd.read_csv(out / "all_customers_summary.csv")
        assert len(summary) == 12 and summary["Customer_ID"].is_unique
        assert {"Cluster", "Customer_Segment", "Risk_Level", "Reliability_Score"} <= set(summary.columns)
        assert set(summary["Risk_Level"]) <= {"High Risk", "Medium Risk", "Low Risk"}
        assert pd.read_csv(out / "cluster_k_evaluation.csv")["k"].tolist() == [2, 3, 4]

    def test_data_quality_report_reflects_the_input(self, workbook, tmp_path):
        out = tmp_path / "out"
        run_main(workbook, out, "--skip-plots")
        report = json.loads((out / "data_quality_report.json").read_text())
        assert report["cancelled_invoice_rows"] == 1
        assert report["fully_duplicate_transaction_rows"] >= 2     # the planted duplicate pair
        assert report["unique_customers"] == 12
        assert report["country_filter"] is None

    def test_skip_plots_omits_charts_and_k_evaluation_but_keeps_data_outputs(self, workbook, tmp_path):
        out = tmp_path / "out"
        run_main(workbook, out, "--skip-plots")
        produced = {p.name for p in out.iterdir()}
        assert produced.isdisjoint(PLOT_FILES | {"cluster_k_evaluation.csv", "feature_correlation.csv"})
        assert {"all_customers_summary.csv", "cluster_summary.csv", "model_metadata.json"} <= produced

    def test_country_filter_restricts_the_scored_population(self, workbook, tmp_path):
        out = tmp_path / "out"
        run_main(workbook, out, "--country", "France", "--skip-plots")
        assert len(pd.read_csv(out / "all_customers_summary.csv")) == 4
        assert json.loads((out / "data_quality_report.json").read_text())["country_filter"] == "France"

    def test_cleaning_flags_are_forwarded_to_the_loader(self, workbook, tmp_path):
        out = tmp_path / "out"
        run_main(workbook, out, "--exclude-cancellations", "--drop-duplicates", "--skip-plots")
        report = json.loads((out / "data_quality_report.json").read_text())
        assert report["cancellations_excluded"] is True and report["duplicates_dropped"] is True

    def test_spend_threshold_option_changes_the_scores(self, workbook, tmp_path):
        out = tmp_path / "out"
        run_main(workbook, out, "--spend-threshold", "1000000000", "--skip-plots")
        summary = pd.read_csv(out / "all_customers_summary.csv")
        assert (summary["Reliability_Score"] == 0).all()
        assert set(summary["Reliability_Band"]) == {"Likely to Exit"}
        assert set(summary["Risk_Level"]) == {"High Risk"}

    def test_unknown_customer_id_logs_a_warning_and_the_run_continues(self, workbook, tmp_path, caplog):
        out = tmp_path / "out"
        with caplog.at_level(logging.WARNING, logger=cra.logger.name):
            run_main(workbook, out, "--customer-id", "99999", "--skip-plots")
        assert "No transactions found for customer 99999" in caplog.text
        assert not (out / "customer_99999.json").exists()
        assert (out / "all_customers_summary.csv").exists()

    def test_second_run_is_served_from_the_raw_cache(self, workbook, tmp_path, monkeypatch):
        run_main(workbook, tmp_path / "out1", "--skip-plots")
        assert (tmp_path / "transactions.raw.pkl").exists()

        def explode(path):
            raise AssertionError("workbook was re-parsed despite a fresh cache")
        monkeypatch.setattr(cra, "_read_year_sheets", explode)
        run_main(workbook, tmp_path / "out2", "--skip-plots")      # must not raise

    # ---- failure modes ----------------------------------------------------------------

    def test_missing_input_file_exits_with_status_1(self, tmp_path, caplog):
        with caplog.at_level(logging.ERROR, logger=cra.logger.name):
            with pytest.raises(SystemExit) as exc:
                cra.main(["--data-path", str(tmp_path / "nope.xlsx"), "--output-dir", str(tmp_path / "out")])
        assert exc.value.code == 1
        assert "Transactions file not found" in caplog.text

    def test_missing_required_column_exits_with_status_1(self, tmp_path, write_workbook, caplog):
        bad = write_workbook(tmp_path / "bad.xlsx", {"Year 2009-2010": [["Invoice", "Quantity"], ["1", 1]]})
        with caplog.at_level(logging.ERROR, logger=cra.logger.name):
            with pytest.raises(SystemExit) as exc:
                cra.main(["--data-path", str(bad), "--output-dir", str(tmp_path / "out")])
        assert exc.value.code == 1
        assert "missing required column" in caplog.text

    def test_requesting_more_clusters_than_customers_raises_a_clear_error(self, workbook, tmp_path):
        with pytest.raises(ValueError, match="at least 50 scored customers"):
            cra.main(["--data-path", str(workbook), "--output-dir", str(tmp_path / "out"),
                      "--n-clusters", "50", "--skip-plots"])


# ======================================================================
# Known limitations  (strict xfail: each documents a defect found during code review.
# When the script is fixed the test XPASSes, strict mode turns that into a failure, and
# the marker should simply be deleted.)
# ======================================================================

class TestKnownLimitations:

    @pytest.mark.xfail(strict=True, raises=AttributeError,
                       reason="run_portfolio_analysis calls NaT.to_period when the frame is empty "
                              "(e.g. --country that matches nothing).")
    def test_portfolio_analysis_of_an_empty_frame_should_return_an_empty_result(self, make_clean):
        empty = make_clean([("I1", 1, "2010-01-10", 150.0, 1)]).iloc[0:0]
        assert cra.run_portfolio_analysis(empty, cra.AnalysisConfig()).empty

    @integration
    @pytest.mark.xfail(strict=True, raises=AttributeError,
                       reason="Same root cause, user-visible: main() crashes instead of logging "
                              "'No customers were scored' when the country filter matches nothing.")
    def test_main_should_exit_cleanly_when_the_country_filter_matches_nothing(self, workbook, tmp_path):
        cra.main(["--data-path", str(workbook), "--output-dir", str(tmp_path / "out"),
                  "--country", "Atlantis", "--skip-plots"])

    @pytest.mark.xfail(strict=True, raises=ValueError,
                       reason="Non-numeric Customer ID survives dropna, becomes NaN in to_numeric(coerce), "
                              "then astype('int64') raises instead of the row being dropped and counted.")
    def test_non_numeric_customer_id_should_be_dropped_not_crash(self, load_raw, make_raw):
        df, _ = load_raw(make_raw([{"Invoice": "1", "Customer ID": "N/A"},
                                   {"Invoice": "2", "Customer ID": 5.0}]))
        assert df["Invoice"].tolist() == ["2"]

    @pytest.mark.xfail(strict=True, raises=ValueError,
                       reason="evaluate_k_range fits k up to the requested max even when k > n_samples; "
                              "the default --k-range 2-8 therefore crashes for fewer than 8 customers.")
    def test_k_range_larger_than_the_sample_should_be_clamped_or_skipped(self):
        tiny = np.random.default_rng(0).normal(size=(5, 3))
        out = cra.evaluate_k_range(tiny, range(2, 9))
        assert out["k"].max() <= len(tiny)