

import pytest

from customer_churn import ActivePeriodClassifier


class TestActivePeriodClassifier:
    """Unit tests for ActivePeriodClassifier."""

    # ------------------------------------------------------------------
    # spend_flag
    # ------------------------------------------------------------------

    def test_spend_above_threshold_sets_spend_flag(self):
        classifier = ActivePeriodClassifier(spend_threshold=100)

        result = classifier.classify_month(
            spend=101,
            relative_increment=0.10,
        )

        assert result.spend_flag == 1

    def test_spend_equal_to_threshold_fails_spend_flag(self):
        """Rule says spend must be strictly greater than threshold."""
        classifier = ActivePeriodClassifier(spend_threshold=100)

        result = classifier.classify_month(
            spend=100,
            relative_increment=0.10,
        )

        assert result.spend_flag == 0

    def test_spend_below_threshold_fails_spend_flag(self):
        classifier = ActivePeriodClassifier(spend_threshold=100)

        result = classifier.classify_month(
            spend=99,
            relative_increment=0.10,
        )

        assert result.spend_flag == 0

    def test_zero_spend_fails_spend_flag(self):
        classifier = ActivePeriodClassifier(spend_threshold=100)

        result = classifier.classify_month(
            spend=0,
            relative_increment=0.10,
        )

        assert result.spend_flag == 0

    def test_custom_spend_threshold(self):
        classifier = ActivePeriodClassifier(spend_threshold=500)

        result = classifier.classify_month(
            spend=501,
            relative_increment=0.10,
        )

        assert result.spend_flag == 1

    # ------------------------------------------------------------------
    # no_decline_flag
    # ------------------------------------------------------------------

    def test_positive_increment_passes_no_decline_flag(self):
        classifier = ActivePeriodClassifier(max_decline_months=4)

        result = classifier.classify_month(
            spend=200,
            relative_increment=0.10,
        )

        assert result.no_decline_flag == 1

    def test_zero_increment_is_not_decline(self):
        classifier = ActivePeriodClassifier(max_decline_months=4)

        result = classifier.classify_month(
            spend=200,
            relative_increment=0.0,
        )

        assert result.no_decline_flag == 1

    def test_single_negative_increment_does_not_immediately_fail(self):
        classifier = ActivePeriodClassifier(max_decline_months=4)

        classifier.classify_month(200, 0.10)
        result = classifier.classify_month(200, -0.05)

        assert result.no_decline_flag == 1

    def test_declines_until_limit_are_allowed(self):
        classifier = ActivePeriodClassifier(max_decline_months=4)

        classifier.classify_month(200, 0.10)

        for _ in range(4):
            result = classifier.classify_month(200, -0.05)

        assert result.no_decline_flag == 1

    def test_fifth_consecutive_decline_fails(self):
        classifier = ActivePeriodClassifier(max_decline_months=4)

        classifier.classify_month(200, 0.10)

        for _ in range(4):
            classifier.classify_month(200, -0.05)

        result = classifier.classify_month(200, -0.05)

        assert result.no_decline_flag == 0

    def test_positive_increment_resets_decline_streak(self):
        classifier = ActivePeriodClassifier(max_decline_months=4)

        classifier.classify_month(200, -0.05)
        classifier.classify_month(200, -0.05)
        classifier.classify_month(200, 0.10)

        result = classifier.classify_month(200, -0.05)

        assert result.no_decline_flag == 1

    def test_zero_increment_resets_decline_streak(self):
        classifier = ActivePeriodClassifier(max_decline_months=4)

        classifier.classify_month(200, -0.05)
        classifier.classify_month(200, -0.05)
        classifier.classify_month(200, 0.0)

        result = classifier.classify_month(200, -0.05)

        assert result.no_decline_flag == 1

    # ------------------------------------------------------------------
    # active flag = spend_flag AND no_decline_flag
    # ------------------------------------------------------------------

    def test_month_is_active_when_both_flags_are_one(self):
        classifier = ActivePeriodClassifier(
            spend_threshold=100,
            max_decline_months=4,
        )

        result = classifier.classify_month(
            spend=200,
            relative_increment=0.10,
        )

        assert result.spend_flag == 1
        assert result.no_decline_flag == 1
        assert result.active_flag == 1

    def test_month_is_inactive_when_spend_flag_is_zero(self):
        classifier = ActivePeriodClassifier(
            spend_threshold=100,
            max_decline_months=4,
        )

        result = classifier.classify_month(
            spend=100,
            relative_increment=0.10,
        )

        assert result.spend_flag == 0
        assert result.no_decline_flag == 1
        assert result.active_flag == 0

    def test_month_is_inactive_when_no_decline_flag_is_zero(self):
        classifier = ActivePeriodClassifier(
            spend_threshold=100,
            max_decline_months=2,
        )

        classifier.classify_month(200, 0.10)
        classifier.classify_month(200, -0.05)
        classifier.classify_month(200, -0.05)

        result = classifier.classify_month(200, -0.05)

        assert result.spend_flag == 1
        assert result.no_decline_flag == 0
        assert result.active_flag == 0

    def test_month_is_inactive_when_both_flags_are_zero(self):
        classifier = ActivePeriodClassifier(
            spend_threshold=100,
            max_decline_months=2,
        )

        classifier.classify_month(50, 0.10)
        classifier.classify_month(50, -0.05)
        classifier.classify_month(50, -0.05)

        result = classifier.classify_month(50, -0.05)

        assert result.spend_flag == 0
        assert result.no_decline_flag == 0
        assert result.active_flag == 0

    # ------------------------------------------------------------------
    # Boundary / configuration tests
    # ------------------------------------------------------------------

    def test_max_decline_months_of_one(self):
        classifier = ActivePeriodClassifier(max_decline_months=1)

        classifier.classify_month(200, 0.10)

        # First decline is still allowed.
        result = classifier.classify_month(200, -0.05)
        assert result.no_decline_flag == 1

        # Second consecutive decline exceeds the limit.
        result = classifier.classify_month(200, -0.05)
        assert result.no_decline_flag == 0

    def test_decline_streak_only_counts_consecutive_months(self):
        classifier = ActivePeriodClassifier(max_decline_months=2)

        classifier.classify_month(200, -0.05)
        classifier.classify_month(200, 0.10)
        classifier.classify_month(200, -0.05)

        result = classifier.classify_month(200, -0.05)

        assert result.no_decline_flag == 1

    def test_high_spend_does_not_override_decline_failure(self):
        classifier = ActivePeriodClassifier(max_decline_months=2)

        classifier.classify_month(1000, 0.10)
        classifier.classify_month(1000, -0.05)
        classifier.classify_month(1000, -0.05)

        result = classifier.classify_month(1000, -0.05)

        assert result.spend_flag == 1
        assert result.no_decline_flag == 0
        assert result.active_flag == 0
