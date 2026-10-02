# ============================================================
# TEST CASES 5 - 8
# Focus: Relative Increase and Spend Flag
# ============================================================


# TEST CASE 5
def test_05_positive_relative_increase():
    rows = [
        create_row(100, 10),
        create_row(300, 10)
    ]

    classifier = create_classifier(rows)
    classifier.compute()

    assert rows[0].rel_incr_increase == 0.0
    assert rows[1].rel_incr_increase == 10.0


# TEST CASE 6
def test_06_negative_relative_increase():
    rows = [
        create_row(300, 10),
        create_row(100, 10)
    ]

    classifier = create_classifier(rows)
    classifier.compute()

    assert rows[0].rel_incr_increase == 0.0
    assert rows[1].rel_incr_increase == -10.0


# TEST CASE 7
def test_07_spend_above_threshold():
    rows = [
        create_row(150, 10)
    ]

    classifier = create_classifier(
        rows,
        spend_threshold=100
    )

    classifier.compute()

    assert rows[0].spend_flag == 1


# TEST CASE 8
def test_08_spend_exact_threshold():
    rows = [
        create_row(100, 10)
    ]

    classifier = create_classifier(
        rows,
        spend_threshold=100
    )

    classifier.compute()

    assert rows[0].spend_flag == 0