# ============================================================
# TEST CASES 9 - 11
# Focus: No Decline Flag
# ============================================================


# TEST CASE 9
def test_09_initial_no_decline():
    rows = [
        create_row(300, 10),
        create_row(200, 10)
    ]

    classifier = create_classifier(
        rows,
        max_decline_months=3
    )

    classifier.compute()

    assert rows[0].no_decline_flag == 1
    assert rows[1].no_decline_flag == 1


# TEST CASE 10
def test_10_three_consecutive_declines():
    rows = [
        create_row(500, 10),
        create_row(400, 10),
        create_row(300, 10),
        create_row(200, 10)
    ]

    classifier = create_classifier(
        rows,
        max_decline_months=3
    )

    classifier.compute()

    assert rows[3].no_decline_flag == 0


# TEST CASE 11
def test_11_recovery_after_decline():
    rows = [
        create_row(500, 10),
        create_row(400, 10),
        create_row(300, 10),
        create_row(600, 10)
    ]

    classifier = create_classifier(
        rows,
        max_decline_months=3
    )

    classifier.compute()

    assert rows[3].no_decline_flag == 1