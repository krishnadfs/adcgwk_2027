# ============================================================
# Focus: Value per Invoice, Till Date, Average Spent
# ============================================================

# TEST CASE 1
def test_01_value_per_invoice():
    rows = [
        create_row(1000, 10)
    ]

    classifier = create_classifier(rows)
    classifier.compute()

    assert rows[0].value_per_invoice == 100.0

# TEST CASE 2
def test_02_zero_invoices():
    rows = [
        create_row(1000, 0)
    ]

    classifier = create_classifier(rows)
    classifier.compute()

    assert rows[0].value_per_invoice == 0.0

# TEST CASE 3
def test_03_till_date():
    rows = [
        create_row(100, 5),
        create_row(200, 10),
        create_row(300, 15)
    ]

    classifier = create_classifier(rows)
    classifier.compute()

    assert rows[0].till_date == 100
    assert rows[1].till_date == 300
    assert rows[2].till_date == 600

# TEST CASE 4
def test_04_average_spent():
    rows = [
        create_row(100, 10),
        create_row(200, 10),
        create_row(300, 10)
    ]

    classifier = create_classifier(rows)
    classifier.compute()

    assert rows[0].avg_spent == 10.0
    assert rows[1].avg_spent == 15.0
    assert rows[2].avg_spent == 20.0