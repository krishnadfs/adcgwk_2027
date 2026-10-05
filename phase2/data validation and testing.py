# ============================================================
# TEAM-6
# Member: Spurthi
# TEST CASES: TC01 - TC05
# ============================================================

import unittest

try:
    import pandas as pd  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pandas is required for this test file. Install it with: pip install pandas"
    ) from exc


def create_test_data():

    data = {
        "Customer_ID": ["C001", "C002", "C003", "C004",
                        "C005", "C006", "C007", "C008"],

        "Total_Spend": [1000, 2500, 5000, 7500,
                        1500, 3000, 9000, 2000],

        "Reliability_Score": [20, 45, 75, 90,
                              25, 55, 95, 65],

        "Active_Months": [2, 5, 8, 12,
                          3, 6, 14, 7],

        "Inactive_Months": [10, 7, 4, 1,
                            9, 5, 0, 3],

        "Cluster": [0, 1, 2, 3,
                    0, 1, 3, 2]
    }

    return pd.DataFrame(data)


class TestSpurthi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.summary = create_test_data()

    # TC01
    def test_01_data_loading(self):

        self.assertIsNotNone(self.summary)
        self.assertGreater(len(self.summary), 0)

        print("TC01 PASS - Customer data loaded successfully.")

    # TC02
    def test_02_required_columns(self):

        required_columns = [
            "Customer_ID",
            "Total_Spend",
            "Reliability_Score",
            "Active_Months",
            "Inactive_Months",
            "Cluster"
        ]

        for column in required_columns:
            self.assertIn(column, self.summary.columns)

        print("TC02 PASS - All required columns are available.")

    # TC03
    def test_03_customer_id(self):

        self.assertFalse(
            self.summary["Customer_ID"].isnull().any()
        )

        self.assertTrue(
            self.summary["Customer_ID"]
            .astype(str)
            .str.len()
            .gt(0)
            .all()
        )

        print("TC03 PASS - Customer IDs are valid.")

    # TC04
    def test_04_total_spend(self):

        self.assertFalse(
            self.summary["Total_Spend"].isnull().any()
        )

        self.assertTrue(
            (self.summary["Total_Spend"] >= 0).all()
        )

        print("TC04 PASS - Total Spend values are valid.")

    # TC05
    def test_05_reliability_score_range(self):

        self.assertTrue(
            (self.summary["Reliability_Score"] >= 0).all()
        )

        self.assertTrue(
            (self.summary["Reliability_Score"] <= 100).all()
        )

        print("TC05 PASS - Reliability Score values are within 0-100.")


if __name__ == "__main__":
    unittest.main()