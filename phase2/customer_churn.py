# ActivePeriodClassifier
# Mirrors the logic on the "pivot" sheet (columns F..Q) in Python.
# Consumes monthly aggregates and populates: valuePerInvoice, tillDate,
# avgSpent (invoice-weighted running average), relIncrIncrease,
# spendFlag, noDeclineFlag, activePeriod.

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd
import os
import json
import matplotlib.pyplot as plt


@dataclass
class MonthRow:
    """One row of the pivot table (one Year_Month bucket)."""
    period: str                 # col F  e.g. "2010_3"
    quantity: float             # col G  Sum of Quantity
    price: float                # col H  Sum of Price (amount spent in month)
    invoices: int               # col I  Count of Invoice
    value_per_invoice: Optional[float] = None   # col J  = H / I
    till_date: Optional[float] = None           # col K  = cumulative sum of H
    avg_spent: Optional[float] = None           # col L  weighted running avg
    rel_incr_increase: Optional[float] = None   # col N  = L(t) - L(t-1)
    spend_flag: Optional[int] = None            # spend > threshold -> 1
    no_decline_flag: Optional[int] = None       # not 4 straight declines -> 1
    active_period: Optional[str] = None         # col O  Active / Inactive


class ActivePeriodClassifier:
    """
    Rules
    -----
    1) spend_flag        : monthly spend (Sum of Price) > spend_threshold  -> 1 else 0
    2) no_decline_flag   : the relative incremental increase (col N) must NOT be
                           negative for 'max_decline_months' consecutive months
                           ending at this row -> 1 else 0
    A month is Active only when BOTH flags equal 1.
    """

    def __init__(self, spend_threshold: float = 100.0, max_decline_months: int = 4):
        """
        Parameters
        ----------
        spend_threshold : minimum monthly spend (Sum of Price) to pass spend_flag (pivot!G32).
        max_decline_months : consecutive declining months allowed before no_decline_flag fails (pivot!G33).
        """
        self.spend_threshold = spend_threshold      # pivot!G32
        self.max_decline_months = max_decline_months  # pivot!G33
        self.rows: List[MonthRow] = []

    # ---------- consume ----------
    def load(self, records: List[Dict]) -> "ActivePeriodClassifier":
        """records: [{'period','quantity','price','invoices'}, ...]"""
        self.rows = [MonthRow(period=r["period"],
                              quantity=float(r["quantity"]),
                              price=float(r["price"]),
                              invoices=int(r["invoices"])) for r in records]
        self.rows.sort(key=self._period_key)
        return self

    @staticmethod
    def _period_key(row: "MonthRow"):
        year, month = str(row.period).split("_")
        return (int(year), int(month))

    # ---------- populate ----------
    def compute(self) -> "ActivePeriodClassifier":
        """Populate value_per_invoice, till_date, avg_spent, rel_incr_increase,
        spend_flag, no_decline_flag, and active_period on every row, in period order.
        Must be called after load(). Returns self for chaining."""
        cum_price = 0.0
        cum_invoices = 0
        prev_avg = None

        for i, r in enumerate(self.rows):
            r.value_per_invoice = r.price / r.invoices if r.invoices else 0.0

            cum_price += r.price
            cum_invoices += r.invoices
            r.till_date = cum_price

            # invoice-weighted running average spend per invoice
            # Avg Spend (Excel Column L)

            if i == 0:
                r.avg_spent = r.value_per_invoice
            else:
                prev = self.rows[i - 1]

                r.avg_spent = (
                (r.value_per_invoice * r.invoices) +
                (prev.value_per_invoice * prev.invoices)
            ) / (r.invoices + prev.invoices)

            r.rel_incr_increase = (
            0.0 if prev_avg is None
            else r.avg_spent - prev_avg
            )

            prev_avg = r.avg_spent

            r.spend_flag = 1 if r.price > self.spend_threshold else 0
            r.no_decline_flag = self._no_decline(i)
            r.active_period = "Active" if (r.spend_flag == 1 and r.no_decline_flag == 1) else "Inactive"

        return self

    def _no_decline(self, i: int) -> int:
        """1 unless rel_incr_increase was negative for all of the last
        max_decline_months rows up to and including row i; 1 if there
        isn't yet enough history to form a full window."""
        n = self.max_decline_months
        if i + 1 < n:                       # not enough history for a full run
            return 1
        window = self.rows[i - n + 1: i + 1]
        if all((w.rel_incr_increase or 0.0) < 0 for w in window):
            return 0
        return 1

    # ---------- output ----------
    def to_records(self) -> List[Dict]:
        """Return computed rows as a list of plain dicts, one per period."""
        return [r.__dict__.copy() for r in self.rows]

    def to_dataframe(self):
        """Return computed rows as a pandas DataFrame (requires pandas)."""
        import pandas as pd
        return pd.DataFrame(self.to_records())

    def active_periods(self) -> List[str]:
        """Return the list of period labels classified as Active."""
        return [r.period for r in self.rows if r.active_period == "Active"]

    def summary(self) -> Dict:
        """Return aggregate stats: month counts, total spend, and final avg_spent."""
        active = self.active_periods()
        return {"months": len(self.rows),
                "active_months": len(active),
                "inactive_months": len(self.rows) - len(active),
                "total_spend": round(sum(r.price for r in self.rows), 2),
                "final_avg_spent": round(self.rows[-1].avg_spent, 4) if self.rows else None}

class ReliabilityScoreCalculator:
    """
    Calculates reliability score based on Active/Inactive months.
    """

    def calculate(self, classifier: ActivePeriodClassifier):
        summary = classifier.summary()

        total_months = summary["months"]
        active_months = summary["active_months"]
        inactive_months = summary["inactive_months"]

        reliability_score = (
            active_months / total_months * 100
            if total_months > 0 else 0
        )
        if reliability_score >= 80:
            prediction = "Very Loyal"

        elif reliability_score >= 60:
            prediction = "Likely to Stay"

        elif reliability_score >= 40:
            prediction = "At Risk"

        else:
            prediction = "Likely to Exit"
        numerical_score = (
            reliability_score * 0.7 +
            (active_months / total_months) * 100 * 0.3
        )

        return {
            "total_months": total_months,
            "active_months": active_months,
            "inactive_months": inactive_months,
            "reliability_score": round(reliability_score, 2),
            "numerical_score": round(numerical_score, 2),
            "prediction": prediction
        }
def analyze_customer(df, customer_id):

    customer_df = df[
        (df["Customer ID"] == customer_id) &
        (df["Country"] == "United Kingdom")
    ].copy()

    if customer_df.empty:
        return None

    customer_df["InvoiceDate"] = pd.to_datetime(customer_df["InvoiceDate"])

    monthly_spend = (
        customer_df
        .groupby(customer_df["InvoiceDate"].dt.to_period("M"))["Price"]
        .sum()
        .reset_index()
    )

    monthly_spend["Abs_Growth"] = monthly_spend["Price"].diff()
    monthly_spend["Pct_Growth"] = monthly_spend["Price"].pct_change() * 100

    monthly_records = []

    for month, group in customer_df.groupby(customer_df["InvoiceDate"].dt.to_period("M")):

        monthly_records.append({
            "period": f"{month.year}_{month.month}",
            "quantity": group["Quantity"].sum(),
            "price": group["Price"].sum(),
            "invoices": group["Invoice"].count()
        })

    clf = ActivePeriodClassifier(
        spend_threshold=100,
        max_decline_months=4
    )

    clf.load(monthly_records).compute()

    reliability = ReliabilityScoreCalculator()
    result = reliability.calculate(clf)

    return {
    "Customer_ID": customer_id,
    "Monthly_KPIs": monthly_spend,
    "Summary": clf.summary(),
    "Reliability": result,
    "Details": clf.to_dataframe()
    }
# ---------- example ----------



if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    df = pd.read_excel(
        "data/customer_transactions.xlsx",
        engine="openpyxl"
    )

    # Single customer
    result = analyze_customer(df, 13078)
    with open("output/customer13078.json", "w") as f:
        json.dump({
        "Customer_ID": result["Customer_ID"],
        "Summary": result["Summary"],
        "Reliability": result["Reliability"],
        "Details": result["Details"].to_dict(orient="records")
    }, f, indent=4)

    print("Customer JSON saved.")

    print("\nSummary")
    print(result["Summary"])

    print("\nReliability")
    print(result["Reliability"])

    print("\nDetailed Results")
    print(result["Details"])

    # All customers summary
    all_results = []

    customer_ids = df["Customer ID"].dropna().unique()

    for cid in customer_ids:

        result = analyze_customer(df, cid)

        if result is None:
            continue

        all_results.append({
        "Customer_ID": cid,
        "Total_Months": result["Reliability"]["total_months"],
        "Active_Months": result["Reliability"]["active_months"],
        "Inactive_Months": result["Reliability"]["inactive_months"],
        "Reliability_Score": result["Reliability"]["reliability_score"],
        "Total_Spend": result["Summary"]["total_spend"],
        "Final_Avg_Spend": result["Summary"]["final_avg_spent"],
        "Numerical_Score": result["Reliability"]["numerical_score"],
        "Prediction": result["Reliability"]["prediction"]
        })

    summary = pd.DataFrame(all_results)

    summary.to_csv(
        "output/all_customers_summary.csv",
        index=False
    )
    

    summary.plot(
    x="Customer_ID",
    y="Reliability_Score",
    kind="bar"
)

    plt.title("Customer Reliability Score")
    plt.tight_layout()
    plt.savefig("output/reliability_plot.png")
    plt.show()

    print("Plot saved to output/reliability_plot.png")
    

    summary.to_json(
    "output/all_customers_summary.json",
    orient="records",
    indent=4
    )

    print("All customers JSON saved.")

    print("\nAll customers summary saved.")