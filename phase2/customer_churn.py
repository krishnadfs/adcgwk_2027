# ActivePeriodClassifier
# Mirrors the logic on the "pivot" sheet (columns F..Q) in Python.
# Consumes monthly aggregates and populates: valuePerInvoice, tillDate,
# avgSpent (invoice-weighted running average), relIncrIncrease,
# spendFlag, noDeclineFlag, activePeriod.

from dataclasses import dataclass, field
from typing import List, Dict, Optional


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
            r.avg_spent = cum_price / cum_invoices if cum_invoices else 0.0

            r.rel_incr_increase = 0.0 if prev_avg is None else r.avg_spent - prev_avg
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


# ---------- example ----------
if __name__ == "__main__":
    data = [
        {"period": "2009_12", "quantity": 735, "price": 218.52, "invoices": 55},
        {"period": "2010_1",  "quantity": 293, "price": 95.83,  "invoices": 25},
        {"period": "2010_2",  "quantity": 273, "price": 71.31,  "invoices": 19},
        {"period": "2010_3",  "quantity": 423, "price": 163.01, "invoices": 41},
    ]
    clf = ActivePeriodClassifier(spend_threshold=100, max_decline_months=4).load(data).compute()
    for rec in clf.to_records():
        print(rec["period"], round(rec["avg_spent"], 4), rec["spend_flag"], rec["no_decline_flag"], rec["active_period"])
    print(clf.summary())
