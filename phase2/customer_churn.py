from pathlib import Path
from datetime import datetime
import json
import math

import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# ============================================================
# 1. USER SETTINGS
# ============================================================

CUSTOMER_ID = 12589

INPUT_FILE = Path("data/source.xlsx")

OUTPUT_DIR = Path("output")

# Minimum number of months required
MIN_MONTHS_REQUIRED = 5

MONTHLY_SPEND_THRESHOLD = 100

MAX_CONSECUTIVE_DECLINING_MONTHS = 4


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_customer_id(value):

    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def convert_excel_date(value):

    if value is None:
        return pd.NaT

    if isinstance(value, float) and math.isnan(value):
        return pd.NaT

    if isinstance(value, datetime):
        return pd.Timestamp(value)

    if isinstance(value, (int, float)):
        return (
            pd.Timestamp("1899-12-30")
            + pd.to_timedelta(float(value), unit="D")
        )

    return pd.to_datetime(value, errors="coerce")


# ============================================================
# 3. FIND TRANSACTION SHEETS
# ============================================================

def find_transaction_sheets(workbook):

    expected_sheets = [
        "Year 2009-2010",
        "Year 2010-2011"
    ]

    existing_sheets = set(workbook.sheetnames)

    found = [
        sheet
        for sheet in expected_sheets
        if sheet in existing_sheets
    ]

    if len(found) == 2:
        return found

    year_sheets = [
        sheet
        for sheet in workbook.sheetnames
        if str(sheet).lower().startswith("year ")
    ]

    if len(year_sheets) >= 2:
        return year_sheets[:2]

    raise ValueError(
        "Could not find the transaction sheets. "
        "Expected 'Year 2009-2010' and 'Year 2010-2011'."
    )


# ============================================================
# 4. READ CUSTOMER DATA
# ============================================================

def read_customer_data(input_file, customer_id):

    if not input_file.exists():

        raise FileNotFoundError(
            f"Excel file not found:\n"
            f"{input_file.resolve()}\n\n"
            f"Make sure the source Excel file is located at:\n"
            f"data/source.xlsx"
        )

    workbook = load_workbook(
        input_file,
        read_only=True,
        data_only=True
    )

    sheets = find_transaction_sheets(workbook)

    required_columns = {
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country"
    }

    customer_id = clean_customer_id(customer_id)

    records = []

    for sheet_name in sheets:

        worksheet = workbook[sheet_name]

        rows = worksheet.iter_rows(values_only=True)

        try:
            headers = list(next(rows))
        except StopIteration:
            continue

        header_map = {
            str(header).strip(): index
            for index, header in enumerate(headers)
            if header is not None
        }

        missing = required_columns - set(header_map.keys())

        if missing:
            workbook.close()

            raise ValueError(
                f"Missing columns in {sheet_name}: {missing}"
            )

        for row in rows:

            row_customer_id = row[
                header_map["Customer ID"]
            ]

            if clean_customer_id(row_customer_id) != customer_id:
                continue

            invoice_date = convert_excel_date(
                row[header_map["InvoiceDate"]]
            )

            records.append({

                "Invoice": row[header_map["Invoice"]],

                "StockCode": row[header_map["StockCode"]],

                "Description": row[header_map["Description"]],

                "Quantity": row[header_map["Quantity"]],

                "InvoiceDate": invoice_date,

                "Price": row[header_map["Price"]],

                "Customer ID": row_customer_id,

                "Country": row[header_map["Country"]],

                "SourceSheet": sheet_name
            })

    workbook.close()

    if not records:

        raise ValueError(
            f"No transactions found for Customer ID "
            f"{customer_id}."
        )

    df = pd.DataFrame(records)

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    ).fillna(0)

    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce"
    ).fillna(0)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["InvoiceDate"]
    ).copy()

    df["Year_Month"] = (
        df["InvoiceDate"]
        .dt.to_period("M")
        .astype(str)
    )

    df = df.sort_values(
        ["InvoiceDate", "Invoice"]
    ).reset_index(drop=True)

    return df, sheets


# ============================================================
# 5. CREATE AUTOMATIC PIVOT
# ============================================================

def create_pivot(customer_df):

    pivot = (
        customer_df
        .groupby("Year_Month", as_index=False)
        .agg(
            Quantity=("Quantity", "sum"),
            Price=("Price", "sum"),
            Invoice=("Invoice", "count")
        )
        .sort_values("Year_Month")
        .reset_index(drop=True)
    )

    pivot["ATV"] = (
        pivot["Price"]
        /
        pivot["Invoice"].replace(0, pd.NA)
    ).fillna(0)

    pivot["TillDate"] = (
        pivot["Price"].cumsum()
    )

    pivot["CumulativeInvoices"] = (
        pivot["Invoice"].cumsum()
    )

    pivot["avgSpent"] = (
        pivot["TillDate"]
        /
        pivot["CumulativeInvoices"].replace(
            0,
            pd.NA
        )
    ).fillna(0)

    pivot["weightedSpent"] = (
        pivot["avgSpent"]
        -
        pivot["ATV"]
    )

    # Month-to-month change
    pivot["relativeIncrementalIncrease"] = (
        pivot["avgSpent"].diff().fillna(0)
    )

    pivot["spendFlag"] = (
        pivot["Price"]
        >
        MONTHLY_SPEND_THRESHOLD
    ).astype(int)

    pivot["noDeclineFlag"] = 1

    n = MAX_CONSECUTIVE_DECLINING_MONTHS

    for i in range(len(pivot)):

        if i + 1 < n:
            pivot.loc[i, "noDeclineFlag"] = 1
            continue

        window = pivot.loc[
            i - n + 1:i,
            "relativeIncrementalIncrease"
        ]

        if (window < 0).all():

            pivot.loc[
                i,
                "noDeclineFlag"
            ] = 0

        else:

            pivot.loc[
                i,
                "noDeclineFlag"
            ] = 1

    pivot["activePeriod"] = "Inactive"

    active_condition = (
        (pivot["spendFlag"] == 1)
        &
        (pivot["noDeclineFlag"] == 1)
    )

    pivot.loc[
        active_condition,
        "activePeriod"
    ] = "Active"

    return pivot


# ============================================================
# 6. CUSTOMER SEGMENTATION
# ============================================================

def identify_customer_stage(pivot):

    pivot = pivot.copy()

    pivot["recentSpendChange"] = (
        pivot["Price"].diff()
    )

    pivot["recentAvgSpendChange"] = (
        pivot["avgSpent"].diff()
    )

    pivot["customerStage"] = "Stable"

    exit_condition = (
        (pivot["activePeriod"] == "Inactive")
        &
        (pivot["noDeclineFlag"] == 0)
    )

    pivot.loc[
        exit_condition,
        "customerStage"
    ] = "Exit Risk"

    watch_condition = (
        (pivot["activePeriod"] == "Inactive")
        &
        (pivot["noDeclineFlag"] == 1)
    )

    pivot.loc[
        watch_condition,
        "customerStage"
    ] = "Watch"

    if len(pivot) >= 3:

        recent = pivot.tail(3)

        average_spend_change = (
            recent["Price"].diff().mean()
        )

        latest_active = (
            pivot.iloc[-1]["activePeriod"]
            == "Active"
        )

        if (
            latest_active
            and
            average_spend_change > 0
        ):

            pivot.loc[
                pivot.index[-1],
                "customerStage"
            ] = "Raise/Growth"

    latest = pivot.iloc[-1]

    current_stage = latest[
        "customerStage"
    ]

    exit_rows = pivot.index[
        pivot["customerStage"]
        == "Exit Risk"
    ].tolist()

    if exit_rows:
        first_exit_period = pivot.loc[
            exit_rows[0],
            "Year_Month"
        ]
    else:
        first_exit_period = None

    growth_rows = pivot.index[
        pivot["customerStage"]
        == "Raise/Growth"
    ].tolist()

    if growth_rows:
        latest_growth_period = pivot.loc[
            growth_rows[-1],
            "Year_Month"
        ]
    else:
        latest_growth_period = None

    summary = {

        "current_stage": str(current_stage),

        "latest_period":
            str(latest["Year_Month"]),

        "latest_monthly_spend":
            round(float(latest["Price"]), 2),

        "latest_active_period":
            str(latest["activePeriod"]),

        "first_exit_risk_signal_period":
            first_exit_period,

        "latest_raise_growth_signal_period":
            latest_growth_period,

        "minimum_months_required":
            MIN_MONTHS_REQUIRED,

        "monthly_spend_threshold":
            MONTHLY_SPEND_THRESHOLD,

        "max_consecutive_declining_months":
            MAX_CONSECUTIVE_DECLINING_MONTHS,

        "note":
            "Segmentation is a rule-based historical "
            "signal and is not a guaranteed prediction "
            "of future customer behaviour."
    }

    return pivot, summary
# ============================================================
# 7. CREATE JSON
# ============================================================

def create_json(customer_id, customer_df, pivot, sheets, summary):

    pivot_data = (
        pivot
        .where(pd.notna(pivot), None)
        .to_dict(orient="records")
    )

    month_count = customer_df["Year_Month"].nunique()

    return {
        "customer_id": clean_customer_id(customer_id),

        "source_sheets": sheets,

        "transaction_count": int(len(customer_df)),

        "distinct_months": int(month_count),

        "minimum_months_required": int(MIN_MONTHS_REQUIRED),

        "minimum_month_requirement_met": bool(
            month_count >= MIN_MONTHS_REQUIRED
        ),

        "date_from": customer_df["InvoiceDate"]
        .min()
        .strftime("%Y-%m-%d"),

        "date_to": customer_df["InvoiceDate"]
        .max()
        .strftime("%Y-%m-%d"),

        "signal_summary": summary,

        "monthly_pivot": pivot_data
    }


# ============================================================
# 8. SAVE EXCEL
# ============================================================

def save_excel(customer_df, pivot, output_file):

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl"
    ) as writer:

        customer_df.to_excel(
            writer,
            sheet_name="Customer_Transactions",
            index=False
        )

        pivot.to_excel(
            writer,
            sheet_name="Customer_Pivot",
            index=False
        )

    workbook = load_workbook(output_file)

    for worksheet in workbook.worksheets:

        worksheet.freeze_panes = "A2"

        worksheet.auto_filter.ref = worksheet.dimensions

        for cell in worksheet[1]:

            cell.font = Font(bold=True)

            cell.fill = PatternFill(
                "solid",
                fgColor="D9EAF7"
            )

            cell.alignment = Alignment(
                horizontal="center"
            )

        for column in worksheet.columns:

            maximum_length = 0

            column_letter = get_column_letter(
                column[0].column
            )

            for cell in column:

                value = (
                    ""
                    if cell.value is None
                    else str(cell.value)
                )

                maximum_length = max(
                    maximum_length,
                    len(value)
                )

            worksheet.column_dimensions[
                column_letter
            ].width = min(
                max(maximum_length + 2, 10),
                35
            )

    workbook.save(output_file)
    # ============================================================
# 9. CREATE PLOT
# ============================================================

def create_plot(pivot, customer_id, output_file):

    figure, axis = plt.subplots(figsize=(14, 7))

    x = range(len(pivot))

    axis.plot(
        x,
        pivot["Price"],
        marker="o",
        label="Monthly Spend"
    )

    axis.plot(
        x,
        pivot["avgSpent"],
        marker="o",
        label="Running Average Spend"
    )

    axis.axhline(
        MONTHLY_SPEND_THRESHOLD,
        linestyle="--",
        label="Spend Threshold"
    )

    active = pivot["activePeriod"] == "Active"

    if active.any():
        axis.scatter(
            [i for i, value in enumerate(active) if value],
            pivot.loc[active, "Price"],
            s=60,
            label="Active"
        )

    exit_risk = pivot["customerStage"] == "Exit Risk"

    if exit_risk.any():
        axis.scatter(
            [i for i, value in enumerate(exit_risk) if value],
            pivot.loc[exit_risk, "Price"],
            marker="x",
            s=100,
            label="Exit Risk"
        )

    axis.set_xticks(list(x))

    axis.set_xticklabels(
        pivot["Year_Month"],
        rotation=45,
        ha="right"
    )

    axis.set_title(
        f"Customer {customer_id} - "
        "Spending and Activity Analysis"
    )

    axis.set_xlabel("Year-Month")
    axis.set_ylabel("Amount")

    axis.grid(True, alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_file,
        dpi=160
    )

    plt.close(figure)
    # ============================================================
# 10. MAIN PROGRAM
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    print(f"Customer ID : {CUSTOMER_ID}")

    print(f"Input file  : {INPUT_FILE}")

    print(
        f"Minimum months required : "
        f"{MIN_MONTHS_REQUIRED}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Excel file not found:\n"
            f"{INPUT_FILE.resolve()}\n\n"
            f"Make sure your Excel file is saved as:\n"
            f"data/source.xlsx"
        )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print(
        "\nSTEP 1: Reading customer transactions..."
    )

    customer_df, sheets = read_customer_data(
        INPUT_FILE,
        CUSTOMER_ID
    )

    print(
        f"Transactions found: "
        f"{len(customer_df)}"
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print(
        "\nSTEP 2: Checking minimum month requirement..."
    )

    month_count = (
        customer_df["Year_Month"].nunique()
    )

    print(
        f"Distinct months: "
        f"{month_count}"
    )

    if month_count < MIN_MONTHS_REQUIRED:

        raise ValueError(
            f"Customer {CUSTOMER_ID} has only "
            f"{month_count} months of data. "
            f"At least {MIN_MONTHS_REQUIRED} "
            f"months are required."
        )

    print(
        f"Minimum month requirement "
        f"({MIN_MONTHS_REQUIRED} months): PASSED"
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print(
        "\nSTEP 3: Creating automatic pivot..."
    )

    pivot = create_pivot(
        customer_df
    )

    print(
        f"Pivot rows created: "
        f"{len(pivot)}"
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print(
        "\nSTEP 4: Identifying customer stage..."
    )

    pivot, summary = identify_customer_stage(
        pivot
    )

    print(
        f"Current customer stage: "
        f"{summary['current_stage']}"
    )
    # --------------------------------------------------------
    # STEP 5 - Create JSON
    # --------------------------------------------------------

    print(
        "\nSTEP 5: Creating JSON..."
    )

    json_data = create_json(
        CUSTOMER_ID,
        customer_df,
        pivot,
        sheets,
        summary
    )

    customer_id_text = clean_customer_id(
        CUSTOMER_ID
    )

    json_file = (
        OUTPUT_DIR
        / f"{customer_id_text}_analysis.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # STEP 6 - Create Excel
    # --------------------------------------------------------

    print(
        "\nSTEP 6: Creating Excel output..."
    )

    excel_file = (
        OUTPUT_DIR
        / f"{customer_id_text}_analysis.xlsx"
    )

    save_excel(
        customer_df,
        pivot,
        excel_file
    )

    # --------------------------------------------------------
    # STEP 7 - Create Plot
    # --------------------------------------------------------

    print(
        "\nSTEP 7: Creating plot..."
    )

    plot_file = (
        OUTPUT_DIR
        / f"{customer_id_text}_analysis.png"
    )

    create_plot(
        pivot,
        CUSTOMER_ID,
        plot_file
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "PROCESS COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )

    print(
        f"Customer ID     : {CUSTOMER_ID}"
    )

    print(
        f"Transactions    : {len(customer_df)}"
    )

    print(
        f"Distinct months : {month_count}"
    )

    print(
        f"Minimum required: "
        f"{MIN_MONTHS_REQUIRED} months"
    )

    print(
        f"Current stage   : "
        f"{summary['current_stage']}"
    )

    print(
        "\nFiles created:"
    )

    print(
        f"  {excel_file}"
    )

    print(
        f"  {json_file}"
    )

    print(
        f"  {plot_file}"
    )

    print(
        "=" * 60
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()