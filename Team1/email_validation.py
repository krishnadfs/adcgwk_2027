from pathlib import Path
import re
import pandas as pd


def validate_email(email: object) -> str:
    if pd.isna(email) or str(email).strip() == "":
        return "❌ Empty / Missing"

    email = str(email).strip()

    if email[0].isupper():
        return "❌ Must not start with a capital letter"

    pattern = r'^[a-z][a-zA-Z0-9._%+-]*@([a-zA-Z]+)\.com$'
    match = re.match(pattern, email)

    if not match:
        if "@" not in email:
            return "❌ Missing @"
        if not email.split("@")[0][0].isalpha():
            return "❌ Must start with a letter"
        if not email.endswith(".com"):
            return "❌ Must end with .com"
        return "❌ Invalid Format"

    provider = match.group(1).lower()

    if provider not in {"gmail", "hotmail", "yahoo", "google", "microsoft"}:
        return f"❌ Invalid Provider ({provider})"

    return "✅ Valid"


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "team-1_dataset.xlsx"
    output_file = base_dir / "email_validation_results.xlsx"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"📂 Reading file: {input_file}")
    df = pd.read_excel(input_file, header=0)

    print("\n🔍 Preview of your file:")
    print(df.head())
    print(f"\n✅ File loaded! Rows found: {len(df)}")
    print(f"   Columns: {list(df.columns)}\n")

    email_column = "Email Id"
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found. Available columns: {list(df.columns)}")

    df["Validation Result"] = df[email_column].apply(validate_email)

    print("========== EMAIL VALIDATION RESULTS ==========" )
    print(df[["Student Name", email_column, "Validation Result"]].to_string(index=False))

    print("\n---------- SUMMARY ----------")
    print(df["Validation Result"].value_counts().to_string())

    valid_count = (df["Validation Result"] == "✅ Valid").sum()
    invalid_count = len(df) - valid_count
    print(f"\nTotal: {len(df)}  |  Valid: {valid_count}  |  Invalid: {invalid_count}")

    df.to_excel(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
