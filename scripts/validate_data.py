import pandas as pd
import sys

def validate_dataset(filepath):
    df = pd.read_csv(filepath)
    errors = []

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        errors.append(f"Missing values found: {missing.to_dict()}")

    dupes = df.duplicated().sum()
    if dupes > 0:
        errors.append(f"Duplicate rows found: {dupes}")

    required = ['customerID', 'Churn', 'tenure', 'MonthlyCharges', 'TotalCharges']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if (df['tenure'] < 0).any():
        errors.append("Negative tenure values detected")

    print("=" * 50)
    print("AUTOMATED DATA VALIDATION REPORT")
    print("=" * 50)
    if errors:
        print("STATUS: VALIDATION FAILED")
        for e in errors:
            print(f"  [FAIL] {e}")
        sys.exit(1)
    else:
        print("STATUS: ALL CHECKS PASSED")

if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'data/telco_clean.csv'
    validate_dataset(filepath)
