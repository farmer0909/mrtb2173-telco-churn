import pytest
import pandas as pd
import os

DATA_PATH = 'data/telco_clean.csv'

@pytest.fixture
def df():
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Data file not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def test_no_missing_values(df):
    assert df.isnull().sum().sum() == 0, "Cleaned dataset should have no missing values"

def test_no_duplicate_rows(df):
    assert df.duplicated().sum() == 0, "Cleaned dataset should have no duplicate rows"

def test_totalcharges_is_numeric(df):
    assert df['TotalCharges'].dtype in ['float64', 'float32'], \
        "TotalCharges should be numeric (float)"

def test_churn_column_exists(df):
    assert 'Churn' in df.columns, "Churn column must exist"

def test_tenure_non_negative(df):
    assert (df['tenure'] >= 0).all(), "Tenure values must be non-negative"

def test_required_columns_present(df):
    required = ['customerID', 'Churn', 'tenure', 'MonthlyCharges', 'TotalCharges']
    for col in required:
        assert col in df.columns, f"Required column '{col}' is missing"
