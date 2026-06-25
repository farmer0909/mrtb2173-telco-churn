# MRTB 2173 – Agile Data Science PMA

**Student:** Kow Ming Chuan | **Matric:** MRT241056  
**Course:** MRTB 2173 Agile Data Science | Semester 2, Session 2025/2026  
**Dataset:** [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Project Structure

```
mrtb2173-telco-churn/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← download from Kaggle
├── notebooks/
│   └── MRTB2173_PMA_Notebook.ipynb             ← main Colab notebook
├── scripts/
│   └── validate_data.py                         ← automated validation
├── tests/
│   └── test_pipeline.py                         ← pytest unit tests
├── app/
│   └── streamlit_app.py                         ← Streamlit dashboard
└── .github/
    └── workflows/
        └── ci.yml                               ← GitHub Actions CI/CD
```

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas scikit-learn matplotlib seaborn streamlit pytest
```

### 2. Run data validation
```bash
python scripts/validate_data.py data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Run unit tests
```bash
pytest tests/ -v
```

### 4. Launch Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## CI/CD Pipeline

Every push to `main` triggers GitHub Actions to automatically:
1. Set up Python 3.10 environment
2. Install all dependencies
3. Run `validate_data.py` on the cleaned dataset
4. Run all 6 pytest unit tests

---

## Sprint Summary

| Sprint | Focus | Deliverable |
|--------|-------|-------------|
| Sprint 1 | Data cleaning & EDA | Cleaned dataset, quality checks |
| Sprint 2 | Baseline model | Logistic Regression (AUC: 0.8507, Recall: 0.56) |
| Sprint 3 | Improved model | Random Forest balanced (AUC: 0.8641, Recall: 0.73) |
| Sprint 4 | Monitoring & dashboard | Streamlit app, PSI drift analysis |

---

## Dataset

- **Source:** Kaggle – blastchar/telco-customer-churn
- **Records:** 7,043 customers
- **Features:** 21 columns
- **Target:** `Churn` (Yes/No) – 26.5% churn rate
- **Problem:** Binary classification to predict customer churn

---

*MRTB 2173 Agile Data Science | Universiti Teknologi Malaysia | Faculty of Artificial Intelligence*
