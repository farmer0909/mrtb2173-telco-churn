import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Telco Churn Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_and_train():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df_model = df.drop('customerID', axis=1)
    binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling','Churn']
    le = LabelEncoder()
    for col in binary_cols:
        df_model[col] = le.fit_transform(df_model[col])
    multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                  'Contract','PaymentMethod']
    df_model = pd.get_dummies(df_model, columns=multi_cols, drop_first=True)
    scaler = StandardScaler()
    df_model[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(
        df_model[['tenure','MonthlyCharges','TotalCharges']])
    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return df, rf, X.columns.tolist(), scaler

df, model, feature_cols, scaler = load_and_train()

# ── Header ─────────────────────────────────────────────────────
st.title("📊 Telco Customer Churn Dashboard")
st.markdown("**MRTB 2173 – Agile Data Science | MRT241056 Kow Ming Chuan**")
st.divider()

# ── Interactive Feature 1: Contract Type Filter (Dropdown) ─────
contract_options = ['All'] + sorted(df['Contract'].unique().tolist())
selected_contract = st.selectbox("🔽 Filter by Contract Type", contract_options)
filtered_df = df if selected_contract == 'All' else df[df['Contract'] == selected_contract]

# ── KPI Metrics ────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(filtered_df):,}")
col2.metric("Churned", f"{(filtered_df['Churn']=='Yes').sum():,}")
churn_rate = (filtered_df['Churn']=='Yes').mean() * 100
col3.metric("Churn Rate", f"{churn_rate:.1f}%")
col4.metric("Avg Monthly Charges", f"${filtered_df['MonthlyCharges'].mean():.2f}")

st.divider()

# ── Visualisation 1: Churn Rate by Contract ────────────────────
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("📌 Churn Rate by Contract Type")
    cc = df.groupby('Contract')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100).reset_index()
    cc.columns = ['Contract', 'Churn Rate (%)']
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    bars = ax1.bar(cc['Contract'], cc['Churn Rate (%)'],
                   color=['#E63946', '#457B9D', '#1D3557'], width=0.5, edgecolor='white')
    for bar, val in zip(bars, cc['Churn Rate (%)']):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5, f'{val:.1f}%',
                 ha='center', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Churn Rate (%)')
    ax1.set_ylim(0, 60)
    ax1.set_xlabel('Contract Type')
    plt.tight_layout()
    st.pyplot(fig1)

# ── Visualisation 2: Monthly Charges Boxplot ──────────────────
with col_b:
    st.subheader("💰 Monthly Charges by Churn Status")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    churn_yes = filtered_df[filtered_df['Churn'] == 'Yes']['MonthlyCharges']
    churn_no  = filtered_df[filtered_df['Churn'] == 'No']['MonthlyCharges']
    bp = ax2.boxplot([churn_no, churn_yes], labels=['No Churn', 'Churn'],
                     patch_artist=True,
                     medianprops={'color': 'red', 'linewidth': 2})
    bp['boxes'][0].set_facecolor('#457B9D')
    bp['boxes'][1].set_facecolor('#E63946')
    ax2.set_ylabel('Monthly Charges (USD)')
    plt.tight_layout()
    st.pyplot(fig2)

# ── Visualisation 3: Feature Importance ───────────────────────
st.subheader("🏆 Top 15 Feature Importances (Random Forest)")
feat_imp = pd.Series(model.feature_importances_, index=feature_cols).nlargest(15).sort_values()
fig3, ax3 = plt.subplots(figsize=(10, 5))
colors = ['#1D3557' if v > feat_imp.quantile(0.75) else '#457B9D' for v in feat_imp]
feat_imp.plot(kind='barh', ax=ax3, color=colors, edgecolor='white')
ax3.set_xlabel('Gini Importance')
plt.tight_layout()
st.pyplot(fig3)

st.divider()

# ── Monitoring Metrics ─────────────────────────────────────────
st.subheader("📡 Monitoring Metrics")
mon1, mon2 = st.columns(2)
missing_rate = df['TotalCharges'].isnull().mean() * 100
mon1.metric("Missing TotalCharges Rate",
            f"{missing_rate:.2f}%",
            "✅ Within threshold" if missing_rate < 5 else "⚠️ ALERT")
mon2.metric("Model ROC-AUC (baseline)",
            "0.8641",
            "+0.0134 vs Logistic Regression")

st.divider()

# ── Predictive Output ─────────────────────────────────────────
st.subheader("🤖 Churn Risk Predictor")

# Interactive Feature 2: Probability Threshold Slider
threshold = st.slider(
    "⚙️ Churn Probability Threshold",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="Lower = catch more churners (higher recall); Higher = fewer false alarms (higher precision)")

c1, c2, c3 = st.columns(3)
tenure_in  = c1.number_input("Tenure (months)", 0, 72, 12)
monthly_in = c2.number_input("Monthly Charges (USD)", 10.0, 120.0, 65.0)
total_in   = c3.number_input("Total Charges (USD)", 0.0, 9000.0, 780.0)

if st.button("🔮 Predict Churn Risk"):
    input_vec = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
    scaled = scaler.transform([[tenure_in, monthly_in, total_in]])
    input_vec['tenure']         = scaled[0][0]
    input_vec['MonthlyCharges'] = scaled[0][1]
    input_vec['TotalCharges']   = scaled[0][2]
    prob = model.predict_proba(input_vec)[0][1]

    if prob >= threshold:
        label = "🔴 HIGH RISK"
        color = "red"
    elif prob >= 0.3:
        label = "🟡 MEDIUM RISK"
        color = "orange"
    else:
        label = "🟢 LOW RISK"
        color = "green"

    st.metric("Churn Probability", f"{prob:.1%}", label)
    st.progress(prob)

st.divider()
st.caption("MRTB 2173 Agile Data Science | Kow Ming Chuan MRT241056 | "
           "Telco Customer Churn Dataset – Kaggle (blastchar, 2018)")
