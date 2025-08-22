# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Optional fuzzy matching
try:
    from rapidfuzz import process
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

# -------------------------------
# 🎨 Page Config & Styling
# -------------------------------
st.set_page_config(page_title="🛒 Grocery Price Optimization", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #1e1e2f; color: white; }
    [data-testid="stSidebar"] * { color: white !important; }
    .stApp { background-color: #f7f9fc; }
    h1, h2, h3 { color: #2c3e50; }
    .small-note { color:#6b7280; font-size:0.9rem; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 📂 Load / Hold Dataset
# -------------------------------
@st.cache_data
def load_default_data():
    try:
        return pd.read_csv("data/Grocery_data (1).csv")
    except Exception:
        return pd.DataFrame()

if "df" not in st.session_state:
    st.session_state.df = load_default_data()

def require_columns(df: pd.DataFrame, required: list) -> tuple[bool, list]:
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)

# -------------------------------
# 🧰 Preprocessing Helpers
# -------------------------------
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required columns exist
    if 'Discounted Price (Rs.)' not in df.columns:
        df['Discounted Price (Rs.)'] = np.nan
    if 'Original Price (Rs.)' not in df.columns:
        df['Original Price (Rs.)'] = np.nan
    if 'Quantity' not in df.columns:
        df['Quantity'] = 1
    if 'Category' not in df.columns:
        df['Category'] = "Unknown"

    # Conversions
    df['Discounted Price (Rs.)'] = pd.to_numeric(df['Discounted Price (Rs.)'], errors='coerce')
    df['Original Price (Rs.)'] = pd.to_numeric(df['Original Price (Rs.)'], errors='coerce')

    df['Quantity'] = df['Quantity'].astype(str).str.extract(r'(\d+\.?\d*)')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(1)

    df['Category'] = df['Category'].astype(str).fillna("Unknown")

    # Clean rows
    df = df[df['Discounted Price (Rs.)'] > 0]

    # Discount %
    with np.errstate(divide='ignore', invalid='ignore'):
        disc = (df['Original Price (Rs.)'] - df['Discounted Price (Rs.)']) / df['Original Price (Rs.)'] * 100
    disc = disc.replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Discount_Percent'] = disc.clip(0, 100)

    # Price bins
    bins = [0, 50, 200, 500, 1000, np.inf]
    labels = ["Very Cheap", "Cheap", "Medium", "Expensive", "Very Expensive"]
    df['Price_Level'] = pd.cut(df['Discounted Price (Rs.)'], bins=bins, labels=labels, include_lowest=True)

    return df

def build_train_matrix(df: pd.DataFrame):
    df = add_derived_columns(df)
    keep_cols = ['Category', 'Quantity', 'Discount_Percent', 'Price_Level']
    df = df[keep_cols].dropna()

    le = LabelEncoder()
    df['Category_enc'] = le.fit_transform(df['Category'])

    X = df[['Category_enc', 'Quantity', 'Discount_Percent']]
    y = df['Price_Level']

    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask], le

def prepare_single_features(category: str, quantity: float, discount_percent: float, le: LabelEncoder):
    if category not in le.classes_:
        if "Unknown" in le.classes_:
            category = "Unknown"
        else:
            category = le.classes_[0]
    cat_enc = le.transform([category])[0]
    return pd.DataFrame([[cat_enc, quantity, discount_percent]],
                        columns=['Category_enc', 'Quantity', 'Discount_Percent'])

# -------------------------------
# 📍 Sidebar Navigation
# -------------------------------
st.sidebar.title("🛒 Grocery App Navigation")
choice = st.sidebar.radio("Go to", [
    "🏠 Home", "ℹ️ About", "📊 Upload & Explore", "📈 EDA",
    "🤖 Train Model", "🔮 Predict Price Level", "🛒 Optimize Grocery List"
])

if choice == "🏠 Home":
    st.markdown(
        """
        <style>
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #0f172a;
            margin-top: 1.2rem;
        }
        .section-text {
            font-size: 1rem;
            color: #334155;
            margin-top: 0.3rem;
            text-align: justify;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Centered image
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.image("assets/shopping.jpg", width=280)

    st.markdown(
        """
        <h1 style='text-align:center;'>🛒 Grocery Price Optimization</h1>
        <p style='text-align:center; font-size:1.1rem; color:#475569;'>
            Make smarter, cost-effective grocery decisions.<br>
            Save money and optimize shopping with data-driven insights!
        </p>

        <div class="section-title">📌 Project Background</div>
        <div class="section-text">
            Grocery shopping is one of the most frequent and necessary household tasks.  
            However, consumers often struggle to compare across brands and find the best deals.  
            With increasing living costs, smarter tools are needed to stretch household budgets.  
            This project leverages <b>data analysis</b> and <b>machine learning</b>  
            to support cost-effective grocery decisions.
        </div>

        <div class="section-title">🎯 Final Problem Statement</div>
        <div class="section-text">
            <b>How can grocery items be categorized by price levels and paired with cheaper alternatives,  
            without compromising quality?</b>  
            The objective is to help households save money while making informed choices.
        </div>

        <div class="section-title">❓ Gap in Knowledge</div>
        <div class="section-text">
            Existing retail platforms display prices but lack <b>interactive, user-friendly tools</b> that combine:<br>  
            ✅ Price optimization<br>
            ✅ Predictive modeling of price levels<br>
            ✅ Cheaper substitute recommendations  
            <br>
            This app fills the gap by uniting <b>EDA, ML models, and optimization</b> in a single platform.
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------
# ℹ️ ABOUT
# -------------------------------
elif choice == "ℹ️ About":
    st.title("ℹ️ About this Project")
    st.markdown("""
    This mini project tackles **Grocery Price Optimization** using:
    - 📊 **EDA** for insights  
    - 🤖 **ML models** to categorize products  
    - 🛒 **Optimization** to suggest cheaper substitutes  

    **Stack**: Python, Pandas, Seaborn, Scikit-Learn, Streamlit
    """)

# -------------------------------
# 📊 UPLOAD & EXPLORE
# -------------------------------
elif choice == "📊 Upload & Explore":
    st.title("📊 Upload & Explore Dataset")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        st.session_state.df = pd.read_csv(uploaded)
        st.success("✅ Dataset uploaded.")

    df = st.session_state.df
    if not df.empty:
        st.write("### Preview", df.head())
        st.write("### Summary", df.describe(include="all").transpose())
    else:
        st.info("👉 Upload a CSV to continue.")

# -------------------------------
# 📈 EDA
# -------------------------------
elif choice == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")
    df = st.session_state.df
    needed = ['Category', 'Discounted Price (Rs.)']
    ok, missing = require_columns(df, needed)

    if df.empty or not ok:
        st.warning(f"⚠️ Need columns: {', '.join(missing or needed)}")
    else:
        dfe = add_derived_columns(df)

        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(dfe['Discounted Price (Rs.)'], bins=50, kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Top Categories")
        fig, ax = plt.subplots()
        dfe['Category'].value_counts().head(10).plot(kind='bar', ax=ax)
        st.pyplot(fig)

# -------------------------------
# 🤖 TRAIN MODEL
# -------------------------------
elif choice == "🤖 Train Model":
    st.title("🤖 Train ML Models")
    df = st.session_state.df
    ok, missing = require_columns(df, ['Category', 'Quantity', 'Discounted Price (Rs.)'])

    if df.empty or not ok:
        st.warning("⚠️ Upload dataset with Category, Quantity, Discounted Price.")
    else:
        X, y, le = build_train_matrix(df)
        if X.empty:
            st.error("No valid training data.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

            rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
            lr = LogisticRegression(max_iter=2000).fit(X_train, y_train)

            rf_acc = accuracy_score(y_test, rf.predict(X_test))
            lr_acc = accuracy_score(y_test, lr.predict(X_test))

            st.write(f"🌲 Random Forest: {rf_acc:.3f}")
            st.write(f"📊 Logistic Regression: {lr_acc:.3f}")

            best_model = rf if rf_acc >= lr_acc else lr
            os.makedirs("outputs", exist_ok=True)
            bundle = {"model": best_model, "label_encoder": le, "classes": list(le.classes_)}
            joblib.dump(bundle, "outputs/best_grocery_model.pkl")
            st.success("✅ Model trained and saved locally.")

# -------------------------------
# 🔮 PREDICT PRICE LEVEL
# -------------------------------
elif choice == "🔮 Predict Price Level":
    st.title("🔮 Predict Price Level")
    try:
        bundle = joblib.load("outputs/best_grocery_model.pkl")
        model, le, classes = bundle["model"], bundle["label_encoder"], bundle["classes"]
    except Exception:
        st.error("⚠️ Train a model first.")
        model, le, classes = None, None, None

    if model:
        category = st.selectbox("Category", classes)
        qty = st.number_input("Quantity", min_value=1.0, value=1.0)
        disc = st.slider("Discount %", 0, 100, 0)
        if st.button("Predict"):
            X = prepare_single_features(category, qty, disc, le)
            pred = model.predict(X)[0]
            st.success(f"Predicted Price Level: **{pred}**")

# -------------------------------
# 🛒 OPTIMIZE GROCERY LIST
# -------------------------------
elif choice == "🛒 Optimize Grocery List":
    st.title("🛒 Optimize Grocery List")
    df = st.session_state.df
    needed = ['Product Name', 'Category', 'Discounted Price (Rs.)']
    ok, missing = require_columns(df, needed)

    if df.empty or not ok:
        st.warning(f"⚠️ Need columns: {', '.join(missing or needed)}")
    else:
        dfo = add_derived_columns(df)
        dfo = dfo.dropna(subset=['Product Name', 'Category', 'Discounted Price (Rs.)'])
        product_list = dfo['Product Name'].astype(str).unique().tolist()

        if not HAS_RAPIDFUZZ:
            st.warning("For best matching, install rapidfuzz: `pip install rapidfuzz`")

        def match_item(user_item, product_list):
            if HAS_RAPIDFUZZ:
                matches = process.extract(str(user_item), product_list, limit=1)
                if matches:
                    best_match, score, _ = matches[0]
                    return best_match, score
            user_item = str(user_item).lower()
            candidates = [p for p in product_list if user_item in p.lower()]
            if candidates:
                return candidates[0], 100
            return None, 0

        def find_substitute(matched_product):
            try:
                cat = dfo.loc[dfo['Product Name'] == matched_product, 'Category'].values[0]
                cat_items = dfo[dfo['Category'] == cat]
                cheapest = cat_items.loc[cat_items['Discounted Price (Rs.)'].idxmin()]
                if cheapest['Product Name'] != matched_product:
                    return cheapest['Product Name'], float(cheapest['Discounted Price (Rs.)'])
            except Exception:
                return None, None
            return None, None

        user_input = st.text_area("Enter grocery items (comma-separated)", "2kg rice, milk, bread, biscuits")

        if st.button("Optimize List"):
            items = [i.strip() for i in user_input.split(",") if i.strip()]
            results, total = [], 0.0

            for item in items:
                match, score = match_item(item, product_list)
                if match:
                    price = dfo.loc[dfo['Product Name'] == match, 'Discounted Price (Rs.)'].mean()
                    sub_name, sub_price = find_substitute(match)

                    if pd.notna(price):
                        if sub_name and (sub_price is not None) and sub_price < price:
                            chosen, chosen_price, sub_used = sub_name, sub_price, "Yes"
                        else:
                            chosen, chosen_price, sub_used = match, float(price), "No"

                        results.append([item, match, round(float(price), 2),
                                        chosen, round(float(chosen_price), 2), sub_used])
                        total += chosen_price

            if results:
                res_df = pd.DataFrame(results, columns=[
                    "User Input", "Matched Product", "Matched Price",
                    "Chosen Product", "Final Price", "Used Substitute"
                ])
                st.dataframe(res_df)
                st.success(f"💰 Total Optimized Cost: Rs. {round(total, 2)}")

                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Optimized List (CSV)", data=csv,
                                   file_name="optimized_grocery_list.csv", mime="text/csv")
            else:
                st.info("No items matched. Try generic names like 'rice', 'milk'.")
