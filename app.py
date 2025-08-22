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

# -------------------------------
# 🏠 HOME (full version)
# -------------------------------
if choice == "🏠 Home":
    st.markdown("""
        <style>
        .home-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 2.5rem 2rem;
            margin-top: 2rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            text-align: center;
        }
        .home-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.5rem;
        }
        .home-sub {
            font-size: 1.2rem;
            color: #475569;
            margin-bottom: 2rem;
        }
        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #0f172a;
            margin-top: 1.8rem;
        }
        .section-text {
            font-size: 1rem;
            color: #334155;
            margin-top: 0.3rem;
            text-align: justify;
        }
        .feature-list {
            text-align: left;
            margin-top: 1.5rem;
            font-size: 1.05rem;
            color: #1e293b;
        }
        .feature-list li {
            margin-bottom: 0.6rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Center image
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.image("assets/shopping.jpg", width=280)

    # Card with explanations + features
    st.markdown("""
        <div class="home-card">
            <div class="home-title">🛒 Grocery Price Optimization</div>
            <p class="home-sub">
                Make smarter, cost-effective grocery decisions.<br>
                Save money and optimize shopping with data-driven insights!
            </p>

            <div class="section-title">📌 Project Background</div>
            <div class="section-text">
                Grocery shopping is one of the most frequent and necessary activities for households.  
                However, many consumers struggle to identify the best deals, compare products across brands,  
                and make cost-effective purchase decisions. With the rise in living costs,  
                families need smarter ways to stretch their budgets.  
                This project explores how **data analysis and machine learning** can help consumers  
                make smarter, data-driven grocery choices.
            </div>

            <div class="section-title">🎯 Final Problem Statement</div>
            <div class="section-text">
                The key challenge is:  
                <b>How can we categorize grocery items by price levels and suggest cheaper yet similar alternatives?</b>  
                The goal is to enable households to make informed, cost-saving choices without compromising  
                on quality or variety.
            </div>

            <div class="section-title">❓ Gap in Knowledge</div>
            <div class="section-text">
                While many retail platforms display prices and discounts,  
                there is a lack of **interactive tools** that combine:  
                <ul style="text-align:left;">
                    <li>✅ Price optimization</li>
                    <li>✅ Predictive modeling of price levels</li>
                    <li>✅ Automatic substitution suggestions</li>
                </ul>
                This app bridges that gap by integrating **EDA, ML models,  
                and grocery list optimization** into a single interactive platform.
            </div>

            <div class="section-title">✨ Key Features of this App</div>
            <ul class="feature-list">
                <li>📊 Upload & Explore your grocery dataset</li>
                <li>📈 Visualize price trends and categories (EDA)</li>
                <li>🤖 Train ML models to predict price levels</li>
                <li>🔮 Predict price level for new grocery items</li>
                <li>🛒 Optimize your grocery list with cheaper substitutes</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


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
