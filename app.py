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
# 🏠 HOME
# -------------------------------
if choice == "🏠 Home":
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(120deg, #e0e7ff 0%, #f0fdfa 100%) !important;
        }
        .home-card {
            background: #ffffffee;
            border-radius: 24px;
            padding: 2.5rem 2rem 2rem 2rem;
            margin-top: 2rem;
            box-shadow: 0 8px 32px 0 #00000022;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }
        .home-title {
            font-size: 2.7rem;
            font-weight: 700;
            color: #1e293b;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .home-sub {
            font-size: 1.25rem;
            color: #475569;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .feature-list {
            font-size: 1.13rem;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        .feature-list li {
            margin-bottom: 0.7rem;
        }
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #0f172a;
            margin-top: 2rem;
        }
        .section-text {
            font-size: 1.05rem;
            color: #374151;
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Centered logo if available
    logo_path = "assets/shopping.jpg"
    cols = st.columns([1, 2, 1])
    with cols[1]:
        try:
            st.image(logo_path, width=320)
        except Exception:
            st.write("")

    # Main Home Card with sections
    st.markdown("""
        <div class="home-card">
            <div class="home-title">🛒 Grocery Price Optimization App</div>
            <div class="home-sub">
                Make smarter, cost-effective grocery decisions.<br>
                <span style="color:#16a34a;font-weight:600;">Save money</span> and 
                <span style="color:#0ea5e9;font-weight:600;">optimize your shopping</span> with data-driven insights!
            </div>
            <ul class="feature-list">
                <li>📊 <b>Upload & Explore</b> your grocery datasets</li>
                <li>📈 <b>Visualize</b> price trends and categories</li>
                <li>🤖 <b>Train ML models</b> to predict price levels</li>
                <li>🔮 <b>Predict</b> price level for new products</li>
                <li>🛒 <b>Optimize</b> your grocery list with cheaper substitutes</li>
            </ul>

            <div class="section-title">📌 Project Background</div>
            <div class="section-text">
                Grocery shopping is a frequent and essential activity for households. 
                However, consumers often face challenges in identifying the best deals, 
                comparing across brands, and making cost-effective decisions. 
                With rising living costs, leveraging data-driven insights can help optimize grocery expenses.
            </div>

            <div class="section-title">🎯 Final Problem Statement</div>
            <div class="section-text">
                How can we design a system that categorizes grocery items by price levels and 
                provides recommendations for cheaper yet similar alternatives, 
                enabling households to make informed, cost-saving choices?
            </div>

            <div class="section-title">❓ Gap in Knowledge</div>
            <div class="section-text">
                While many e-commerce and retail platforms display prices, there is a lack of 
                user-friendly tools that combine <b>price optimization, predictive modeling, 
                and substitution suggestions</b>. This project bridges that gap by integrating 
                <i>EDA, ML models, and optimization strategies</i> into a single interactive app.
            </div>

            <div style="text-align:center; margin-top:2rem;">
                <span style="color:#64748b;">Use the sidebar to get started!</span>
            </div>
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
