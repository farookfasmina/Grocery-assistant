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


try:
    from rapidfuzz import process
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename uploaded dataset columns to match required names (case-insensitive)."""
    rename_map = {
        "product": "Product Name",
        "productname": "Product Name",
        "item": "Product Name",
        "product name": "Product Name",

        "category": "Category",
        "cat": "Category",

        "discounted price": "Discounted Price (Rs.)",
        "discount_price": "Discounted Price (Rs.)",
        "price": "Discounted Price (Rs.)",

        "original price": "Original Price (Rs.)",
        "orig price": "Original Price (Rs.)",

        "quantity": "Quantity",
        "qty": "Quantity",
        "amount": "Quantity"
    }

    df = df.rename(columns={c: rename_map.get(c.lower().strip(), c) for c in df.columns})
    return df



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


@st.cache_data
def load_default_data():
    try:
        df = pd.read_csv("data/Grocery_data (1).csv")
        return standardize_columns(df)
    except Exception:
        return pd.DataFrame()


if "df" not in st.session_state:
    st.session_state.df = load_default_data()

def require_columns(df: pd.DataFrame, required: list) -> tuple[bool, list]:
    """Check required columns in a case-insensitive way."""
    df_cols = {c.lower(): c for c in df.columns}
    missing = [c for c in required if c.lower() not in df_cols]
    return (len(missing) == 0, missing)



def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    
    if 'Discounted Price (Rs.)' not in df.columns:
        df['Discounted Price (Rs.)'] = np.nan
    if 'Original Price (Rs.)' not in df.columns:
        df['Original Price (Rs.)'] = np.nan
    if 'Quantity' not in df.columns:
        df['Quantity'] = 1
    if 'Category' not in df.columns:
        df['Category'] = "Unknown"

   
    df['Discounted Price (Rs.)'] = pd.to_numeric(df['Discounted Price (Rs.)'], errors='coerce')
    df['Original Price (Rs.)'] = pd.to_numeric(df['Original Price (Rs.)'], errors='coerce')

    df['Quantity'] = df['Quantity'].astype(str).str.extract(r'(\d+\.?\d*)')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(1)

    df['Category'] = df['Category'].astype(str).fillna("Unknown")

    
    df = df[df['Discounted Price (Rs.)'] > 0]

    
    with np.errstate(divide='ignore', invalid='ignore'):
        disc = (df['Original Price (Rs.)'] - df['Discounted Price (Rs.)']) / df['Original Price (Rs.)'] * 100
    disc = disc.replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Discount_Percent'] = disc.clip(0, 100)

    
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


st.sidebar.title("🛒 Grocery App Navigation")
choice = st.sidebar.radio("Go to", [
    "🏠 Home", "ℹ️ About", "📊 Upload & Explore", "📈 EDA",
    "🤖 Train Model", "🔮 Predict Price Level", "🛒 Optimize Grocery List"
])

if choice == "🏠 Home":
   
    st.write("")  
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("assets/shopping.jpg", use_container_width=False, width=350)
        except Exception:
            st.warning("⚠️ Shopping image not found. Please place 'shopping.jpg' inside the 'assets' folder.")


    
    st.markdown(
        """
        <h1 style='text-align:center;'>🛒 Grocery Price Optimization</h1>
        <p style='text-align:center; font-size:1.1rem; color:#475569;'>
            Make smarter, cost-effective grocery decisions.<br>
            Save money and optimize shopping with data-driven insights!
        </p>
        """,
        unsafe_allow_html=True
    )

    
    st.subheader("📌 Project Background")
    st.markdown(
        """
        Grocery shopping is one of the most frequent and necessary household tasks.  
        However, consumers often struggle to compare across brands and find the best deals.  
        With increasing living costs, smarter tools are needed to stretch household budgets.  
        This project leverages **data analysis** and **machine learning**  
        to support cost-effective grocery decisions.
        """
    )

    
    st.subheader("🎯 Final Problem Statement")
    st.markdown(
        """
        **How can grocery items be categorized by price levels and paired with cheaper alternatives,  
        without compromising quality?**  

        The objective is to help households save money while making informed choices.
        """
    )

    
    st.subheader("❓ Gap in Knowledge")
    st.markdown(
        """
        Existing retail platforms display prices but lack **interactive, user-friendly tools** that combine:  

        - ✅ Price optimization  
        - ✅ Predictive modeling of price levels  
        - ✅ Cheaper substitute recommendations  

        This app fills the gap by uniting **EDA, ML models, and optimization** in a single platform.
        """
    )

   
    st.subheader("🔍 Research Questions")
    st.markdown(
        """
        1. Which features (product category, brand, store location, price fluctuations, user preferences, etc.)  
           most significantly influence the total cost savings in grocery shopping?  
        2. How accurately can machine learning and optimization models (e.g., Linear Programming, Integer Programming,  
           K-Nearest Neighbors for substitutes, or NLP-based matching algorithms) identify the cheapest combination  
           of grocery items across multiple stores?  
        3. What is the most interpretable and practical model or algorithm that can be deployed in a  
           consumer-facing Smart Grocery Shopping Assistant to optimize shopping cost and provide useful recommendations?  
        """
    )

    st.subheader("🎯 Research Objectives")
    st.markdown(
        """
        1. To develop an **NLP-based system** that accurately interprets and maps free-text grocery lists  
           to structured product entries across multiple stores.  
        2. To design and implement an **optimization algorithm** that selects the cheapest combination  
           of stores and items to minimize total shopping cost.  
        3. To explore **machine learning techniques** for predicting missing prices and recommending  
           cost-saving substitutes or alternative brands.  
        """
    )

    
    st.subheader("📂 Availability of Data")
    st.markdown(
        """
        The dataset is sourced from publicly available grocery data (CSV format).  
        Users can also **upload their own grocery datasets** for analysis.  
        This ensures flexibility and adaptability across different stores and regions.
        """
    )


elif choice == "ℹ️ About":
    st.title("ℹ️ About this Project")
    st.image("https://cdn-icons-png.flaticon.com/512/2331/2331970.png", width=160, caption="About Project")

    st.markdown("""
    This mini project tackles **Grocery Price Optimization** with **EDA + ML**.  
    We categorize items into **Very Cheap → Very Expensive**, and help users optimize shopping lists.  

    **Stack**: Python, Pandas, Seaborn, Scikit-Learn, Streamlit.
    """)

    st.markdown("---")
    st.subheader("👩‍💻 Group Members")
    st.markdown("""
    | Name        | Student ID        |
    |-------------|------------------|
    | **S.F. Saheela** | ITBIN-2211-0274 |
    | **F.F. Fasmina** | ITBIN-2211-0116 |
    | **M.S. Labeeba** | ITBIN-2211-0215 |
    | **S.M. Sukry**   | ITBIN-2211-0297 |
    """)


elif choice == "📊 Upload & Explore":
    st.title("📊 Upload & Explore Dataset")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

        # 🔹 Use the helper function we defined earlier
        df = standardize_columns(df)

        st.session_state.df = df
        st.success("✅ Dataset uploaded and standardized.")

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
        st.warning(f"⚠️ Need columns: {', '.join(missing or needed)}. Upload a compatible dataset.")
    else:
        # Derive clean DF
        dfe = add_derived_columns(df)
        dfe = dfe.dropna(subset=['Discounted Price (Rs.)'])

        # 1. Price distribution
        st.subheader("Price Distribution (All Products)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(dfe['Discounted Price (Rs.)'], bins=50, kde=True, ax=ax)
        ax.set_title("Distribution of Discounted Prices")
        st.pyplot(fig)

        # 2. Top 10 categories
        st.subheader("Top 10 Categories (by Product Count)")
        fig, ax = plt.subplots(figsize=(10, 5))
        dfe['Category'].value_counts().head(10).plot(kind='bar', ax=ax, color="skyblue")
        ax.set_ylabel("Count")
        ax.set_title("Most Common Categories")
        st.pyplot(fig)

        # 3. Boxplot: Prices by Category
        st.subheader("Price Distribution by Category")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=dfe, x="Category", y="Discounted Price (Rs.)", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title("Boxplot of Prices by Category")
        st.pyplot(fig)

        # 4. Average price per category
        st.subheader("Average Discounted Price per Category")
        avg_prices = dfe.groupby("Category")["Discounted Price (Rs.)"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_prices.plot(kind="barh", ax=ax, color="lightgreen")
        ax.set_xlabel("Average Price (Rs.)")
        ax.set_title("Category-wise Average Price")
        st.pyplot(fig)

        # 5. Discount percentage distribution
        st.subheader("Distribution of Discounts (%)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(dfe["Discount_Percent"], bins=40, kde=True, ax=ax, color="orange")
        ax.set_title("Discount Percentage Distribution")
        st.pyplot(fig)

        # 6. Correlation heatmap
        st.subheader("Correlation Heatmap")
        num_cols = ["Quantity", "Discounted Price (Rs.)", "Original Price (Rs.)", "Discount_Percent"]
        corr = dfe[num_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation between Numerical Features")
        st.pyplot(fig)



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
