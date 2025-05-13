# ==============================================
# 🌈 THEME SETTINGS: Light/Dark Mode Toggle + Background
# ==============================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Sidebar theme switch
st.sidebar.title("🧰 App Settings")
theme_choice = st.sidebar.radio("Select Theme:", ["Dark", "Light"], index=0)
sns_theme = "darkgrid" if theme_choice == "Dark" else "whitegrid"
plot_bg = "#0e1117" if theme_choice == "Dark" else "#ffffff"
text_color = "#ffffff" if theme_choice == "Dark" else "#000000"
sns.set_style(sns_theme)

# 🔲 Custom background injection
# Theme colors
bg_color = "#0e1117" if theme_choice == "Dark" else "#ffffff"
text_color = "#ffffff" if theme_choice == "Dark" else "#000000"
custom_style = f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .css-1d391kg, .css-10trblm, .css-1v0mbdj p,
    .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3 {{
        color: {text_color};
    }}
    </style>
"""


st.markdown(custom_style, unsafe_allow_html=True)

st.title("📊 Customer Behavior Analytics Dashboard")
st.markdown("""
<span style='font-size:18px; font-weight:500;'>👥 <b>Members:</b> Ahmed Bilal Nazim & Ahmed Ali Khan</span>
""", unsafe_allow_html=True)


# Set global Seaborn and Matplotlib style
sns.set_style(sns_theme)
plt.rcParams.update({
    'axes.facecolor': plot_bg,
    'axes.edgecolor': text_color,
    'axes.labelcolor': text_color,
    'xtick.color': text_color,
    'ytick.color': text_color,
    'text.color': text_color,
    'figure.facecolor': plot_bg,
    'legend.edgecolor': text_color
})


# ==============================================
# 📁 STEP 1: Upload CSV File
# ==============================================
uploaded_file = st.file_uploader("Upload your marketing_campaign.csv", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file, sep='\t')
    st.subheader("📄 Original Uploaded Data Sample")
    st.dataframe(df_raw.head())

    st.write(f"🔎 Original shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    st.write(f"🧭 Missing 'Income' values: {df_raw['Income'].isnull().sum()}")

    # ==============================================
    # 🧹 STEP 2: Data Preprocessing
    # ==============================================
    df = df_raw.copy()
    st.subheader("🧹 Data Preprocessing Steps")

    df = df.dropna(subset=["Income"])
    st.write(f"✅ Dropped rows with missing Income — New shape: {df.shape}")

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    st.write("✅ Converted 'Dt_Customer' to datetime")

    df["Customer_Since_Days"] = (pd.to_datetime("today") - df["Dt_Customer"]).dt.days
    st.write("✅ Created new feature: 'Customer_Since_Days'")

    st.subheader("🧼 Cleaned Data Sample")
    st.dataframe(df.head())

    # ==============================================
    # 📊 STEP 3: Clustering (KMeans)
    # ==============================================
    st.subheader("📊 Customer Segmentation (KMeans Clustering)")
    
    # Define features based on notebook
    features = ['Income', 'Recency', 'Customer_Since_Days',
                'MntWines', 'MntMeatProducts', 'MntFishProducts',
                'MntSweetProducts', 'MntGoldProds',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    st.write("📌 Cluster Group Summary")
    st.dataframe(df.groupby("Cluster")[features].mean())
    
    # ==============================================
    # 🌍 2D Cluster Visualization (Notebook Style)
    # ==============================================
    from sklearn.decomposition import PCA
    
    st.subheader("🌍 2D Cluster Visualization")
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=df['Cluster'], palette='Set2', ax=ax)
    ax.set_title("Customer Segments Visualized with PCA (2D)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(title='Cluster')
    st.pyplot(fig)
    
    # Explanation
    st.markdown("""
    **🧠 About PCA Plot**  
    - PCA (Principal Component Analysis) reduces multi-dimensional data into 2D.  
    - Each point represents a customer.  
    - Color indicates the customer cluster assigned by KMeans.  
    - This helps us visualize group behavior patterns and separability in a simplified 2D space.
    """)


    # ==============================================
    # 🔭 3D View of Customer Segments (Notebook Style)
    # ==============================================
    from mpl_toolkits.mplot3d import Axes3D  # Ensure this import is at the top
    
    st.subheader("🔭 3D View of Customer Segments")
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use PCA components from earlier
    ax.scatter(
        pca_components[:, 0],
        pca_components[:, 1],
        df["Customer_Since_Days"],
        c=df["Cluster"],
        cmap='Set2',
        s=50,
        alpha=0.6
    )
    
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("Customer Since Days")
    ax.set_title("3D View of Customer Segments")
    st.pyplot(fig)
    
    # Explanation
    st.markdown("""
    **🧠 About 3D Cluster Plot**  
    - The X and Y axes are the PCA components used to reduce dimensionality.  
    - The Z-axis shows how long a customer has been with the company.  
    - Colors represent KMeans clusters, helping visualize group trends in 3D space.
    """)



    # ==============================================
    # 📈 STEP 4: Exploratory Visual Analysis
    # ==============================================
    st.subheader("📈 Exploratory Visual Analysis")
    df_encoded = pd.get_dummies(df, columns=["Education", "Marital_Status"], drop_first=True)
    spending_cols = ['MntWines', 'MntMeatProducts', 'MntFishProducts', 'MntGoldProds']

    # Histograms
    st.markdown("### 🧺 Distribution of Spending Features")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    for i, col in enumerate(spending_cols):
        axs[i].hist(df[col], bins=20, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{col} Distribution")
    plt.tight_layout()
    st.pyplot(fig)
    # Explanation for histogram
    st.markdown("""
    **🧾 About Spending Histograms**  
    - Shows how frequently customers purchase each product category.  
    - Skewness or clustering indicates popular vs. niche spending behavior.  
    - Helps understand customer preferences by product type.
    """)


    # Heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_encoded[spending_cols + ['Income']].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("""
    **🔍 About Correlation Heatmap**  
    - Displays correlation between income and spending variables.  
    - Strong positive values show high co-purchase behavior.  
    - Useful to detect bundled buying or related preferences.
    """)


    # Bar Plots
    st.subheader("🎯 Campaign Response by Category")
    st.markdown("#### 📚 Response Rate by Education")
    edu_response = df.groupby("Education")["Response"].mean().sort_values()
    fig, ax = plt.subplots()
    sns.barplot(x=edu_response.index, y=edu_response.values, palette="Blues_d", ax=ax)
    plt.ylabel("Response Rate")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("""
    **🎓 Interpretation of Education Response Barplot**  
    - Highlights how education level influences marketing campaign response.  
    - Bars show average response rate per education group.  
    - Helps tailor messaging for target demographics.
    """)


    st.markdown("#### 💍 Response Rate by Marital Status")
    marital_response = df.groupby("Marital_Status")["Response"].mean().sort_values()
    fig, ax = plt.subplots()
    sns.barplot(x=marital_response.index, y=marital_response.values, palette="Greens_d", ax=ax)
    plt.ylabel("Response Rate")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("""
    **💑 Interpretation of Marital Status Response Barplot**  
    - Compares campaign engagement across relationship types.  
    - Identifies which demographic is more receptive to promotions.
    """)


    # Boxplot
    st.subheader("📦 Spending Patterns by Product Category")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df[spending_cols], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("""
    **📦 About Boxplot of Product Spending**  
    - Shows distribution and variability in customer spending.  
    - Boxes indicate interquartile range; dots show outliers.  
    - Helps find which products have consistent vs. erratic purchases.
    """)


    # ==============================================
    # 🔍 3D Plot: Customer Distribution by Response
    # ==============================================
    st.subheader("📌 Customer Distribution by Response (3D View)")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors = df['Response'].map({0: 'blue', 1: 'red'})
    ax.scatter(
        df["Income"] / df["Income"].max(),
        df["Recency"] / df["Recency"].max(),
        df["Customer_Since_Days"] / df["Customer_Since_Days"].max(),
        c=colors,
        alpha=0.6
    )
    ax.set_xlabel("Income")
    ax.set_ylabel("Recency")
    ax.set_zlabel("Customer Since Days")
    ax.set_title("Customer Distribution by Response (3D View)")
    st.pyplot(fig)

    st.markdown("""
    **📌 About 3D Customer Response View**  
    - Shows customer response (red = yes, blue = no) in relation to income, recency, and tenure.  
    - Reveals patterns in response likelihood.  
    - Helpful for visualizing segmentation impact.
    """)



    # ==============================================
    # 🔁 STEP 5: Association Rule Mining
    # ==============================================
    st.subheader("🧠 Association Rule Mining (Product Bundling Insights)")

    product_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    basket = df[product_cols].applymap(lambda x: 1 if x > 0 else 0)

    # ✅ Generate frequent itemsets (leave itemsets as frozenset)
    frequent_items = apriori(basket, min_support=0.05, use_colnames=True)

    # Copy for display only (convert itemsets to string)
    frequent_items_display = frequent_items.copy()
    frequent_items_display["itemsets"] = frequent_items_display["itemsets"].apply(lambda x: ', '.join(list(x)))

    st.write("🔍 Top Frequent Itemsets")
    st.dataframe(frequent_items_display.sort_values(by="support", ascending=False).head(10))

    # ✅ Generate association rules using actual frozenset itemsets
    rules = association_rules(frequent_items, metric="confidence", min_threshold=0.3)
    rules = rules.sort_values(by='lift', ascending=False)

    # Convert frozensets in rules to strings for display
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    # 📋 Show rules
    st.subheader("📏 Strongest Association Rules")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))


    # 📊 Visualize support vs confidence
    st.subheader("📊 Rule Metrics: Support vs Confidence")
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.7, c=rules['lift'], cmap='viridis')
    plt.colorbar(label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Association Rules: Support vs Confidence')
    st.pyplot(fig)

    st.markdown("""
    **📈 About Support vs. Confidence Plot**  
    - Each point is a product rule (if buy X → then buy Y).  
    - Higher support = more frequent combo; higher confidence = more reliable rule.  
    - Color intensity = lift (strength of the rule).
    """)


    # 3D Plot – Association Rule Metrics

    st.subheader("📊 3D Plot of Association Rules")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(rules['support'], rules['confidence'], rules['lift'],
                    c=rules['lift'], cmap='coolwarm', alpha=0.8)
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_zlabel("Lift")
    ax.set_title("3D Plot of Association Rules")
    st.pyplot(fig)

    st.markdown("""
    **🧠 About 3D Association Rule View**  
    - Adds a third dimension (Lift) to better compare rule strength.  
    - Useful when evaluating multiple bundling strategies together.
    """)



    # ==============================================
    # 🤖 STEP 6: Classification + ROC + Expanders
    # ==============================================
    st.subheader("🔮 Predict Campaign Response (Compare Multiple Models)")
    X = df_encoded.drop(columns=['ID', 'Dt_Customer', 'Response'], errors='ignore')
    y = df_encoded['Response']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_options = ["Decision Tree", "Naive Bayes", "KNN"]
    selected_models = st.multiselect("Choose model(s) to evaluate", model_options, default=model_options[:1])

    def evaluate_model(model, name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result = {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 2),
            "Precision": round(precision_score(y_test, y_pred), 2),
            "Recall": round(recall_score(y_test, y_pred), 2),
            "F1 Score": round(f1_score(y_test, y_pred), 2),
            "y_pred": y_pred,
            "model": model
        }
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            result["fpr"] = fpr
            result["tpr"] = tpr
            result["auc"] = auc(fpr, tpr)
        except:
            result["fpr"] = None
            result["tpr"] = None
            result["auc"] = None
        return result

    results = []
    for name in selected_models:
        if name == "Decision Tree":
            clf = DecisionTreeClassifier(random_state=42)
        elif name == "Naive Bayes":
            clf = GaussianNB()
        elif name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=5)

        metrics = evaluate_model(clf, name)
        results.append(metrics)

        with st.expander(f"📊 Details for {name}"):
            st.write(f"**Accuracy:** {metrics['Accuracy']}")
            st.write(f"**Precision:** {metrics['Precision']}")
            st.write(f"**Recall:** {metrics['Recall']}")
            st.write(f"**F1 Score:** {metrics['F1 Score']}")

            # Confusion Matrix
            st.write("🧩 Confusion Matrix:")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, metrics['y_pred'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            st.pyplot(fig)

            # ROC Curve
            if metrics["fpr"] is not None:
                st.write("📈 ROC Curve:")
                fig, ax = plt.subplots()
                ax.plot(metrics["fpr"], metrics["tpr"], label=f"AUC = {metrics['auc']:.2f}", color="orange")
                ax.plot([0, 1], [0, 1], 'k--', lw=1)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc='lower right')
                st.pyplot(fig)
            else:
                st.warning("This model does not support probability prediction.")

    if results:
        st.subheader("📋 Model Comparison Summary")
        summary_df = pd.DataFrame(results)[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]].set_index("Model")
        st.dataframe(summary_df)

    # ==============================================
    # 📊 Bar Chart: Comparison of Classifier Performance
    # ==============================================
    st.subheader("📈 Comparison of Classifier Performance")

    # ==============================================
    # 📊 Bar Chart: Themed Comparison of Classifier Performance
    # ==============================================
    if results:
        with st.expander("📊 Expand to View Classifier Performance Chart"):
            summary_df = pd.DataFrame(results)[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]]
            melted = summary_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

            fig, ax = plt.subplots(figsize=(10, 5))

            # Themed styling
            sns.set_style(sns_theme)
            background_color = "#0e1117" if theme_choice == "Dark" else "#ffffff"
            text_col = "#ffffff" if theme_choice == "Dark" else "#000000"

            sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="pastel", ax=ax)

            # Apply background and text theming
            fig.patch.set_facecolor(background_color)
            ax.set_facecolor(background_color)
            ax.tick_params(colors=text_col)
            ax.spines['bottom'].set_color(text_col)
            ax.spines['left'].set_color(text_col)
            ax.yaxis.label.set_color(text_col)
            ax.xaxis.label.set_color(text_col)
            ax.title.set_color(text_col)
            ax.legend().get_frame().set_facecolor(background_color)
            for text in ax.legend().get_texts():
                text.set_color(text_col)

            plt.ylim(0, 1)
            plt.title("Comparison of Classifier Performance")
            st.pyplot(fig)

            st.markdown("""
            **📊 Interpreting Classifier Comparison Chart**  
            - Compares multiple model scores on Accuracy, Precision, Recall, and F1 Score.  
            - Helps pick best model for campaign prediction.  
            - Consider trade-offs between high recall vs. precision.
            """)



# ==============================================
# 📘 Challenges Faced & Learning Reflection
# ==============================================
st.subheader("📘 Challenges Faced & Key Reflections")

# 8.1 Common Challenges
st.markdown("### 🔧 Common Challenges")

st.markdown("""
- **Dataset Cleaning:**  
  The original dataset was stored as a single tab-separated column.  
  ✔️ Resolved using `sep='\\t'` in `pd.read_csv()`.

- **Missing Values:**  
  24 missing entries were found in the `Income` column.  
  ✔️ Dropped them to ensure reliable modeling.

- **High Dimensionality (from Categorical Encoding):**  
  One-hot encoding introduced many new features.  
  ✔️ Managed through careful preprocessing and feature selection.

- **Imbalanced Target Variable (`Response`):**  
  The dataset had more non-responders (0) than responders (1).  
  ✔️ Evaluated models using precision, recall, and F1-score instead of just accuracy.

- **Choosing Number of Clusters:**  
  Initially uncertain about optimal cluster count.  
  ✔️ Used 3 clusters based on domain logic and PCA separation.

- **Association Rule Mining Issue:**  
  Raw spending values couldn't be directly used in Apriori.  
  ✔️ Converted to binary format (1 = purchased) before applying the algorithm.
""")

# 8.2 Reflections
st.markdown("### 🎓 Reflections & Learning Outcomes")

st.markdown("""
- Gained practical experience in **data preprocessing, cleaning, and feature engineering**.
- Applied and interpreted both **unsupervised (KMeans)** and **supervised (classification)** models.
- Understood how **association rules** can support product bundling strategies.
- Saw the power of **PCA, heatmaps, and 3D plots** in communicating insights visually.
- Collaborated effectively as a team by dividing roles in **data prep, modeling, and interpretation**.
""")




