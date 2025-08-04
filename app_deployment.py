import numpy as np
import pandas as pd
pd.set_option('display.width',None)

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, classification_report, mean_absolute_error, mean_squared_error, roc_auc_score, silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import streamlit as st
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
def about_model():
    st.header("🧠 Cracking the Code: What the OSMI Survey Tells Us About Mental Health in Tech")
    st.subheader("Capstone Project 2025 | OpenLearn Cohort 1.0")
    st.markdown("👨‍💻*This project has been developed under the mentorship and guidance of the [OpenLearn Community](https://www.openlearn.org.in/). Proudly part of the OpenLearn Cohort 1.0.*")
    st.markdown("---")
    st.subheader("🎯Objective Of The Model")
    objective={
       'Task':["1.Classification","2.Regression","3.Unsupervised Learning"],
       'Input Features':["Demographics, workplace metrics, self‑reported stress",
                         "Behaviorial patterns, job role, wellness indicators",
                         "Multiple wellness metrics and survey responses"],
        'Goal':["Probability of seeking treatment","Predict chronological age","Wellness-based employee clusters"],
        'Potential Impact':["Prioritize outreach for high‑risk employees","Customize wellness content and communication by age",
                            "Design tailored HR interventions and resource allocation"]
    }
    st.dataframe(objective)
    
    st.divider()
    st.subheader("🧾 Dataset Overview")
    st.markdown("- **Source:** Mental Health in Tech Survey")
    st.markdown("- **Collected by:** [OSMI (Open Sourcing Mental Illness)](https://osmihelp.org/)")
    st.markdown("""**Features include:**
  - Demographic attributes (age, gender, country)
  - Workplace environment (mental health benefits, leave policies)
  - Personal mental health history (self and family)
  - Attitudes and perceptions around mental health in the workplace""")
    
    st.divider()
    st.subheader("🧪 Project Case Study")
    st.write(""" While using this model, you'll consider yourself as a **Machine Learning Engineer** at NeuronInsights Analytics and 
    you've been contracted by a coalition of leading tech companies including **CodeLab**,**QuantumEdge**, and **SynapseWorks.**
             
    **Roles:**
    - Analyzing survey data from over **1,500 tech professionals.**
    - **Identifying risk patterns** based on various features.
    - Predicting whether a person **will seek treatment or no.**
    - **Predicting Age** of a person.""")

    st.divider()
    st.subheader("🔍 Key Questions Explored")
    st.subheader("Q1.Who is most likely to suffer silently and avoid seeking treatment?")
    st.write("""*Answer:*  Employees who are most likely to suffer silently and avoid seeking treatment are:""")
    st.write("**1. Young Employees(Age 18-24):**  According to the survey, this group reports high stress level and more likely to take " \
    "leave due to mental health issues as compared to older employees.")
    st.write("**2. Employees Without Mental Health Benefits:**  Employees had lack of access to mental health benefits which resulted into " \
    "untreated conditions. Also, this employees were hesitant to seek help.")
    st.write("**3. Workers with Limited Managerial Support:**  Employees didn't have a good access to guidance on mental health condition " \
    "due to which they didn't have any courage to seek treatment.")
    st.write("**4. Remote Workers:**  Yes, remote working offers great help nowadays but it leads to the feeling of isolation and loneliness" \
    " of the employee if not managed properly.")

    st.subheader("Q2.How Do Factors Like Remote Work, Mental Health Benefits, and Managerial Support Influence Mental Well Being?")
    st.write("*Answer:*")
    st.write("**1. Remote Work:**  Remote Work can surely reduce stress and work load but it may increase isolation and burnout on others." \
    "Remote workers can experience increased loneliness and stress if they lack support and clear boundaries between work and personal life.")
    st.write("**2. Mental Health Benefits:**  It is necessary to get mental health benefits in order to seek help and manage stress effectively.")
    st.write("**3. Managerial Support:**  Mangers can play an important role here as if they proviide a clear commumnication,recognition and" \
    " flexibility can significantly reduce stress and promote a positive environment.")

    st.subheader("Q3.Can employee profiles be segmented to enable targeted outreach and HR intervention?")
    st.write("*Answer:* Yes,employee profiles can be segmented to tailor mental health initiatives effectively:")
    st.write("**1. Demographic Segmentation:** Age, gender, and role can influence stress levels and treatment-seeking behaviour.")
    st.write("**2. Work Arrangement:** Remote workers may require more social interaction and support " \
    "while in office employees may benefit from stress management workshops.")
    st.write("**3. Health Benefits Access:** It is necessary to get mental health benefits in order to seek help and manage stress effectively.")
    st.write("**4. Managerial Support Levels:** Managerial Support can lead to great importance, as they can help employees " \
    "to seek treatment.")
    st.markdown("**SOLUTION:** Implementing a data driven approach using employee surveys, feedback and analytics can help department " \
    "to find risk employees and design targeted interventions such as vaious wellness programs and workshops.")

df=pd.read_csv("clean_data.csv")
feature_cols = [c for c in df.columns if c not in [ "Age",'Gender','treatment','Country']]
X = df[feature_cols]

def data_visualisation():
    st.header("Mental Health In Tech Survey Data Visualisation")
    st.code(X.head())
    st.subheader("Dataset Shape")
    st.write(f"Features shape: {df[feature_cols].shape}")
    st.write(f"Target shape: {df['treatment'].shape}")

    st.subheader("🔍📅Dataset NULL counts and dtypes")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Null Values Count")
        st.write(df.isna().sum())

    with col2:
        st.subheader("Data Types")
        st.write(df.dtypes)

    st.divider()
    st.title("Visualisation  of the Dataset")

    sns.set_style("whitegrid")
    col1, col2 = st.columns(2)
    with col1:
      fig, ax = plt.subplots(figsize=(10, 6))
      sns.countplot(data=df, x='treatment', hue='work_interfere', palette="deep", ax=ax)
      ax.set_title("Treatment by Workplace Interference")
      st.pyplot(fig)

      if st.button("Show Insights for Treatment by Workplace Interference"):
        st.write("""
        - **Treatment Seeking Behavior**: The countplot illustrates how individual's 
                 decisions to seek treatment vary across different levels of workplace interference.
        - **Influence of Workplace Environment**: By examining the hue (workplace interference),
                  we can assess whether higher levels of workplace interference correlate with a greater likelihood of seeking treatment.
        - **Policy Implications**: If a significant portion of individuals experiencing high workplace interference seek treatment, 
                 organizations might consider implementing supportive policies to address mental health concerns.
        """)

    with col2:
      fig, ax = plt.subplots(figsize=(10, 6))
      sns.countplot(data=df, x='Gender', hue='treatment', palette="deep", ax=ax)
      ax.set_title("Gender Distribution Based On Treatment")
      st.pyplot(fig)

      if st.button("Show Insights for Gender Distribution Based On Treatment"):
        st.write("""
        - **Gender and Treatment Seeking**: This plot reveals the relationship between gender and the propensity to seek treatment.
        - **Identifying Disparities**: By analyzing the counts, we can identify if there's a gender disparity in seeking treatment.
                  For instance, if one gender has a significantly higher count in the 'treatment' category, 
                 it might indicate a trend or bias.
        - **Targeted Interventions**: Recognizing gender-based differences in treatment-
                 seeking behavior can inform targeted mental health initiatives tailored to specific gender groups.
        """)

    col1, col2 = st.columns(2)
    with col1:
       fig, ax = plt.subplots(figsize=(10, 6))
       sns.countplot(data=df, x='mental_health_consequence', hue='treatment', palette="viridis", ax=ax)
       ax.set_title("Mental Health Consequences vs Treatment")
       st.pyplot(fig)

       if st.button("Show Insights for Mental Health Consequences vs Treatment"):
          st.write("""
        - **Perception of Mental Health Consequences**: The 'mental_health_consequence' variable likely categorizes 
                   individuals based on their perception of mental health consequences.
        - **Treatment Correlation**: By examining the hue, we can determine if individuals 
                   who perceive mental health consequences are more or less likely to seek treatment.
        - **Awareness Campaigns**: If a significant number of individuals who perceive 
                   mental health consequences do not seek treatment, it may highlight the need for awareness
                    campaigns to encourage treatment-seeking behavior.
        """)

    with col2:
       fig, ax = plt.subplots(figsize=(10, 6))
       sns.histplot(df, x='Age', kde=True, ax=ax)
       ax.set_title("Age Distribution")
       st.pyplot(fig)

       if st.button("Show Insights for Age Distribution"):
         st.write("""
        - **Age Range**: The histogram provides a visual representation of the age distribution within the dataset.
        - **KDE Curve**: The Kernel Density Estimate (KDE) curve overlays the histogram to show the
                   probability density function of the age variable.
        - **Age Trends**: By analyzing the histogram and KDE, we can identify the most common 
                  age ranges and any skewness in the data. For instance, a peak in the 25-35 age range 
                  might suggest that this group is more prevalent in the dataset.
        """)

def predict_treatment():
    st.header("🔍Treatment Prediction")
    st.subheader("📊Random Forest Classifier Results")
    st.markdown("**Accuracy:** 0.8026")
    st.markdown("**ROC AUC:** 0.8890")
    st.markdown("**F1 Score:** 0.8026")

    st.subheader("📊Logistic Regression Results")
    st.markdown("**Accuracy:** 0.8293")
    st.markdown("**ROC AUC:** 0.9015")
    st.markdown("**F1 Score:** 0.8289")

    st.subheader("📊K neighbors Classifier Results")
    st.markdown("**Accuracy:** 0.7386")
    st.markdown("**ROC AUC:** 0.8032")
    st.markdown("**F1 Score:** 0.7389")
    
    feature_cols = ['family_history','work_interfere','mental_vs_physical','obs_consequence','benefits','care_options',
                    'wellness_program','seek_help','leave','coworkers','supervisor','mental_health_consequence',
                    'phys_health_consequence','no_employees','remote_work','tech_company']
    st.header("🤖📈It's Time To Predict")
    with st.form("input_form"):
       st.subheader("Enter details of the Employee:")
       col1, col2 = st.columns(2)
       inputs = {}
       mid = len(feature_cols) // 2
       left = feature_cols[:mid]
       right = feature_cols[mid:]

       with col1:
           for col in left:
                options = sorted(df[col].dropna().unique())
                inputs[col] = st.selectbox(col,options)

       with col2:
           for col in right:
            options = sorted(df[col].dropna().unique())
            inputs[col] = st.selectbox(col,options)

       submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        new_df = pd.DataFrame([inputs])

        full = pd.concat([df.drop(columns=['treatment']), new_df], ignore_index=True)

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        preprocessor = ColumnTransformer([('category', ohe, feature_cols)])

        X_encoded = preprocessor.fit_transform(full)

        fitted_ohe = preprocessor.named_transformers_['category']
        feature_names = fitted_ohe.get_feature_names_out(feature_cols)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        X_train = X_scaled[:-1]
        X_new = X_scaled[-1].reshape(1, -1)

        y_treatment = df['treatment']

        X_train_part, X_test, y_train_part, y_test = train_test_split(X_train, y_treatment, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        clf.fit(X_train_part, y_train_part)

        importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)

        st.subheader("Top 10 Important Features:")
        st.write(importances.head(10))

        pred= clf.predict(X_new)[0]
        proba = clf.predict_proba(X_new)[0]

        classes = list(clf.classes_)
        st.subheader("Prediction Result:")
        if pred == 1 or pred == "Yes":
            st.write("✅ Will seek treatment")
        else:
            st.write("❌ Will not seek treatment")
        st.subheader("Prediction Result:")

        pred_index = classes.index(pred) if pred in classes else int(pred)

        st.write(f"Confidence: {proba[pred_index]*100:.1f}%")

        st.subheader("Prediction Probabilities:")
        proba_df = pd.DataFrame(proba.reshape(1, -1), columns=classes)
        st.write(proba_df.T)

        y_pred_test = clf.predict(X_test)
        st.subheader("Model Performance on Test Set")
        st.text(classification_report(y_test, y_pred_test))

        fig_roc = plt.figure(figsize=(8,6))
        RocCurveDisplay.from_estimator(clf, X_test, y_test)
        st.pyplot(fig_roc)

        fig_cm = plt.figure(figsize=(8,6))
        ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
        st.pyplot(fig_cm)

def predict_age():
    st.header("🔍Age Prediction")
    st.subheader("📊Linear Regression Results")
    st.markdown("**RMSE:** 7.152")
    st.markdown("**MAE:** 5.484")
    st.markdown("**R²:** 0.022")

    st.subheader("📊Random Forest Regressor Results")
    st.markdown("**RMSE:** 7.214")
    st.markdown("**MAE:** 5.579")
    st.markdown("**R²:** 0.005")

    st.markdown("*R² Score is not giving good score since features given in the dataset are not liable for predicting age.*")

    feature_cols = ['Gender','remote_work','tech_company','wellness_program','treatment']
    X_original = df[feature_cols]
    y_original = df['Age']

    pipeline = Pipeline([
        ("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)])),("scaler", StandardScaler()),
        ("reg", RandomForestRegressor(n_estimators=100, random_state=42))])

    st.subheader("🤖📈It's Time To Predict")
    with st.form("input_form"):
        st.subheader("Enter Employee Details:")
        inputs = {}
        for col in feature_cols:
            options = sorted(df[col].dropna().unique())
            inputs[col] = st.selectbox(col, options)
        submitted = st.form_submit_button("🔍 Predict Age")

    if not submitted:
        st.info("Fill the form and press **Predict Age**")
        return

    new_df = pd.DataFrame([inputs])

    X_train, X_test, y_train, y_test = train_test_split(X_original, y_original, test_size=0.3, random_state=42)

    pipeline.fit(X_train, y_train)
    predicted_age = pipeline.predict(new_df)[0]
    st.success(f"Predicted Age: **{predicted_age:.1f} years**")

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"- MAE: {mae:.2f}")
    st.write(f"- RMSE: {rmse:.2f}")
    st.write(f"- R²: {r2:.3f}")

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax1, hue=y_test, palette="viridis", alpha=0.7)
    ax1.set_xlabel("Actual Age")
    ax1.set_ylabel("Predicted Age")
    ax1.set_title("Actual vs Predicted")

    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.7, edgecolors='w', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel("Predicted Age")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residuals Plot")

    plt.tight_layout()
    st.pyplot(fig)

def clustering():
   st.header("🤖📊Unsupervised Learning")
   st.subheader("K-Means Clustering")
   st.markdown("K-Means Clustering is a popular unsupervised machine learning algorithm used to group unlabeled " \
   "data points into a pre-defined number of clusters. The “K” in K-Means refers to the number of clusters you want to create," \
   " which dictate know much information you can extract that is actually useful from the data. It’s an iterative, " \
   "centroid-based algorithm, meaning it aims to find groups based on the distance of data points to central points (centroids)." \
   "The goal is similar to Supervised learning: Minimize the distance from a point, " \
   "here the (K) centroids instead of the actual target labels used for training.")
   
   st.divider()
   st.header("🧮Clustering: K Means, Agglomerative & DBSCAN")
   st.write("""Here, I have used **3 methods**:
   - **K Means** clustering
   - **Agglomerative** clustering
   - **DBSCAN**

   Out of these, **K Means** gives the best results with **Silhouette Score=0.307** at **K=3**.""")
   st.subheader("**📋Features used**")
   st.markdown("**1. Perceived mental health support 📋✔️** contains features like **benefits, care_options, wellness_program, seek_help, anonymity.**")
   st.markdown("**2. Openness to discussing mental health 📝➡️✅** contains features like **coworkers, supervisor, mental_health_interview.**")
   st.markdown("**3. Prior experience with mental health issues 📑🔍** contains features like **family_history, work_interfere, mental_health_consequence, obs_consequences.**")
   
   st.divider()
   st.subheader("CLustering Plot")
   image=Image.open("clustering.png")
   st.image(image,use_container_width=True)

   st.header("🧩Interpretation of Clusters")
   st.markdown("""#### **1. Cluster 0**
 **1. Support:** Mean=4.02 [relatively high]
               
 **2. Openness:** Mean=1.29 [very low]
               
 **3. Experience:** Mean=1.62 [very low]

 **This shows that Cluster 0 shows : High mental health support with low to medium openness and experience with mental health issues.**""")
   st.markdown("""#### **2. Cluster 1**
 **1. Support:** Mean=1.48 [very low]
               
 **2. Openness:** Mean=1.51 [modestly low]
               
 **3. Experience:** Mean=0.69 [quite low]

 **This shows that Cluster 1 shows : High openess to discuss there mental health issues but has low mental health support and experience.**""")
   st.markdown("""#### **3. Cluster 2**
 **1. Support:** Mean=1.55 [low]
               
 **2. Openness:** Mean=0.45 [extremely low]
               
 **3. Experience:** Mean=1.95 [relatively high]

 **This shows that Cluster 2 shows : Very low openness and experience indicating that they are unaware but they are provided with some facities for mental health.**""")

pg = st.navigation([
  st.Page(about_model, title="About The Model"),
  st.Page(data_visualisation, title="Dataset Visualization"),
  st.Page(predict_treatment, title="Treatment Prediction"),
  st.Page(predict_age, title="Age Prediction"),
  st.Page(clustering, title="Unsupervised Learning")
])
pg.run()