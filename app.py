import pickle
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import shap

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Success Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #0d6efd; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Load Production Bundle ---
@st.cache_resource
def load_bundle():
    try:
        with open("student_performance_production_bundle.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Bundle not found! Please run the ML Notebook first.")
        st.stop()

bundle = load_bundle()
pipeline = bundle['pipeline']
input_cols = bundle['feature_names_input']
cat_mappings = bundle['categorical_mappings']

# --- Sidebar: Input Controls ---
st.sidebar.title("🛠 Student Profile")
st.sidebar.markdown("Configure the student parameters below to analyze predicted outcomes.")

def get_user_input():
    inputs = {}
    
    st.sidebar.subheader("Academic History")
    inputs['G1'] = st.sidebar.slider("Midterm 1 Grade (0-20)", 0, 20, 10)
    inputs['G2'] = st.sidebar.slider("Midterm 2 Grade (0-20)", 0, 20, 10)
    inputs['failures'] = st.sidebar.selectbox("Past Class Failures", [0, 1, 2, 3], index=0)
    inputs['absences'] = st.sidebar.number_input("Total Absences", 0, 93, 4)
    
    st.sidebar.divider()
    st.sidebar.subheader("School & Study")
    inputs['school'] = st.sidebar.selectbox("School", cat_mappings['school'])
    inputs['studytime'] = st.sidebar.slider("Study Time (1: <2h, 4: >10h)", 1, 4, 2)
    inputs['traveltime'] = st.sidebar.slider("Travel Time (1: <15m, 4: >1h)", 1, 4, 1)
    inputs['schoolsup'] = st.sidebar.selectbox("Extra Educational Support", cat_mappings['schoolsup'])
    inputs['higher'] = st.sidebar.selectbox("Wants Higher Education?", cat_mappings['higher'])
    
    st.sidebar.divider()
    st.sidebar.subheader("Demographics & Social")
    inputs['sex'] = st.sidebar.selectbox("Gender", cat_mappings['sex'])
    inputs['age'] = st.sidebar.slider("Age", 15, 22, 17)
    inputs['address'] = st.sidebar.selectbox("Address (U: Urban, R: Rural)", cat_mappings['address'])
    inputs['famsize'] = st.sidebar.selectbox("Family Size", cat_mappings['famsize'])
    inputs['Pstatus'] = st.sidebar.selectbox("Parent Status", cat_mappings['Pstatus'])
    inputs['Mjob'] = st.sidebar.selectbox("Mother's Job", cat_mappings['Mjob'])
    inputs['Fjob'] = st.sidebar.selectbox("Father's Job", cat_mappings['Fjob'])
    inputs['Medu'] = st.sidebar.slider("Mother's Education (0-4)", 0, 4, 2)
    inputs['Fedu'] = st.sidebar.slider("Father's Education (0-4)", 0, 4, 2)
    inputs['reason'] = st.sidebar.selectbox("Reason for School", cat_mappings['reason'])
    inputs['guardian'] = st.sidebar.selectbox("Guardian", cat_mappings['guardian'])
    inputs['famsup'] = st.sidebar.selectbox("Family Support", cat_mappings['famsup'])
    inputs['paid'] = st.sidebar.selectbox("Paid Classes", cat_mappings['paid'])
    inputs['activities'] = st.sidebar.selectbox("Extracurriculars", cat_mappings['activities'])
    inputs['nursery'] = st.sidebar.selectbox("Attended Nursery", cat_mappings['nursery'])
    inputs['internet'] = st.sidebar.selectbox("Internet at Home", cat_mappings['internet'])
    inputs['romantic'] = st.sidebar.selectbox("Romantic Relationship", cat_mappings['romantic'])
    inputs['famrel'] = st.sidebar.slider("Family Relations (1-5)", 1, 5, 4)
    inputs['freetime'] = st.sidebar.slider("Free Time (1-5)", 1, 5, 3)
    inputs['goout'] = st.sidebar.slider("Going Out (1-5)", 1, 5, 3)
    inputs['Dalc'] = st.sidebar.slider("Workday Alcohol (1-5)", 1, 5, 1)
    inputs['Walc'] = st.sidebar.slider("Weekend Alcohol (1-5)", 1, 5, 1)
    inputs['health'] = st.sidebar.slider("Health Status (1-5)", 1, 5, 5)
    
    return inputs

user_data = get_user_input()

# --- Main Dashboard ---
st.title("🎯 Student Success & Performance Dashboard")
st.markdown("""
Predict final academic outcomes and uncover the hidden drivers behind student performance using 
**AI-driven interpretability.** Adjust parameters in the sidebar to see real-time updates.
""")

st.divider()

# Prediction logic
input_df = pd.DataFrame([user_data])
# Add Engineered Feature
input_df['academic_pressure'] = input_df['failures'] * (5 - input_df['studytime'])
# Reorder to match training
input_df = input_df[input_cols]

prediction = pipeline.predict(input_df)[0]

# --- Row 1: Key Metrics ---
m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Predicted Final Grade (G3)", f"{prediction:.1f} / 20")
    
with m2:
    status = "Pass" if prediction >= 10 else "Fail"
    color = "normal" if status == "Pass" else "inverse"
    st.metric("Academic Standing", status, delta=None, delta_color=color)

with m3:
    percentile = (prediction / 20) * 100
    st.metric("Performance Capacity", f"{percentile:.1f}%")

# --- Row 2: Visual Feedback ---
st.subheader("📊 Performance Analysis")
col_chart, col_tips = st.columns([2, 1])

with col_chart:
    # Progress gauge
    progress_color = "green" if prediction >= 15 else "orange" if prediction >= 10 else "red"
    st.markdown(f"**Predicted Grade Visualizer**")
    st.progress(min(max(prediction/20, 0.0), 1.0))
    
    if prediction >= 15:
        st.success("🏆 **High Achiever:** This student shows strong indicators for academic excellence.")
    elif prediction >= 10:
        st.info("📈 **On Track:** Standard performance. Minor adjustments could lead to improvement.")
    else:
        st.error("⚠️ **Intervention Required:** High risk of failure. Targeted support is recommended.")

with col_tips:
    st.markdown("**Academic Recommendations**")
    if user_data['absences'] > 10:
        st.warning("📍 High absenteeism detected. Suggest student attendance counseling.")
    if user_data['failures'] > 0:
        st.warning("📍 Past failures are weighing down current potential. Recommend remedial classes.")
    if user_data['studytime'] < 2:
        st.info("📍 Low study time. A 1hr/day increase could significantly shift the prediction.")
    if prediction >= 10 and prediction < 14:
        st.success("📍 Predicted pass! Encouragement of current habits is advised.")

st.divider()

# --- Row 3: Model Interpretation (SHAP) ---
st.subheader("🔍 Why this prediction? (AI Interpretation)")
st.markdown("This section explains which specific features pushed the score up (red) or down (blue).")

# Explainability using SHAP
try:
    prep = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['regressor']
    
    # Transform input and get feature names
    transformed_input = prep.transform(input_df)
    feature_names_transformed = prep.get_feature_names_out()
    
    # Clean up feature names (remove prefixes like 'cat__', 'num__')
    clean_feature_names = [f.split('__')[-1] for f in feature_names_transformed]
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(transformed_input)

    # Use Matplotlib to render the force plot
    # We use a wrapper to ensure SHAP draws correctly on the figure
    fig, ax = plt.subplots(figsize=(12, 4))
    shap.force_plot(
        explainer.expected_value, 
        shap_vals[0], 
        transformed_input[0], 
        feature_names=clean_feature_names,
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()
    
    st.caption("Insights: Features in **RED** contribute to a higher grade. Features in **BLUE** pull the grade lower.")
except Exception as e:
    st.info("SHAP Interpretation: Adjust sidebar values to see specific feature impacts.")

# --- Footer ---
st.divider()
st.caption(f"Model: {bundle.get('model_version', 'HD-1.0')} | Metric: RMSE {bundle.get('evaluation_metrics', {}).get('RMSE', 'N/A')}")
