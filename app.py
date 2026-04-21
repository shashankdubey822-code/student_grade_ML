import pickle
import numpy as np
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt
import fairlearn
from fairlearn.reductions import ExponentiatedGradient

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_bundle():
    import sys
    try:
        import fairlearn.reductions
        sys.modules['fairlearn.reductions'] = fairlearn.reductions
    except:
        pass

    try:
        with open("student_performance_production_bundle.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Bundle file not found. Run the notebook first to generate it.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading bundle: {type(e).__name__}: {str(e)[:200]}")
        st.info("Attempting to load RF model only...")
        try:
            import joblib
            rf_model = joblib.load("student_performance_production_bundle.pkl")
            st.success("Loaded as joblib file")
            return rf_model
        except:
            st.stop()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Student Feature Inputs")

try:
    bundle               = load_bundle()
    rf_model             = bundle["rf_model"]
    fair_model           = bundle["fair_model"]
    feature_names        = bundle["feature_names"]
    categorical_mappings = bundle["categorical_mappings"]
except:
    st.warning("⏳ App initializing... Model file will be available once you upload the .pkl file")
    st.stop()

user_input = {}
for feat in feature_names:
    if feat in categorical_mappings:
        options  = categorical_mappings[feat]
        selected = st.sidebar.selectbox(feat, options)
        user_input[feat] = options.index(selected)
    else:
        RANGES = {
            "age": (15, 22, 17), "Medu": (0, 4, 2), "Fedu": (0, 4, 2),
            "traveltime": (1, 4, 1), "studytime": (1, 4, 2),
            "failures": (0, 3, 0), "famrel": (1, 5, 4),
            "freetime": (1, 5, 3), "goout": (1, 5, 3),
            "Dalc": (1, 5, 1), "Walc": (1, 5, 2),
            "health": (1, 5, 3), "absences": (0, 75, 4),
            "G1": (0, 20, 12), "G2": (0, 20, 13),
        }
        lo, hi, default = RANGES.get(feat, (0, 10, 5))
        user_input[feat] = st.sidebar.slider(feat, lo, hi, default)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🎓 Student Performance Prediction")
st.markdown("Predict a student's final grade (G3) using academic and social features.")

model_choice = st.radio(
    "Select Model",
    ["Baseline Random Forest", "Fairness-Aware XGBoost"],
    horizontal=True
)

if st.button("Predict Final Grade", type="primary"):
    input_df       = np.array([user_input])
    model          = rf_model if "Random Forest" in model_choice else fair_model
    predicted_grade = float(model.predict(input_df)[0])

    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Final Grade (G3)", f"{predicted_grade:.2f} / 20")
    col2.progress(int(predicted_grade * 5), text="Grade Progress (0-20 scale)")

    if "Random Forest" in model_choice:
        st.subheader("Why this prediction? (SHAP Force Plot)")
        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(input_df)
        plt.figure(figsize=(14, 4))
        shap.force_plot(
            explainer.expected_value, shap_vals[0],
            input_df.iloc[0], matplotlib=True, show=False
        )
        st.pyplot(plt.gcf(), use_container_width=True)
        st.caption("Red features push the prediction higher; blue features push it lower.")