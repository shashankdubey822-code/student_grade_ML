---
title: Student Performance Prediction
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.40.2"
app_file: app.py
pinned: false
license: mit
---

# 🎓 Student Performance Prediction - Machine Learning Project

A comprehensive machine learning system that predicts student final grades using academic and social features. This project includes two models: a **Baseline Random Forest** and a **Fairness-Aware XGBoost** model with SHAP explainability.

## 📋 Project Overview

This project aims to help educators understand what factors influence student performance and make fair predictions. It uses multiple machine learning algorithms and provides interpretable predictions through SHAP force plots.

### Key Features:
- ✅ **Two Prediction Models**: Baseline Random Forest and Fairness-Aware XGBoost
- ✅ **SHAP Explainability**: Understand why the model makes specific predictions
- ✅ **Fairness-Aware Learning**: Reduces bias in predictions across different student groups
- ✅ **Interactive Web Interface**: Built with Streamlit for easy use
- ✅ **Real-time Predictions**: Get instant grade predictions based on student features

---

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/machine-learning-project.git
   cd machine-learning-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the model bundle exists:**
   - Make sure `student_performance_production_bundle.pkl` is in the project directory
   - This file contains the trained models and feature mappings

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:**
   - Navigate to `http://localhost:8501`

---

## 📊 Frontend Parameters Explained

The application has a **sidebar** where users input student information. Here's what each parameter means:

### **Input Parameters:**

#### 📚 **Academic Features:**

| Parameter | Range | What It Means | Example |
|-----------|-------|---------------|---------|
| **age** | 15-22 years | Student's current age | 17 (typical student) |
| **Medu** | 0-4 | Mother's education level | 2 (secondary education) |
| **Fedu** | 0-4 | Father's education level | 2 (secondary education) |
| **G1** | 0-20 | First period grade | 12 (good performance) |
| **G2** | 0-20 | Second period grade | 13 (improving performance) |

**Education Levels Mapping (Medu, Fedu):**
- 0 = None
- 1 = Primary education (4 years)
- 2 = Secondary education (6-9 years)
- 3 = Higher education (9-12 years)
- 4 = University degree

#### ⏱️ **Time & Study Habits:**

| Parameter | Range | What It Means | Example |
|-----------|-------|---------------|---------|
| **traveltime** | 1-4 | Travel time to school | 1 (very short, <15 min) |
| **studytime** | 1-4 | Weekly study hours | 2 (5-10 hours) |
| **failures** | 0-3 | Past class failures | 0 (no failures) |
| **absences** | 0-75 | Number of school absences | 4 (low absenteeism) |

**Travel Time Scale:**
- 1 = <15 minutes
- 2 = 15-30 minutes
- 3 = 30 minutes to 1 hour
- 4 = >1 hour

**Study Time Scale:**
- 1 = <2 hours/week
- 2 = 2-5 hours/week
- 3 = 5-10 hours/week
- 4 = >10 hours/week

#### 👨‍👩‍👧‍👦 **Social & Health Features:**

| Parameter | Range | What It Means | Example |
|-----------|-------|---------------|---------|
| **famrel** | 1-5 | Family relationship quality | 4 (very good) |
| **freetime** | 1-5 | Leisure time after school | 3 (moderate) |
| **goout** | 1-5 | How often they go out | 3 (moderate) |
| **health** | 1-5 | Health status | 3 (good) |
| **Dalc** | 1-5 | Workday alcohol consumption | 1 (very low) |
| **Walc** | 1-5 | Weekend alcohol consumption | 2 (low) |

**Quality/Consumption Scale (1-5):**
- 1 = Very Low
- 2 = Low
- 3 = Medium
- 4 = High
- 5 = Very High

---

## 🤖 Model Selection

The app provides two prediction models:

### **1. Baseline Random Forest** 
- Fast and accurate traditional machine learning model
- Provides SHAP explanations
- May have some bias across different student groups
- Best for: Understanding feature importance

### **2. Fairness-Aware XGBoost**
- Designed to reduce prediction bias
- Ensures fair treatment across different student demographics
- May have slightly different predictions than baseline
- Best for: Ethical, unbiased predictions

**How to choose:**
- Use **Baseline RF** for understanding which features matter most
- Use **Fairness-Aware XGBoost** for making fair decisions about student support

---

## 📈 Prediction Output

After clicking **"Predict Final Grade"**, you'll see:

### **1. Prediction Result**
- **Predicted Final Grade (G3)**: The model's estimated final grade (0-20 scale)
- **Grade Progress**: Visual progress bar showing performance level

### **2. SHAP Force Plot** (Random Forest only)
A visualization showing:
- **Red arrows**: Features pushing prediction UP (positive impact)
- **Blue arrows**: Features pushing prediction DOWN (negative impact)
- **Base value**: Average prediction for all students
- **Output value**: Final prediction for this student

**Example interpretation:**
```
If G1=12 is RED, it means high first period grade increases final grade
If failures=0 is BLUE pointing down, it means no failures decrease... 
(wait, that's backwards - it actually REDUCES negative impact)
```

---

## 📁 Project Structure

```
machine-learning-project/
├── app.py                                    # Main Streamlit application
├── student_performance_production_bundle.pkl # Trained models & mappings
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
└── .streamlit/                               # Streamlit configuration
    └── config.toml                           # App settings
```

---

## 🔧 Dependencies

All required packages are in `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| **streamlit** | 1.40.2 | Web app framework |
| **pandas** | 2.2.0 | Data manipulation |
| **numpy** | 1.26.4 | Numerical computing |
| **scikit-learn** | 1.4.2 | Random Forest model |
| **xgboost** | 2.0.3 | XGBoost model |
| **fairlearn** | 0.10.0 | Fairness algorithms |
| **shap** | 0.45.1 | Model explainability |
| **matplotlib** | 3.8.3 | Plotting visualizations |
| **joblib** | 1.3.2 | Model serialization |

---

## 🌐 Deployment to Hugging Face Spaces

### Option 1: Deploy to HF Spaces

```bash
# Install git-lfs if not already installed
git lfs install

# Clone your HF Space
git clone https://huggingface.co/spaces/YOUR-USERNAME/student-performance-ML_project
cd student-performance-ML_project

# Copy project files
cp app.py .
cp requirements.txt .
cp student_performance_production_bundle.pkl .

# Push to HF
git add .
git commit -m "Deploy student performance ML app"
git push
```

---

## 💡 Understanding Model Predictions

### Why predictions matter:
- **Early intervention**: Identify struggling students early
- **Resource allocation**: Focus support where it's needed
- **Fair assessment**: Understand what truly affects performance

### Important Notes:
- The model predicts based on current features - actual outcomes may vary
- Past grades (G1, G2) are strong predictors - this is expected
- Social factors matter - well-being affects academic performance
- Fairness model ensures equitable predictions across demographics

---

## 📊 Data Features Summary

**Total Input Features**: 16

- **Academic**: 5 features (age, parents' education, past grades)
- **Time/Study**: 4 features (travel time, study hours, failures, absences)
- **Social**: 7 features (family relations, free time, going out, health, alcohol use)

**Target Variable**: 
- **G3**: Final grade (0-20 scale)

---

## 🔍 How SHAP Explains Predictions

SHAP (SHapley Additive exPlanations) tells you:
1. **Base prediction**: What the model predicts on average
2. **Feature contributions**: How each student's features change the prediction
3. **Direction**: Whether each feature increases or decreases the prediction
4. **Magnitude**: How much each feature matters

**Real Example:**
```
Base prediction: 12.5
+ G2=13 (RED): +1.2 → "Good second period helps"
+ age=17 (BLUE): -0.3 → "Average age slightly hurts"
= Final prediction: 13.4
```

---

## 🛠️ Troubleshooting

### Error: "Bundle file not found"
- Ensure `student_performance_production_bundle.pkl` is in the project root
- Run the training notebook to generate it

### Error: Module not found
- Run: `pip install -r requirements.txt`
- Make sure you're using the correct Python environment

### Streamlit not starting
- Check port 8501 is not in use
- Try: `streamlit run app.py --server.port 8502`

---

## 📚 Dataset Source

This project uses student performance data with features including academic, social, and demographic information. The dataset is commonly used in machine learning education.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Improve documentation
- Add new features

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Author

Created as a machine learning project demonstrating:
- ML model development
- Model explainability (SHAP)
- Fairness in AI
- Interactive web applications

---

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Open an issue on GitHub
3. Review the code comments

---

**Happy predicting! 🎓📊**
