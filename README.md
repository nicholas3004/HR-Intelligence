# 🧠 HR Intelligence Platform
### Predicting Employee Wellbeing, Retention & Conflict Risk Using Behavioural and HR Data

A Data-Science + Behavioral-Science Model to Support HR Decision-Making in Multicultural Scientific Organisations.

---

## 🚀 Quick Setup (5 minutes)

### Step 1 — Install Python dependencies
Open your terminal in VS Code and run:
```bash
pip install -r requirements.txt
```

### Step 2 — Generate dataset & train models
```bash
python generate_data_and_train.py
```
This will:
- Generate a realistic synthetic HR dataset (1,500 employees)
- Train 3 ML models (Retention, Conflict, Wellbeing)
- Save everything to `data/` and `models/` folders

### Step 3 — Launch the web app
```bash
streamlit run app.py
```
App opens automatically at: **http://localhost:8501**

---

## 📁 Project Structure
```
hr_project/
│
├── app.py                      ← Main Streamlit web application
├── generate_data_and_train.py  ← Dataset generator + model trainer (run once)
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
├── data/
│   └── hr_dataset.csv          ← Generated after Step 2
│
└── models/
    ├── retention_model.pkl     ← GradientBoosting attrition classifier
    ├── conflict_model.pkl      ← RandomForest conflict classifier
    ├── wellbeing_model.pkl     ← GradientBoosting wellbeing regressor
    ├── scaler.pkl              ← StandardScaler
    └── features.pkl            ← Feature column list
```

---

## 🎯 Features

| Page | Description |
|---|---|
| 📊 Executive Dashboard | KPIs, risk distributions, heatmaps, top at-risk employees |
| 👥 Employee Explorer | Drill-down per employee with radar charts & risk profiles |
| 🔮 Risk Predictor | Enter any employee profile → instant risk + wellbeing prediction |
| 📈 Model Insights | Feature importance, attrition by dept/role, model performance |
| 🌍 Multicultural Analysis | Risk patterns by gender, education, marital status, travel |

---

## 🤖 Models Used

| Model | Algorithm | Target |
|---|---|---|
| Retention Risk | Gradient Boosting Classifier | P(employee leaves) |
| Conflict Risk | Random Forest Classifier | P(interpersonal conflict) |
| Wellbeing Score | Gradient Boosting Regressor | Score 0–100 |

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **Streamlit** — Web application framework
- **Scikit-learn** — Machine learning models
- **Plotly** — Interactive visualisations
- **Pandas / NumPy** — Data processing

---

## 📌 LinkedIn Description
> Built a full-stack HR Intelligence Platform using Python, Streamlit, and Scikit-learn to predict Employee Wellbeing, Retention Risk, and Conflict Risk across multicultural scientific organisations. Features include an AI risk predictor, executive dashboard, multicultural analysis, and individual employee drill-down — powered by Gradient Boosting and Random Forest models.

**Skills:** Machine Learning · Predictive Analytics · Streamlit · Data Visualization · HR Analytics · Python
