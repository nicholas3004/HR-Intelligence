"""
generate_data_and_train.py
Run this ONCE to create the dataset and train all models.
Usage: python generate_data_and_train.py
"""

import os, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
warnings.filterwarnings("ignore")

np.random.seed(42)
N = 1500

print("=" * 55)
print("  HR Intelligence — Data Generator & Model Trainer")
print("=" * 55)

# ── 1. SYNTHETIC DATASET ────────────────────────────────────────────────────
print("\n[1/4] Generating synthetic HR dataset …")

departments  = ["Research & Development","Sales","Human Resources"]
job_roles    = ["Research Scientist","Sales Executive","Manager","Laboratory Technician",
                "Manufacturing Director","Healthcare Representative","Human Resources",
                "Sales Representative","Research Director"]
edu_fields   = ["Life Sciences","Medical","Marketing","Technical Degree","Human Resources","Other"]
genders      = ["Male","Female"]
marital      = ["Single","Married","Divorced"]
travel_cats  = ["Non-Travel","Travel_Rarely","Travel_Frequently"]

dept_arr     = np.random.choice(departments, N, p=[0.65, 0.25, 0.10])
role_arr     = np.random.choice(job_roles,   N)
gender_arr   = np.random.choice(genders,     N, p=[0.60, 0.40])
marital_arr  = np.random.choice(marital,     N, p=[0.32, 0.46, 0.22])
travel_arr   = np.random.choice(travel_cats, N, p=[0.19, 0.71, 0.10])
edu_field_arr= np.random.choice(edu_fields,  N)

age              = np.random.randint(18, 61, N)
education        = np.random.randint(1, 6, N)
distance         = np.random.randint(1, 30, N)
num_companies    = np.random.randint(0, 10, N)
years_at_co      = np.clip(np.random.exponential(6, N).astype(int), 0, 40)
yrs_promotion    = np.clip(np.random.exponential(2.5, N).astype(int), 0, 15)
yrs_manager      = np.clip(np.random.exponential(4, N).astype(int), 0, 17)
training_times   = np.random.randint(0, 7, N)
stock_option     = np.random.choice([0,1,2,3], N, p=[0.4,0.35,0.15,0.10])
total_working_yrs= np.clip(age - 18 + np.random.randint(-3, 4, N), 0, 43)
perf_rating      = np.random.choice([3,4], N, p=[0.85, 0.15])

job_sat          = np.random.randint(1, 5, N)
env_sat          = np.random.randint(1, 5, N)
rel_sat          = np.random.randint(1, 5, N)
wlb              = np.random.randint(1, 5, N)

# Income with role/edu gradient
base_income = (education * 5000 + np.random.randint(20000, 60000, N))
monthly_income = np.clip(base_income, 10000, 200000)

overtime = (
    (job_sat <= 2).astype(int) * 0.5 +
    (wlb <= 2).astype(int)    * 0.4 +
    np.random.random(N)       * 0.4
) > 0.7

# Encodings
dept_enc = pd.Categorical(dept_arr).codes
role_enc = pd.Categorical(role_arr).codes

# ── Attrition label (realistic logic) ────────────────────────────────────
attrition_score = (
    (overtime.astype(float) * 1.8) +
    ((5 - job_sat) * 0.5) +
    ((5 - wlb) * 0.4) +
    ((5 - env_sat) * 0.3) +
    (distance / 30 * 0.5) +
    (yrs_promotion / 15 * 0.6) +
    ((monthly_income < 30000).astype(float) * 1.0) +
    (num_companies / 10 * 0.4) +
    ((years_at_co <= 2).astype(float) * 0.8) +
    (np.random.random(N) * 0.5)
)
attrition_prob = 1 / (1 + np.exp(-(attrition_score - 3.5)))
attrition = (np.random.random(N) < attrition_prob).astype(int)

# ── Conflict label ────────────────────────────────────────────────────────
conflict_score = (
    ((5 - rel_sat) * 0.7) +
    ((5 - env_sat) * 0.5) +
    (overtime.astype(float) * 0.8) +
    ((5 - wlb) * 0.4) +
    ((yrs_manager <= 1).astype(float) * 0.6) +
    ((perf_rating == 3).astype(float) * 0.4) +
    (np.random.random(N) * 0.5)
)
conflict_prob = 1 / (1 + np.exp(-(conflict_score - 2.8)))
conflict = (np.random.random(N) < conflict_prob).astype(int)

# ── Wellbeing score (0–100) ────────────────────────────────────────────────
wellbeing = np.clip(
    job_sat * 10 +
    wlb     * 8 +
    env_sat * 7 +
    rel_sat * 6 +
    (stock_option * 3) +
    (training_times * 2) -
    (overtime.astype(float) * 15) -
    (distance / 30 * 10) -
    (yrs_promotion / 15 * 8) +
    np.random.normal(0, 5, N),
    0, 100
)

# ── Assemble DataFrame ────────────────────────────────────────────────────
df = pd.DataFrame({
    "EmployeeID":              np.arange(1, N+1),
    "Age":                     age,
    "Department":              dept_arr,
    "JobRole":                 role_arr,
    "Gender":                  gender_arr,
    "MaritalStatus":           marital_arr,
    "BusinessTravel":          travel_arr,
    "EducationField":          edu_field_arr,
    "Education":               education,
    "MonthlyIncome":           monthly_income,
    "DistanceFromHome":        distance,
    "NumCompaniesWorked":      num_companies,
    "YearsAtCompany":          years_at_co,
    "YearsSinceLastPromotion": yrs_promotion,
    "YearsWithCurrManager":    yrs_manager,
    "TrainingTimesLastYear":   training_times,
    "StockOptionLevel":        stock_option,
    "TotalWorkingYears":       total_working_yrs,
    "PerformanceRating":       perf_rating,
    "JobSatisfaction":         job_sat,
    "EnvironmentSatisfaction": env_sat,
    "RelationshipSatisfaction":rel_sat,
    "WorkLifeBalance":         wlb,
    "OverTime":                overtime.astype(int),
    "Attrition":               attrition,
    "Conflict":                conflict,
    "WellbeingScore":          wellbeing.round(1),
    "Department_encoded":      dept_enc,
    "JobRole_encoded":         role_enc,
})

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)
df.to_csv("data/hr_dataset.csv", index=False)
print(f"    ✓ Dataset saved: {N} employees, {df.shape[1]} features")
print(f"    ✓ Attrition rate: {attrition.mean():.1%}  |  Conflict rate: {conflict.mean():.1%}")
print(f"    ✓ Avg Wellbeing: {wellbeing.mean():.1f}/100")


# ── 2. FEATURE SELECTION ─────────────────────────────────────────────────────
print("\n[2/4] Preparing features …")

feature_cols = [
    "Age","Education","MonthlyIncome","DistanceFromHome","NumCompaniesWorked",
    "YearsAtCompany","YearsSinceLastPromotion","YearsWithCurrManager",
    "TrainingTimesLastYear","StockOptionLevel","TotalWorkingYears","PerformanceRating",
    "JobSatisfaction","EnvironmentSatisfaction","RelationshipSatisfaction",
    "WorkLifeBalance","OverTime","Department_encoded","JobRole_encoded",
]

X = df[feature_cols]
y_ret  = df["Attrition"]
y_conf = df["Conflict"]
y_well = df["WellbeingScore"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tr, X_te, yr_tr, yr_te = train_test_split(X_scaled, y_ret,  test_size=0.2, random_state=42, stratify=y_ret)
_,    _,    yc_tr, yc_te = train_test_split(X_scaled, y_conf, test_size=0.2, random_state=42, stratify=y_conf)
_,    _,    yw_tr, yw_te = train_test_split(X_scaled, y_well, test_size=0.2, random_state=42)

print(f"    ✓ Train: {len(X_tr)} samples  |  Test: {len(X_te)} samples")


# ── 3. TRAIN MODELS ──────────────────────────────────────────────────────────
print("\n[3/4] Training models …")

# Retention
print("    → Retention Risk model (GradientBoosting) …")
ret_model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
ret_model.fit(X_tr, yr_tr)
ret_pred = ret_model.predict(X_te)
print(f"      Accuracy: {(ret_pred == yr_te).mean():.1%}")
print(classification_report(yr_te, ret_pred, target_names=["Stay","Leave"], digits=3))

# Conflict
print("    → Conflict Risk model (RandomForest) …")
conf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
conf_model.fit(X_tr, yc_tr)
conf_pred = conf_model.predict(X_te)
print(f"      Accuracy: {(conf_pred == yc_te).mean():.1%}")
print(classification_report(yc_te, conf_pred, target_names=["Low","High"], digits=3))

# Wellbeing
print("    → Wellbeing Score model (GradientBoostingRegressor) …")
well_model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
well_model.fit(X_tr, yw_tr)
well_pred = well_model.predict(X_te)
mae = mean_absolute_error(yw_te, well_pred)
print(f"      MAE: {mae:.2f} points")


# ── 4. SAVE ──────────────────────────────────────────────────────────────────
print("\n[4/4] Saving models …")
joblib.dump(ret_model,    "models/retention_model.pkl")
joblib.dump(conf_model,   "models/conflict_model.pkl")
joblib.dump(well_model,   "models/wellbeing_model.pkl")
joblib.dump(scaler,       "models/scaler.pkl")
joblib.dump(feature_cols, "models/features.pkl")
print("    ✓ retention_model.pkl")
print("    ✓ conflict_model.pkl")
print("    ✓ wellbeing_model.pkl")
print("    ✓ scaler.pkl")
print("    ✓ features.pkl")

print("\n" + "=" * 55)
print("  ✅ All done! Now run:")
print("     streamlit run app.py")
print("=" * 55)
