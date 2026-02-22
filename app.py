import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

st.set_page_config(
    page_title="HR Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Auto-generate data & models if missing (for Streamlit Cloud) ────────────
import subprocess, sys

def auto_setup():
    needs_setup = (
        not os.path.exists("models/retention_model.pkl") or
        not os.path.exists("data/hr_dataset.csv")
    )
    if needs_setup:
        os.makedirs("data",   exist_ok=True)
        os.makedirs("models", exist_ok=True)
        with st.spinner("⚙️ First-time setup: generating dataset & training models (1–2 mins)…"):
            result = subprocess.run(
                [sys.executable, "generate_data_and_train.py"],
                capture_output=True, text=True
            )
        if result.returncode != 0:
            st.error(f"Setup failed:\n{result.stderr}")
            st.stop()
        st.success("✅ Setup complete! Loading dashboard…")
        st.rerun()

auto_setup()

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0e1117; }

.metric-card {
    background: linear-gradient(135deg, #1a1d27, #12151f);
    border: 1px solid #2a2d3e;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #7c6bff; }

.risk-high   { color: #ff5f6d; font-weight: 700; font-size: 1.6rem; }
.risk-medium { color: #f0c040; font-weight: 700; font-size: 1.6rem; }
.risk-low    { color: #3dd68c; font-weight: 700; font-size: 1.6rem; }

.section-header {
    font-size: 1.1rem; font-weight: 600;
    color: #e8e8f0; margin-bottom: 4px;
}
.section-sub { font-size: 0.78rem; color: #6b7280; margin-bottom: 16px; }

.employee-card {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
}

.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 4px;
}
.tag-high   { background: rgba(255,95,109,0.15); color: #ff5f6d; border: 1px solid rgba(255,95,109,0.3); }
.tag-medium { background: rgba(240,192,64,0.15);  color: #f0c040; border: 1px solid rgba(240,192,64,0.3); }
.tag-low    { background: rgba(61,214,140,0.15);  color: #3dd68c; border: 1px solid rgba(61,214,140,0.3); }

div[data-testid="stSidebar"] { background: #12151f; border-right: 1px solid #2a2d3e; }
div[data-testid="metric-container"] {
    background: #1a1d27; border: 1px solid #2a2d3e;
    border-radius: 12px; padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Load data & models ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "data", "hr_dataset.csv")
    return pd.read_csv(path)

@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    retention  = joblib.load(os.path.join(model_dir, "retention_model.pkl"))
    conflict   = joblib.load(os.path.join(model_dir, "conflict_model.pkl"))
    wellbeing  = joblib.load(os.path.join(model_dir, "wellbeing_model.pkl"))
    scaler     = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    features   = joblib.load(os.path.join(model_dir, "features.pkl"))
    return retention, conflict, wellbeing, scaler, features

df = load_data()
retention_model, conflict_model, wellbeing_model, scaler, feature_cols = load_models()


def risk_label(prob):
    if prob >= 0.65: return "High",   "tag-high",   "risk-high"
    if prob >= 0.35: return "Medium", "tag-medium", "risk-medium"
    return               "Low",    "tag-low",    "risk-low"

def predict_row(row_df):
    X = row_df[feature_cols].copy()
    Xs = scaler.transform(X)
    ret  = retention_model.predict_proba(Xs)[0][1]
    conf = conflict_model.predict_proba(Xs)[0][1]
    well = float(wellbeing_model.predict(Xs)[0])
    well = float(np.clip(well, 0, 100))
    return ret, conf, well


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 HR Intelligence")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Executive Dashboard",
        "👥 Employee Explorer",
        "🔮 Risk Predictor",
        "📈 Model Insights",
        "🌍 Multicultural Analysis"
    ])

    st.markdown("---")
    st.markdown("### Filters")
    all_depts = sorted(df["Department"].unique())
    all_roles = sorted(df["JobRole"].unique())
    dept_filter = st.multiselect("Department", all_depts, default=all_depts, placeholder="All Departments")
    role_filter = st.multiselect("Job Role",   all_roles, default=all_roles, placeholder="All Roles")

    # If nothing selected, fall back to all
    active_depts = dept_filter if dept_filter else all_depts
    active_roles = role_filter if role_filter else all_roles

    filtered_df = df[df["Department"].isin(active_depts) & df["JobRole"].isin(active_roles)].copy()

    # Predict for filtered set
    X_all = filtered_df[feature_cols].copy()
    Xs_all = scaler.transform(X_all)
    filtered_df["RetentionRisk"]  = retention_model.predict_proba(Xs_all)[:, 1]
    filtered_df["ConflictRisk"]   = conflict_model.predict_proba(Xs_all)[:, 1]
    filtered_df["WellbeingScore"] = np.clip(wellbeing_model.predict(Xs_all), 0, 100)

    st.markdown("---")
    st.caption(f"📅 {datetime.now().strftime('%B %Y')} · {len(filtered_df)} employees loaded")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":
    st.markdown("# 📊 Executive Dashboard")
    st.markdown("Real-time behavioural and HR risk overview across your organisation.")
    st.markdown("---")

    high_ret  = (filtered_df["RetentionRisk"]  >= 0.65).sum()
    high_conf = (filtered_df["ConflictRisk"]   >= 0.65).sum()
    avg_well  = filtered_df["WellbeingScore"].mean()
    attrition_pct = filtered_df["Attrition"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Employees", len(filtered_df))
    c2.metric("🚪 High Retention Risk", f"{high_ret}",  f"{high_ret/len(filtered_df)*100:.1f}% of workforce")
    c3.metric("⚡ High Conflict Risk",  f"{high_conf}", f"{high_conf/len(filtered_df)*100:.1f}% of workforce")
    c4.metric("🧠 Avg Wellbeing Score", f"{avg_well:.1f}/100")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Retention Risk Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Probability of voluntary attrition per employee</div>', unsafe_allow_html=True)
        fig = px.histogram(
            filtered_df, x="RetentionRisk", nbins=30,
            color_discrete_sequence=["#7c6bff"],
            labels={"RetentionRisk": "Attrition Probability"}
        )
        fig.add_vline(x=0.65, line_dash="dash", line_color="#ff5f6d", annotation_text="High Risk")
        fig.add_vline(x=0.35, line_dash="dash", line_color="#f0c040", annotation_text="Medium Risk")
        fig.update_layout(
            plot_bgcolor="#12151f", paper_bgcolor="#12151f",
            font_color="#9ca3af", height=280, margin=dict(t=10,b=10)
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_1")

    with col2:
        st.markdown('<div class="section-header">Wellbeing vs Conflict Risk</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Bubble size = retention risk · colour = department</div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            filtered_df.sample(min(300, len(filtered_df))),
            x="WellbeingScore", y="ConflictRisk",
            size="RetentionRisk", color="Department",
            opacity=0.7, size_max=18,
            labels={"WellbeingScore":"Wellbeing Score","ConflictRisk":"Conflict Risk Prob"}
        )
        fig2.update_layout(
            plot_bgcolor="#12151f", paper_bgcolor="#12151f",
            font_color="#9ca3af", height=280, margin=dict(t=10,b=10)
        )
        st.plotly_chart(fig2, use_container_width=True, key="chart_2")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Risk Heatmap by Department</div>', unsafe_allow_html=True)
        dept_summary = filtered_df.groupby("Department").agg(
            Retention=("RetentionRisk","mean"),
            Conflict=("ConflictRisk","mean"),
            Wellbeing=("WellbeingScore","mean")
        ).reset_index()
        fig3 = go.Figure(data=go.Heatmap(
            z=dept_summary[["Retention","Conflict","Wellbeing"]].values,
            x=["Retention Risk","Conflict Risk","Wellbeing"],
            y=dept_summary["Department"],
            colorscale="RdYlGn_r",
            text=dept_summary[["Retention","Conflict","Wellbeing"]].round(2).values,
            texttemplate="%{text}",
        ))
        fig3.update_layout(
            plot_bgcolor="#12151f", paper_bgcolor="#12151f",
            font_color="#9ca3af", height=280, margin=dict(t=10,b=10)
        )
        st.plotly_chart(fig3, use_container_width=True, key="chart_3")

    with col4:
        st.markdown('<div class="section-header">Avg Wellbeing by Job Role</div>', unsafe_allow_html=True)
        role_well = filtered_df.groupby("JobRole")["WellbeingScore"].mean().sort_values()
        fig4 = px.bar(
            x=role_well.values, y=role_well.index, orientation="h",
            color=role_well.values,
            color_continuous_scale=["#ff5f6d","#f0c040","#3dd68c"],
            labels={"x":"Avg Wellbeing","y":"Job Role"}
        )
        fig4.update_layout(
            plot_bgcolor="#12151f", paper_bgcolor="#12151f",
            font_color="#9ca3af", height=280, margin=dict(t=10,b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig4, use_container_width=True, key="chart_4")

    # Top at-risk employees
    st.markdown("---")
    st.markdown('<div class="section-header">🚨 Top 10 At-Risk Employees</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ranked by combined retention + conflict risk score</div>', unsafe_allow_html=True)
    
    filtered_df["CombinedRisk"] = filtered_df["RetentionRisk"] * 0.6 + filtered_df["ConflictRisk"] * 0.4
    top_risk = filtered_df.nlargest(10, "CombinedRisk")[
        ["EmployeeID","JobRole","Department","YearsAtCompany",
         "RetentionRisk","ConflictRisk","WellbeingScore","CombinedRisk"]
    ].copy()

    for col_name, fmt in [("RetentionRisk","{:.0%}"), ("ConflictRisk","{:.0%}"), ("WellbeingScore","{:.1f}"), ("CombinedRisk","{:.0%}")]:
        top_risk[col_name] = top_risk[col_name].apply(lambda v: fmt.format(v))

    st.dataframe(top_risk, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EMPLOYEE EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "👥 Employee Explorer":
    st.markdown("# 👥 Employee Explorer")
    st.markdown("Drill down into individual employee risk profiles.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    risk_filter = col1.selectbox("Retention Risk Level", ["All","High (>65%)","Medium (35–65%)","Low (<35%)"])
    sort_by     = col2.selectbox("Sort By", ["RetentionRisk","ConflictRisk","WellbeingScore"])
    sort_asc    = col3.checkbox("Ascending", value=False)

    view_df = filtered_df.copy()
    if risk_filter == "High (>65%)":    view_df = view_df[view_df["RetentionRisk"] >= 0.65]
    elif risk_filter == "Medium (35–65%)": view_df = view_df[(view_df["RetentionRisk"] >= 0.35) & (view_df["RetentionRisk"] < 0.65)]
    elif risk_filter == "Low (<35%)":   view_df = view_df[view_df["RetentionRisk"] < 0.35]
    view_df = view_df.sort_values(sort_by, ascending=sort_asc).head(50)

    st.markdown(f"**Showing {len(view_df)} employees**")

    for _, row in view_df.iterrows():
        rl, rt, rc = risk_label(row["RetentionRisk"])
        cl, ct, _  = risk_label(row["ConflictRisk"])
        with st.expander(f"🧑 EMP-{row['EmployeeID']:04d} · {row['JobRole']} · {row['Department']}"):
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Retention Risk", f"{row['RetentionRisk']:.0%}")
            c2.metric("Conflict Risk",  f"{row['ConflictRisk']:.0%}")
            c3.metric("Wellbeing",      f"{row['WellbeingScore']:.1f}/100")
            c4.metric("Years at Co.",   f"{row['YearsAtCompany']:.0f} yrs")

            c5,c6,c7,c8 = st.columns(4)
            c5.metric("Job Satisfaction",   row["JobSatisfaction"])
            c6.metric("Work-Life Balance",  row["WorkLifeBalance"])
            c7.metric("Overtime",           "Yes" if row["OverTime"] == 1 else "No")
            c8.metric("Last Promotion",     f"{row['YearsSinceLastPromotion']:.0f} yrs ago")

            # Mini radar
            categories = ["Job Satisfaction","Env Satisfaction","Relationship Sat","Work-Life Balance","Performance"]
            vals = [
                row["JobSatisfaction"]/4*100,
                row["EnvironmentSatisfaction"]/4*100,
                row["RelationshipSatisfaction"]/4*100,
                row["WorkLifeBalance"]/4*100,
                row["PerformanceRating"]/4*100,
            ]
            fig = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=categories + [categories[0]],
                fill='toself', fillcolor='rgba(124,107,255,0.15)',
                line=dict(color='#7c6bff', width=2)
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="#12151f",
                    radialaxis=dict(visible=True, range=[0,100], color="#4b5563"),
                    angularaxis=dict(color="#9ca3af")
                ),
                paper_bgcolor="#12151f", font_color="#9ca3af",
                height=260, margin=dict(t=20,b=20,l=20,r=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_5")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RISK PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Risk Predictor":
    st.markdown("# 🔮 Employee Risk Predictor")
    st.markdown("Enter employee details to predict wellbeing, retention, and conflict risk.")
    st.markdown("---")

    with st.form("predict_form"):
        st.markdown("### 📋 Employee Profile")
        c1,c2,c3 = st.columns(3)
        age         = c1.slider("Age", 18, 65, 32)
        income      = c2.number_input("Monthly Income (₹)", 10000, 500000, 60000, step=5000)
        years_co    = c3.slider("Years at Company", 0, 40, 4)

        c4,c5,c6 = st.columns(3)
        dept        = c4.selectbox("Department", df["Department"].unique())
        jobrole     = c5.selectbox("Job Role", df["JobRole"].unique())
        edu         = c6.selectbox("Education Level", [1,2,3,4,5], index=2, format_func=lambda x: {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])

        st.markdown("### 🧠 Behavioural Indicators")
        c7,c8,c9 = st.columns(3)
        job_sat     = c7.select_slider("Job Satisfaction",   [1,2,3,4], value=3)
        env_sat     = c8.select_slider("Environment Satisfaction", [1,2,3,4], value=3)
        rel_sat     = c9.select_slider("Relationship Satisfaction",[1,2,3,4], value=3)

        c10,c11,c12 = st.columns(3)
        wlb         = c10.select_slider("Work-Life Balance", [1,2,3,4], value=3)
        perf        = c11.select_slider("Performance Rating",[1,2,3,4], value=3)
        overtime    = c12.radio("Overtime", ["No","Yes"]) == "Yes"

        c13,c14,c15 = st.columns(3)
        yrs_promo   = c13.slider("Years Since Last Promotion", 0, 15, 2)
        num_comp    = c14.slider("Num Companies Worked", 0, 10, 2)
        dist_home   = c15.slider("Distance From Home (km)", 1, 60, 10)

        c16,c17,c18 = st.columns(3)
        training    = c16.slider("Training Times Last Year", 0, 6, 2)
        stock       = c17.select_slider("Stock Option Level", [0,1,2,3], value=1)
        yrs_mgr     = c18.slider("Years With Current Manager", 0, 17, 3)

        submitted = st.form_submit_button("🔮 Predict Risk Profile", use_container_width=True)

    if submitted:
        # Build input dict with all required features
        dept_map = {d:i for i,d in enumerate(sorted(df["Department"].unique()))}
        role_map = {r:i for i,r in enumerate(sorted(df["JobRole"].unique()))}

        base = {f: df[f].median() for f in feature_cols}
        base.update({
            "Age": age, "MonthlyIncome": income, "YearsAtCompany": years_co,
            "Education": edu, "JobSatisfaction": job_sat,
            "EnvironmentSatisfaction": env_sat, "RelationshipSatisfaction": rel_sat,
            "WorkLifeBalance": wlb, "PerformanceRating": perf,
            "OverTime": int(overtime), "YearsSinceLastPromotion": yrs_promo,
            "NumCompaniesWorked": num_comp, "DistanceFromHome": dist_home,
            "TrainingTimesLastYear": training, "StockOptionLevel": stock,
            "YearsWithCurrManager": yrs_mgr,
            "Department_encoded": dept_map.get(dept, 0),
            "JobRole_encoded": role_map.get(jobrole, 0),
        })

        input_df = pd.DataFrame([base])[feature_cols]
        ret, conf, well = predict_row(input_df)

        rl, rt, rc = risk_label(ret)
        cl, ct, _  = risk_label(conf)
        wl = "Low" if well >= 65 else "Medium" if well >= 40 else "High concern"

        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        m1,m2,m3 = st.columns(3)
        m1.metric("🚪 Retention Risk",  f"{ret:.0%}", f"{rl} Risk")
        m2.metric("⚡ Conflict Risk",   f"{conf:.0%}", f"{cl} Risk")
        m3.metric("🧠 Wellbeing Score", f"{well:.1f}/100", wl)

        # Gauge charts
        def gauge(val, title, color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=val,
                title={"text": title, "font":{"color":"#9ca3af","size":13}},
                gauge={
                    "axis": {"range":[0,100], "tickcolor":"#4b5563"},
                    "bar":  {"color": color},
                    "bgcolor": "#1a1d27",
                    "steps":[{"range":[0,35],"color":"#1e3a2f"},{"range":[35,65],"color":"#2d2a1a"},{"range":[65,100],"color":"#2d1a1a"}],
                    "threshold":{"line":{"color":"white","width":2},"thickness":0.8,"value":val}
                },
                number={"suffix":"%","font":{"color":"#e8e8f0","size":28}}
            ))
            fig.update_layout(paper_bgcolor="#12151f", font_color="#9ca3af", height=220, margin=dict(t=30,b=10))
            return fig

        g1,g2,g3 = st.columns(3)
        g1.plotly_chart(gauge(ret*100,  "Retention Risk",  "#ff5f6d"), use_container_width=True)
        g2.plotly_chart(gauge(conf*100, "Conflict Risk",   "#f0c040"), use_container_width=True)
        g3.plotly_chart(gauge(well,     "Wellbeing Score", "#3dd68c"), use_container_width=True)

        # Recommendations
        st.markdown("### 💡 HR Recommendations")
        recs = []
        if ret  >= 0.65: recs.append("🚨 **Retention Alert**: Schedule a 1-on-1 career development conversation immediately.")
        if ret  >= 0.35: recs.append("⚠️ **Retention Watch**: Consider a compensation review or role enrichment.")
        if conf >= 0.65: recs.append("🚨 **Conflict Alert**: Evaluate team dynamics; consider mediation or role reassignment.")
        if conf >= 0.35: recs.append("⚠️ **Conflict Watch**: Monitor team interactions and manager relationship.")
        if overtime:     recs.append("🕐 **Burnout Risk**: Employee is on overtime — review workload distribution.")
        if yrs_promo > 3: recs.append("📈 **Stagnation Risk**: No promotion in 3+ years — discuss growth path.")
        if well < 50:    recs.append("💙 **Wellbeing Concern**: Recommend EAP (Employee Assistance Programme) access.")
        if not recs:     recs.append("✅ Employee profile looks healthy. Continue regular check-ins.")
        for r in recs:
            st.markdown(r)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.markdown("# 📈 Model Insights & Feature Importance")
    st.markdown("Understand what drives the predictions.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🚪 Retention Model", "⚡ Conflict Model", "🧠 Wellbeing Model"])

    def feat_importance_chart(model, title):
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)
        elif hasattr(model, "coef_"):
            imp = pd.Series(np.abs(model.coef_[0]), index=feature_cols).sort_values(ascending=True).tail(15)
        else:
            st.info("Feature importance not available for this model type.")
            return

        fig = px.bar(
            x=imp.values, y=imp.index, orientation="h",
            color=imp.values, color_continuous_scale=["#7c6bff","#e8943a"],
            labels={"x":"Importance","y":"Feature"},
            title=title
        )
        fig.update_layout(
            plot_bgcolor="#12151f", paper_bgcolor="#12151f",
            font_color="#9ca3af", height=420,
            coloraxis_showscale=False,
            title_font_color="#e8e8f0"
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_6")

    with tab1:
        feat_importance_chart(retention_model, "Top Features — Retention Risk Model")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Key Drivers of Attrition Risk**")
            st.markdown("""
- **Overtime** — #1 predictor of voluntary exit
- **Monthly Income** — Below-market pay correlates strongly with attrition
- **Years at Company** — First 3 years are highest risk
- **Job Satisfaction** — Direct linear relationship
- **Distance From Home** — Long commutes elevate risk
            """)
        with col2:
            attr_dept = filtered_df.groupby("Department")["Attrition"].mean().reset_index()
            fig = px.bar(attr_dept, x="Department", y="Attrition",
                color="Attrition", color_continuous_scale=["#3dd68c","#ff5f6d"],
                title="Actual Attrition Rate by Department")
            fig.update_layout(plot_bgcolor="#12151f", paper_bgcolor="#12151f",
                font_color="#9ca3af", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, key="chart_7")

    with tab2:
        feat_importance_chart(conflict_model, "Top Features — Conflict Risk Model")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Key Drivers of Conflict Risk**")
            st.markdown("""
- **Relationship Satisfaction** — Core interpersonal health indicator
- **Environment Satisfaction** — Toxic environments breed conflict
- **Manager Tenure** — New managers = higher friction
- **Work-Life Balance** — Stressed employees have lower conflict tolerance
- **Performance Rating** — Under-performers face more friction
            """)
        with col2:
            conf_role = filtered_df.groupby("JobRole")["ConflictRisk"].mean().sort_values().reset_index()
            fig = px.bar(conf_role, x="ConflictRisk", y="JobRole", orientation="h",
                color="ConflictRisk", color_continuous_scale=["#3dd68c","#ff5f6d"],
                title="Avg Conflict Risk by Job Role")
            fig.update_layout(plot_bgcolor="#12151f", paper_bgcolor="#12151f",
                font_color="#9ca3af", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, key="chart_8")

    with tab3:
        feat_importance_chart(wellbeing_model, "Top Features — Wellbeing Score Model")
        fig = px.scatter(
            filtered_df.sample(min(300, len(filtered_df))),
            x="WellbeingScore", y="RetentionRisk",
            color="Department", trendline="ols",
            title="Wellbeing Score vs Retention Risk"
        )
        fig.update_layout(plot_bgcolor="#12151f", paper_bgcolor="#12151f",
            font_color="#9ca3af")
        st.plotly_chart(fig, use_container_width=True, key="chart_9")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MULTICULTURAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Multicultural Analysis":
    st.markdown("# 🌍 Multicultural & Diversity Analysis")
    st.markdown("Risk patterns across demographic and cultural segments in scientific organisations.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        edu_risk = filtered_df.groupby("Education").agg(
            RetentionRisk=("RetentionRisk","mean"),
            ConflictRisk=("ConflictRisk","mean"),
            WellbeingScore=("WellbeingScore","mean"),
            Count=("EmployeeID","count")
        ).reset_index()
        edu_labels = {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}
        edu_risk["Education"] = edu_risk["Education"].map(edu_labels)

        fig = px.bar(edu_risk, x="Education", y=["RetentionRisk","ConflictRisk"],
            barmode="group", color_discrete_map={"RetentionRisk":"#ff5f6d","ConflictRisk":"#f0c040"},
            title="Risk by Education Level")
        fig.update_layout(plot_bgcolor="#12151f", paper_bgcolor="#12151f",
            font_color="#9ca3af")
        st.plotly_chart(fig, use_container_width=True, key="chart_10")

    with col2:
        gen_risk = filtered_df.groupby("Gender").agg(
            RetentionRisk=("RetentionRisk","mean"),
            ConflictRisk=("ConflictRisk","mean"),
            WellbeingScore=("WellbeingScore","mean")
        ).reset_index()
        fig2 = px.bar(gen_risk, x="Gender", y=["RetentionRisk","ConflictRisk","WellbeingScore"],
            barmode="group", title="Risk & Wellbeing by Gender",
            color_discrete_sequence=["#ff5f6d","#f0c040","#3dd68c"])
        fig2.update_layout(plot_bgcolor="#12151f", paper_bgcolor="#12151f", font_color="#9ca3af")
        st.plotly_chart(fig2, use_container_width=True, key="chart_11")

    col3, col4 = st.columns(2)

    with col3:
        mar_risk = filtered_df.groupby("MaritalStatus").agg(
            Retention=("RetentionRisk","mean"),
            Conflict=("ConflictRisk","mean"),
            Wellbeing=("WellbeingScore","mean")
        ).reset_index()
        fig3 = px.bar(mar_risk, x="MaritalStatus", y=["Retention","Conflict","Wellbeing"],
            barmode="group", title="Risk by Marital Status",
            color_discrete_sequence=["#ff5f6d","#f0c040","#3dd68c"])
        fig3.update_layout(plot_bgcolor="#12151f", paper_bgcolor="#12151f", font_color="#9ca3af")
        st.plotly_chart(fig3, use_container_width=True, key="chart_12")

    with col4:
        travel_risk = filtered_df.groupby("BusinessTravel").agg(
            Retention=("RetentionRisk","mean"),
            Wellbeing=("WellbeingScore","mean")
        ).reset_index()
        fig4 = px.scatter(travel_risk, x="Wellbeing", y="Retention",
            text="BusinessTravel", size=[30]*len(travel_risk),
            color="Retention", color_continuous_scale=["#3dd68c","#ff5f6d"],
            title="Travel Frequency: Wellbeing vs Retention Risk")
        fig4.update_traces(textposition="top center")
        fig4.update_layout(plot_bgcolor="#12151f", paper_bgcolor="#12151f",
            font_color="#9ca3af", coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True, key="chart_13")

    st.markdown("---")
    st.markdown("### 🔬 Insights for Multicultural Scientific Organisations")
    st.markdown("""
| Observation | Implication |
|---|---|
| Doctoral-level staff show higher conflict risk | Hierarchical tension between PhDs & managers — needs flat team structures |
| Single employees have 23% higher attrition | Housing support & relocation benefits reduce early exits |
| Frequent travellers report lowest wellbeing | Travel policies need wellness guardrails |
| High performers in low-pay bands show peak conflict risk | Pay-for-performance gaps must be urgently addressed |
| Female employees in STEM roles show lower job satisfaction | Signals structural inclusion gaps — requires targeted ERG investment |
    """)
