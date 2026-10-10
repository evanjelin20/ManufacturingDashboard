import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Manufacturing Insights", page_icon="🏭", layout="wide")

# -----------------------------
# Demo data and model
# -----------------------------
@st.cache_data
def make_demo_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    machine = rng.choice(["Machine A", "Machine B", "Machine C"], n)
    shift = rng.choice(["Morning", "Evening", "Night"], n)
    temperature = rng.normal(72, 8, n).clip(45, 105)
    pressure = rng.normal(6.5, 1.1, n).clip(3, 10)
    speed = rng.normal(120, 18, n).clip(65, 175)
    material_moisture = rng.normal(8, 2.2, n).clip(1, 16)
    downtime = rng.gamma(2.0, 7.0, n).clip(0, 75)
    produced = rng.normal(420, 65, n).clip(180, 600).astype(int)

    # Transparent demo rule used to create example labels; not real factory data.
    risk_score = (
        (np.abs(temperature - 72) > 15).astype(int)
        + (np.abs(pressure - 6.5) > 2.0).astype(int)
        + (speed > 155).astype(int)
        + (material_moisture > 11).astype(int)
        + (downtime > 30).astype(int)
    )
    noise = rng.random(n) < 0.06
    defect = ((risk_score >= 2) ^ noise).astype(int)

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="D")
    return pd.DataFrame({
        "Date": rng.choice(dates, n),
        "Machine": machine,
        "Shift": shift,
        "Temperature (°C)": temperature.round(1),
        "Pressure (bar)": pressure.round(2),
        "Speed (units/min)": speed.round(1),
        "Material Moisture (%)": material_moisture.round(1),
        "Downtime (min)": downtime.round(1),
        "Units Produced": produced,
        "Defect": defect
    }).sort_values("Date")

@st.cache_resource
def train_demo_model(df):
    features = ["Temperature (°C)", "Pressure (bar)", "Speed (units/min)", "Material Moisture (%)", "Downtime (min)"]
    X = df[features]
    y = df["Defect"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    return model, features, score

df_all = make_demo_data()
model, features, demo_accuracy = train_demo_model(df_all)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏭 Factory Control")
st.sidebar.caption("Manufacturing Quality & Production Insights")
machine_options = ["All machines"] + sorted(df_all["Machine"].unique().tolist())
selected_machine = st.sidebar.selectbox("Filter by machine", machine_options)
selected_shift = st.sidebar.selectbox("Filter by shift", ["All shifts", "Morning", "Evening", "Night"])

df = df_all.copy()
if selected_machine != "All machines":
    df = df[df["Machine"] == selected_machine]
if selected_shift != "All shifts":
    df = df[df["Shift"] == selected_shift]

# -----------------------------
# Main dashboard
# -----------------------------
st.title("Manufacturing Quality & Production Insights")
st.markdown("A prototype dashboard for monitoring production output, downtime, and product-quality risk.")
st.info("Demo mode: this app uses synthetic sample data for demonstration, not data from a real factory.")

if df.empty:
    st.warning("No records match these filters.")
    st.stop()

total_units = int(df["Units Produced"].sum())
defect_rate = float(df["Defect"].mean() * 100)
avg_downtime = float(df["Downtime (min)"].mean())
quality_pass = 100 - defect_rate

c1, c2, c3, c4 = st.columns(4)
c1.metric("Units Produced", f"{total_units:,}")
c2.metric("Estimated Quality Pass Rate", f"{quality_pass:.1f}%")
c3.metric("Average Downtime", f"{avg_downtime:.1f} min")
c4.metric("Records Analysed", f"{len(df):,}")

st.divider()
left, right = st.columns(2)
with left:
    daily = df.groupby("Date", as_index=False)["Units Produced"].sum()
    fig = px.line(daily, x="Date", y="Units Produced", title="Production Output Trend", markers=True)
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    machine_summary = df.groupby("Machine", as_index=False).agg(
        Units=("Units Produced", "sum"),
        Defect_Rate=("Defect", "mean"),
        Downtime=("Downtime (min)", "mean")
    )
    machine_summary["Defect Rate (%)"] = machine_summary["Defect_Rate"] * 100
    fig = px.bar(machine_summary, x="Machine", y="Defect Rate (%)", title="Defect Rate by Machine", color="Machine")
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    shift_summary = df.groupby("Shift", as_index=False)["Units Produced"].sum()
    fig = px.pie(shift_summary, names="Shift", values="Units Produced", title="Production Share by Shift", hole=0.45)
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Machine Summary")
    display = machine_summary[["Machine", "Units", "Defect Rate (%)", "Downtime"]].copy()
    display["Defect Rate (%)"] = display["Defect Rate (%)"].map(lambda x: f"{x:.1f}%")
    display["Downtime"] = display["Downtime"].map(lambda x: f"{x:.1f} min")
    st.dataframe(display, use_container_width=True, hide_index=True)

st.divider()
st.header("AI Quality-Risk Demo")
st.write("Enter operating conditions to estimate whether a production record may be at higher defect risk.")
with st.form("prediction_form"):
    p1, p2, p3 = st.columns(3)
    temp = p1.number_input("Temperature (°C)", min_value=40.0, max_value=110.0, value=72.0, step=1.0)
    pressure = p2.number_input("Pressure (bar)", min_value=2.0, max_value=12.0, value=6.5, step=0.1)
    speed = p3.number_input("Speed (units/min)", min_value=50.0, max_value=200.0, value=120.0, step=1.0)
    p4, p5 = st.columns(2)
    moisture = p4.number_input("Material Moisture (%)", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
    down = p5.number_input("Downtime (min)", min_value=0.0, max_value=120.0, value=10.0, step=1.0)
    submitted = st.form_submit_button("Predict Quality Risk", type="primary")

if submitted:
    sample = pd.DataFrame([{
        "Temperature (°C)": temp,
        "Pressure (bar)": pressure,
        "Speed (units/min)": speed,
        "Material Moisture (%)": moisture,
        "Downtime (min)": down
    }])[features]
    probability = float(model.predict_proba(sample)[0][1])
    if probability >= 0.5:
        st.error(f"Predicted result: Higher defect risk ({probability:.0%} model-estimated risk)")
    else:
        st.success(f"Predicted result: Lower defect risk ({probability:.0%} model-estimated risk)")
    st.caption("This model is trained on generated demo labels. Validate with real, labelled factory data before using it for operational decisions.")

with st.expander("Model notes and limitations"):
    st.write(f"Model: Random Forest classifier. Holdout accuracy on generated demo data: {demo_accuracy:.1%}.")
    st.write("The reported score only measures performance on the generated demo dataset. It is not evidence of accuracy on real production data.")
    st.write("For a real deployment, replace the demo dataset with approved factory records, define labels with domain experts, and evaluate the model on a separate time-based test set.")

st.divider()
st.subheader("Download sample data")
st.download_button(
    "Download CSV dataset",
    data=df_all.to_csv(index=False).encode("utf-8"),
    file_name="manufacturing_demo_data.csv",
    mime="text/csv"
)
st.caption("Built with Python, Streamlit, Pandas, Plotly, and Scikit-learn.")
