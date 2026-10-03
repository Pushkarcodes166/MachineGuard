import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import base64

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MachineGuard | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "machine_failure_model.pkl"

BACKGROUND_FILES = [
    BASE_DIR / "assets" / "maintenance-bg.png",
    BASE_DIR / "assets" / "maintenance-bg.jpg",
    BASE_DIR / "assets" / "maintenance-bg.jpeg",
]

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error("machine_failure_model.pkl was not found.")
    st.info("Place the model file in the same folder as app.py.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.exception(exc)
    st.stop()

# ============================================================
# BACKGROUND IMAGE
# ============================================================

background_path = next(
    (p for p in BACKGROUND_FILES if p.exists()),
    None,
)

if background_path is not None:
    mime = "image/png" if background_path.suffix.lower() == ".png" else "image/jpeg"

    with open(background_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    st.markdown(
        f"""<style>
        .stApp {{
            background-image:
                linear-gradient(
                    90deg,
                    rgba(4,13,25,0.94) 0%,
                    rgba(5,18,32,0.88) 45%,
                    rgba(5,18,32,0.76) 100%
                ),
                url("data:{mime};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        </style>""",
        unsafe_allow_html=True,
    )

# ============================================================
# CSS
# ============================================================
# This is the ONLY HTML in this application and it contains
# CSS only. All visible content uses native Streamlit widgets.

st.markdown(
    """
<style>
.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background: rgba(4, 12, 23, 0.98);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] h1 {
    font-size: 25px;
    letter-spacing: -0.5px;
}

section[data-testid="stSidebar"] h3 {
    color: #72b8ff;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li {
    color: #aebed0;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(
            90deg,
            rgba(6,22,39,0.97),
            rgba(8,29,49,0.88)
        );
    border: 1px solid rgba(114,184,255,0.20);
    border-radius: 24px;
    box-shadow: 0 25px 70px rgba(0,0,0,0.35);
}

div[data-testid="stVerticalBlockBorderWrapper"] h1 {
    color: #ffffff;
    font-size: 48px;
    line-height: 1.05;
    letter-spacing: -1.5px;
    margin-top: 0;
}

div[data-testid="stVerticalBlockBorderWrapper"] h3 {
    color: #72b8ff;
    font-size: 12px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

div[data-testid="stVerticalBlockBorderWrapper"] p {
    color: #b7c8d9;
    line-height: 1.65;
}

label {
    color: #dce7f2 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div {
    background-color: rgba(4,15,28,0.96) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
}

div[data-testid="stNumberInput"] input {
    background-color: rgba(4,15,28,0.96) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}

div[data-testid="stNumberInput"] button {
    background-color: rgba(28,39,55,0.95) !important;
    color: #ffffff !important;
}

div.stButton > button {
    width: 100%;
    min-height: 52px;
    background: linear-gradient(135deg,#1976d2,#0d47a1);
    color: #ffffff;
    border: none;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.5px;
    box-shadow: 0 10px 30px rgba(25,118,210,0.25);
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(25,118,210,0.40);
}

div[data-testid="stMetric"] {
    background: rgba(7,21,37,0.94);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 17px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #8fa4b9 !important;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

div[data-testid="stExpander"] {
    background: rgba(7,20,36,0.92);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
}

@media (max-width: 768px) {
    div[data-testid="stVerticalBlockBorderWrapper"] h1 {
        font-size: 36px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("MACHINEGUARD")
    st.caption("PREDICTIVE MAINTENANCE")
    st.divider()

    st.subheader("Model")
    st.write("Random Forest Classifier")

    st.subheader("Problem Type")
    st.write("Binary Classification")

    st.subheader("Target")
    st.write("Machine Failure")

    st.subheader("Input Parameters")
    st.write("Temperature • Speed • Torque • Tool Wear")

    st.divider()
    st.caption("AI-assisted machine condition assessment")

# ============================================================
# HERO
# ============================================================

with st.container(border=True):
    st.markdown("### INDUSTRIAL AI • PREDICTIVE MAINTENANCE")
    st.title("Predict Before It Fails.")
    st.write(
        "MachineGuard analyzes critical machine operating "
        "conditions using machine learning to identify "
        "potential equipment failures before they disrupt "
        "production."
    )

# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Machine Operating Parameters")
st.caption("Enter the current operating conditions of the machine.")

left, right = st.columns(2, gap="large")

with left:
    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"],
        index=0,
        help="Machine product type.",
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=298.10,
        step=0.10,
        format="%.2f",
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=308.60,
        step=0.10,
        format="%.2f",
    )

with right:
    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=500,
        max_value=3000,
        value=1500,
        step=1,
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.00,
        step=0.10,
        format="%.2f",
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=0,
        step=1,
    )

st.write("")

predict = st.button(
    "RUN FAILURE ANALYSIS",
    use_container_width=True,
)

# ============================================================
# PREDICTION
# ============================================================

if predict:
    input_data = pd.DataFrame(
        {
            "Type": [machine_type],
            "Air temperature [K]": [air_temperature],
            "Process temperature [K]": [process_temperature],
            "Rotational speed [rpm]": [rotational_speed],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear],
        }
    )

    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        failure_probability = (
            float(probabilities[classes.index(1)])
            if 1 in classes
            else 0.0
        )

        normal_probability = (
            float(probabilities[classes.index(0)])
            if 0 in classes
            else 1.0 - failure_probability
        )

    except Exception as exc:
        st.error("Prediction failed.")
        st.exception(exc)
        st.stop()

    if failure_probability >= 0.70:
        risk_level = "HIGH RISK"
    elif failure_probability >= 0.40:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "LOW RISK"

    st.write("")
    st.subheader("Analysis Result")

    if int(prediction) == 1:
        st.error(
            "MACHINE FAILURE\n\n"
            "The model identifies an elevated probability of "
            "machine failure under the supplied operating conditions."
        )
    else:
        st.success(
            "NORMAL OPERATION\n\n"
            "The model predicts normal machine operation under "
            "the supplied operating conditions."
        )

    st.write("")

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.metric(
            "Failure Probability",
            f"{failure_probability * 100:.2f}%",
        )

    with c2:
        st.metric(
            "Normal Probability",
            f"{normal_probability * 100:.2f}%",
        )

    with c3:
        st.metric(
            "Risk Assessment",
            risk_level,
        )

    st.write("")
    st.subheader("Failure Probability")
    st.progress(failure_probability)
    st.caption(
        f"Estimated probability of machine failure: "
        f"{failure_probability * 100:.2f}%"
    )

    with st.expander("VIEW MACHINE PARAMETERS"):
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# ABOUT
# ============================================================

st.write("")
st.subheader("About MachineGuard")
st.caption("Machine learning based predictive maintenance system.")

a, b, c = st.columns(3, gap="large")

with a:
    st.markdown("#### Predictive Maintenance")
    st.write(
        "Machine learning identifies patterns in machine "
        "operating conditions that may indicate potential failure."
    )

with b:
    st.markdown("#### Random Forest")
    st.write(
        "A Random Forest classifier uses the machine operating "
        "parameters to predict machine failure."
    )

with c:
    st.markdown("#### Decision Support")
    st.write(
        "The prediction can assist maintenance teams in deciding "
        "whether further machine inspection may be appropriate."
    )

st.divider()
st.caption(
    "MACHINEGUARD • Predictive Maintenance • "
    "Machine Failure Prediction using Machine Learning"
)
