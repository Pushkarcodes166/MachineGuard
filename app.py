import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MachineGuard | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "machine_failure_model.pkl"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Trained model not found: machine_failure_model.pkl")
    st.info("Run `python train_model.py` to create the model first.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()


# ============================================================
# OPTIONAL BACKGROUND IMAGE
# ============================================================

background_files = [
    BASE_DIR / "assets" / "maintenance-bg.png",
    BASE_DIR / "assets" / "maintenance-bg.jpg",
    BASE_DIR / "assets" / "maintenance-bg.jpeg",
]

background_path = next(
    (path for path in background_files if path.exists()),
    None,
)

if background_path:
    try:
        extension = background_path.suffix.lower()
        image_type = "image/png" if extension == ".png" else "image/jpeg"

        encoded_image = base64.b64encode(
            background_path.read_bytes()
        ).decode("utf-8")

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image:
                    linear-gradient(
                        rgba(5, 15, 28, 0.94),
                        rgba(5, 15, 28, 0.96)
                    ),
                    url("data:{image_type};base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except OSError:
        pass


# ============================================================
# PROFESSIONAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #10263d, #173b58);
        border: 1px solid #294965;
        border-radius: 18px;
        padding: 32px;
        margin-bottom: 26px;
    }

    .eyebrow {
        color: #83c7ff;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 38px;
        margin: 10px 0;
    }

    .hero p {
        color: #c3d3e2;
        font-size: 15px;
        line-height: 1.7;
        max-width: 760px;
        margin-bottom: 0;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(150, 180, 210, 0.20);
        border-radius: 14px;
    }

    div.stButton > button {
        min-height: 48px;
        font-weight: 700;
        border-radius: 10px;
    }

    h2, h3 {
        letter-spacing: -0.3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DEMO SCENARIOS
# ============================================================

DEMO_SCENARIOS = {
    "Custom Inputs": None,

    "Demo 1 — Low Risk Example": {
        "Type": "L",
        "Air temperature [K]": 298.10,
        "Process temperature [K]": 308.60,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.00,
        "Tool wear [min]": 20,
    },

    "Demo 2 — Moderate Risk Example": {
        "Type": "L",
        "Air temperature [K]": 300.00,
        "Process temperature [K]": 310.00,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 70.00,
        "Tool wear [min]": 200,
    },

    "Demo 3 — High Risk Example": {
        "Type": "L",
        "Air temperature [K]": 300.00,
        "Process temperature [K]": 310.00,
        "Rotational speed [rpm]": 2500,
        "Torque [Nm]": 10.00,
        "Tool wear [min]": 150,
    },
}


def apply_demo_scenario():
    """Fill the input widgets with the selected example."""

    scenario_name = st.session_state["demo_scenario"]
    scenario = DEMO_SCENARIOS[scenario_name]

    if scenario is None:
        return

    st.session_state["machine_type"] = scenario["Type"]
    st.session_state["air_temp"] = scenario["Air temperature [K]"]
    st.session_state["process_temp"] = scenario["Process temperature [K]"]
    st.session_state["speed"] = scenario["Rotational speed [rpm]"]
    st.session_state["torque"] = scenario["Torque [Nm]"]
    st.session_state["wear"] = scenario["Tool wear [min]"]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("⚙️ MachineGuard")
    st.caption("PREDICTIVE MAINTENANCE")

    st.divider()

    st.markdown("### Model Information")
    st.write("**Algorithm:** Random Forest")
    st.write("**Problem:** Binary Classification")
    st.write("**Target:** Machine Failure")

    st.divider()

    st.markdown("### Risk Thresholds")
    st.success("LOW RISK — Below 40%")
    st.warning("MODERATE — 40% to below 70%")
    st.error("HIGH RISK — 70% or above")

    st.caption(
        "Risk levels are assigned by the application "
        "after the model estimates failure probability."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">INDUSTRIAL AI · PREDICTIVE MAINTENANCE</div>
        <h1>Predict Before It Fails.</h1>
        <p>
            Enter machine operating conditions and let the trained
            Random Forest model estimate failure probability.
            MachineGuard converts the result into a simple risk level
            to support maintenance decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GUIDED DEMO
# ============================================================

st.subheader("1. Choose a Demo Scenario")

st.write(
    "For your presentation, select one of the three examples. "
    "The selected values will automatically fill the input fields."
)

st.selectbox(
    "Demo preset",
    options=list(DEMO_SCENARIOS.keys()),
    index=0,
    key="demo_scenario",
    on_change=apply_demo_scenario,
)

selected_demo = st.session_state["demo_scenario"]

if selected_demo != "Custom Inputs":
    st.info(
        "Demo values loaded. Review the readings below and click "
        "**RUN FAILURE ANALYSIS** to see the actual model output."
    )
else:
    st.caption(
        "Choose Custom Inputs to enter your own machine readings."
    )


# ============================================================
# MACHINE INPUTS
# ============================================================

st.subheader("2. Machine Operating Parameters")

st.caption(
    "The model considers all six parameters together. "
    "A single parameter does not determine the final risk level."
)

col1, col2 = st.columns(2, gap="large")

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        options=["L", "M", "H"],
        key="machine_type",
        help="Machine type category in the training dataset.",
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=298.10,
        step=0.10,
        format="%.2f",
        key="air_temp",
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=308.60,
        step=0.10,
        format="%.2f",
        key="process_temp",
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=500,
        max_value=3000,
        value=1500,
        step=1,
        key="speed",
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.00,
        step=0.10,
        format="%.2f",
        key="torque",
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=20,
        step=1,
        key="wear",
    )


# ============================================================
# RISK ASSESSMENT FUNCTION
# ============================================================

def assess_risk(failure_probability):
    """
    Convert the model's failure probability into
    an application-defined risk category.
    """

    if failure_probability >= 0.70:
        return (
            "HIGH RISK",
            "The estimated failure probability is 70% or above.",
            "high",
        )

    if failure_probability >= 0.40:
        return (
            "MODERATE RISK",
            "The estimated failure probability is from 40% "
            "to below 70%.",
            "moderate",
        )

    return (
        "LOW RISK",
        "The estimated failure probability is below 40%.",
        "low",
    )


# ============================================================
# RUN PREDICTION
# ============================================================

st.write("")

run_prediction = st.button(
    "RUN FAILURE ANALYSIS",
    type="primary",
    use_container_width=True,
)

if run_prediction:

    if process_temperature <= air_temperature:
        st.warning(
            "Process temperature is normally expected to be higher "
            "than air temperature. Verify these readings before "
            "interpreting the result."
        )

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
        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]
        model_classes = list(model.classes_)

        normal_index = model_classes.index(0)
        failure_index = model_classes.index(1)

        normal_probability = float(probabilities[normal_index])
        failure_probability = float(probabilities[failure_index])

    except Exception as error:
        st.error(
            "Prediction failed. Check that the saved model was "
            "trained using the same six input features and target classes."
        )
        st.exception(error)
        st.stop()

    risk_level, risk_explanation, risk_style = assess_risk(
        failure_probability
    )

    st.divider()
    st.subheader("3. Analysis Result")

    # Predicted class and risk category are different outputs.
    if prediction == 1:
        st.error("### Predicted Condition: MACHINE FAILURE")
        st.write(
            "The model predicts the failure class for these operating "
            "conditions. This is a model prediction, not a confirmed "
            "physical failure."
        )
    else:
        st.success("### Predicted Condition: NORMAL OPERATION")
        st.write(
            "The model predicts the normal class for these operating "
            "conditions. This does not guarantee that the machine "
            "cannot fail."
        )

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "Failure Probability",
            f"{failure_probability * 100:.2f}%",
        )

    with result2:
        st.metric(
            "Normal Probability",
            f"{normal_probability * 100:.2f}%",
        )

    with result3:
        st.metric(
            "Risk Assessment",
            risk_level,
        )

    st.progress(failure_probability)

    if risk_style == "low":
        st.success(f"**Why Low Risk?** {risk_explanation}")
    elif risk_style == "moderate":
        st.warning(f"**Why Moderate Risk?** {risk_explanation}")
    else:
        st.error(f"**Why High Risk?** {risk_explanation}")

    # --------------------------------------------------------
    # SIMPLE EXPLANATION FOR DEMONSTRATION
    # --------------------------------------------------------

    with st.expander("Understand this result — simple explanation", expanded=True):
        st.markdown(
            """
            **Step 1 — Input:** The six machine readings are passed
            to the trained model.

            **Step 2 — Prediction:** The Random Forest predicts one
            of two classes: Normal (0) or Failure (1).

            **Step 3 — Probability:** The model estimates the
            probability for each class.

            **Step 4 — Risk category:** The application checks the
            failure probability against its thresholds.
            """
        )

        st.markdown("**Risk rules used by MachineGuard:**")

        st.markdown(
            f"""
            - **Below 40%:** Low Risk
            - **40% to below 70%:** Moderate Risk
            - **70% or above:** High Risk

            The current failure probability is
            **{failure_probability * 100:.2f}%**, so the application
            assigns **{risk_level}**.
            """
        )

        st.caption(
            "These risk bands are application-defined rules, not "
            "three separate classes learned by the Random Forest."
        )

    with st.expander("View the exact machine readings"):
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "MachineGuard is an educational decision-support demonstration. "
        "Model probabilities are estimates, not guaranteed real-world "
        "failure rates or a substitute for machine inspection."
    )


# ============================================================
# ABOUT SECTION
# ============================================================

st.divider()

st.subheader("About MachineGuard")

about1, about2, about3 = st.columns(3)

with about1:
    st.markdown("**Predictive Maintenance**")
    st.write(
        "Uses machine operating data to estimate potential "
        "equipment failure."
    )

with about2:
    st.markdown("**Random Forest**")
    st.write(
        "Combines multiple decision trees to make a classification "
        "and estimate class probabilities."
    )

with about3:
    st.markdown("**Decision Support**")
    st.write(
        "Displays a predicted condition and a simple risk category "
        "to help users interpret the result."
    )

st.divider()

st.caption(
    "MachineGuard | ML-Based Predictive Maintenance System"
)