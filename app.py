import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Machine Failure Predictor",
    page_icon="⚙️",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "machine_failure_model.joblib"
DEFAULT_THRESHOLD = 0.3681144773248292

@st.cache_resource
def load_artifact():
    artifact = joblib.load(MODEL_PATH)

    # The project artifact may store the pipeline and threshold together.
    if isinstance(artifact, dict):
        pipeline = None
        for key in ("model", "pipeline", "estimator", "classifier", "best_model"):
            if key in artifact and hasattr(artifact[key], "predict"):
                pipeline = artifact[key]
                break

        if pipeline is None:
            # If the dictionary itself contains a fitted estimator under another key,
            # stop with a useful message rather than silently using the wrong object.
            raise ValueError(
                "Could not identify the fitted pipeline in the saved artifact. "
                f"Available keys: {list(artifact.keys())}"
            )

        threshold = DEFAULT_THRESHOLD
        for key in ("threshold", "selected_threshold", "decision_threshold"):
            if key in artifact:
                try:
                    threshold = float(artifact[key])
                    break
                except (TypeError, ValueError):
                    pass
        return pipeline, threshold

    if hasattr(artifact, "predict"):
        return artifact, DEFAULT_THRESHOLD

    raise ValueError("The saved file does not contain a recognizable prediction model.")

st.title("⚙️ Machine Failure Prediction")
st.write(
    "Enter the machine's operating measurements to estimate whether "
    "the model flags a possible failure."
)
st.info(
    "Student research prototype using the UCI AI4I 2020 Predictive "
    "Maintenance Dataset. It is not a certified industrial safety system."
)

with st.form("prediction_form"):
    st.subheader("Machine details")

    machine_type = st.selectbox(
        "Machine Type",
        options=["L", "M", "H"],
        format_func=lambda x: {
            "L": "L — Low quality",
            "M": "M — Medium quality",
            "H": "H — High quality"
        }[x]
    )

    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.number_input(
            "Air Temperature (K)", min_value=250.0, max_value=350.0,
            value=298.0, step=0.1
        )
        rotational_speed = st.number_input(
            "Rotational Speed (rpm)", min_value=0, max_value=5000,
            value=1500, step=10
        )
        tool_wear = st.number_input(
            "Tool Wear (minutes)", min_value=0, max_value=500,
            value=100, step=1
        )
    with col2:
        process_temp = st.number_input(
            "Process Temperature (K)", min_value=250.0, max_value=400.0,
            value=308.0, step=0.1
        )
        torque = st.number_input(
            "Torque (Nm)", min_value=0.0, max_value=100.0,
            value=40.0, step=0.1
        )

    submitted = st.form_submit_button(
        "Predict Machine Failure",
        type="primary",
        use_container_width=True
    )

if submitted:
    try:
        model, threshold = load_artifact()

        # These column names match the UCI AI4I 2020 dataset.
        input_data = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }])

        if hasattr(model, "predict_proba"):
            failure_score = float(model.predict_proba(input_data)[0][1])
            prediction = int(failure_score >= threshold)
            st.caption(f"Decision threshold used: {threshold:.4f}")
            st.metric("Model failure score", f"{failure_score:.1%}")
            st.progress(min(max(failure_score, 0.0), 1.0))
        else:
            prediction = int(model.predict(input_data)[0])
            failure_score = None

        st.subheader("Prediction result")
        if prediction == 1:
            st.error("⚠️ Possible Machine Failure")
            st.write(
                "The model classified this machine input as a potential failure. "
                "Arrange an appropriate inspection by a qualified technician."
            )
        else:
            st.success("✅ No Failure Flagged")
            st.write(
                "The model did not flag this input as a failure. "
                "This does not guarantee that the machine is safe."
            )

        st.caption(
            "This is a model prediction, not a real-time sensor reading or "
            "a guarantee of machine condition."
        )

    except Exception as exc:
        st.error("The model could not make a prediction.")
        st.exception(exc)

st.divider()
st.caption(
    "Model: Gradient Boosting • Dataset: UCI AI4I 2020 Predictive Maintenance"
)
