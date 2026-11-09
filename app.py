import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Machine Failure Predictor", page_icon="⚙️", layout="centered")
MODEL_PATH = Path(__file__).parent / "machine_failure_model.joblib"
DEFAULT_THRESHOLD = 0.3681144773248292

@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict):
        model = next(
            (artifact[k] for k in ("model", "pipeline", "estimator", "classifier", "best_model")
             if k in artifact and hasattr(artifact[k], "predict")),
            None
        )
        if model is None:
            raise ValueError(f"Could not find a fitted model in artifact keys: {list(artifact)}")
        threshold = DEFAULT_THRESHOLD
        for key in ("threshold", "selected_threshold", "decision_threshold"):
            if key in artifact:
                try:
                    threshold = float(artifact[key])
                    break
                except (TypeError, ValueError):
                    pass
        return model, threshold
    if hasattr(artifact, "predict"):
        return artifact, DEFAULT_THRESHOLD
    raise ValueError("The saved file does not contain a recognizable prediction model.")

st.title("⚙️ Machine Failure Prediction")
st.write("Enter the machine's operating measurements to estimate whether the model flags a possible failure.")
st.info("Student research prototype using the UCI AI4I 2020 Predictive Maintenance Dataset. It is not a certified industrial safety system.")

with st.form("prediction_form"):
    st.subheader("Machine details")
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"],
        format_func=lambda x: {"L": "L — Low quality", "M": "M — Medium quality", "H": "H — High quality"}[x])
    c1, c2 = st.columns(2)
    with c1:
        air_temp = st.number_input("Air Temperature (K)", 250.0, 350.0, 298.0, 0.1)
        rotational_speed = st.number_input("Rotational Speed (rpm)", 0, 5000, 1500, 10)
        tool_wear = st.number_input("Tool Wear (minutes)", 0, 500, 100, 1)
    with c2:
        process_temp = st.number_input("Process Temperature (K)", 250.0, 400.0, 308.0, 0.1)
        torque = st.number_input("Torque (Nm)", 0.0, 100.0, 40.0, 0.1)
    submitted = st.form_submit_button("Predict Machine Failure", type="primary", use_container_width=True)

if submitted:
    try:
        model, threshold = load_model()
        # IMPORTANT: these names match the feature columns expected by your saved pipeline.
        input_data = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature": air_temp,
            "Process temperature": process_temp,
            "Rotational speed": rotational_speed,
            "Torque": torque,
            "Tool wear": tool_wear
        }])

        # Reorder columns to the exact order recorded by the fitted pipeline, if available.
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            input_data = input_data.reindex(columns=list(expected))

        if hasattr(model, "predict_proba"):
            failure_score = float(model.predict_proba(input_data)[0][1])
            prediction = int(failure_score >= threshold)
            st.caption(f"Decision threshold used: {threshold:.4f}")
            st.metric("Model failure score", f"{failure_score:.1%}")
            st.progress(max(0.0, min(1.0, failure_score)))
        else:
            prediction = int(model.predict(input_data)[0])
            failure_score = None

        st.subheader("Prediction result")
        if prediction == 1:
            st.error("⚠️ Possible Machine Failure")
            st.write("The model classified this input as a potential failure. Arrange an appropriate inspection by a qualified technician.")
        else:
            st.success("✅ No Failure Flagged")
            st.write("The model did not flag this input as a failure. This does not guarantee that the machine is safe.")
        st.caption("This is a model prediction, not a real-time sensor reading or a guarantee of machine condition.")
    except Exception as exc:
        st.error("The model could not make a prediction.")
        st.exception(exc)

st.divider()
st.caption("Model: Gradient Boosting • Dataset: UCI AI4I 2020 Predictive Maintenance")
