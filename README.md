# Machine Failure Prediction Website

This folder contains a Streamlit website for the trained machine-failure model.

## Files
- `app.py` — website interface and prediction code
- `machine_failure_model.joblib` — trained model artifact
- `requirements.txt` — pinned Python dependencies

## Run locally
1. Install Python 3.10 or 3.11.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the website:

   ```bash
   streamlit run app.py
   ```

6. Open the local address shown in the terminal (usually http://localhost:8501).

## Publish online with Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `machine_failure_model.joblib`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select the repository and branch, and set the main file to `app.py`.
5. Deploy. Streamlit will install the packages and publish a shareable URL.

## Important model compatibility note
The supplied model artifact was created with scikit-learn 1.6.1. The requirements file pins
that version because scikit-learn model files are not guaranteed to be compatible across
versions. If loading fails in deployment, check the exact versions used in the original
Colab notebook and match them in `requirements.txt`.

## Project limitations
The project README identifies the UCI AI4I 2020 dataset as synthetic. The model score is
not a calibrated real-world failure probability, and this student prototype is not a
certified industrial safety system. Test it with known sample rows from your original
dataset before presenting it as complete.

## Input column names
The app sends these dataset feature names to the saved pipeline:
- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

If your original notebook used different column names, update the DataFrame keys in `app.py`
to match the pipeline's expected feature names exactly.
