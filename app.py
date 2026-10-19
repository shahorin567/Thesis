import json
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Smart Irrigation",
    page_icon="🌱",
    layout="centered"
)

MODEL_PATH = Path("best_irrigation_model.joblib")
META_PATH = Path("model_metadata.json")

# Store these values in Streamlit Secrets, not in public code.
CHANNEL_ID = str(st.secrets["THINGSPEAK_CHANNEL_ID"])
READ_API_KEY = st.secrets.get("THINGSPEAK_READ_API_KEY", "")

FIELD_MAP = {
    "Temperature": "field1",
    "Humidity": "field2",
    "Soil Moisture Percentage": "field4",
    "RawLDR": "field5",
}

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    expected = metadata["feature_columns_in_required_order"]
    if expected != list(FIELD_MAP.keys()):
        raise ValueError(
            f"Model feature order mismatch. Metadata expects: {expected}"
        )
    return model, metadata

def get_latest_reading():
    url = (
        f"https://api.thingspeak.com/channels/"
        f"{CHANNEL_ID}/feeds/last.json"
    )
    params = {}
    if READ_API_KEY:
        params["api_key"] = READ_API_KEY

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    values = {}
    for feature, field in FIELD_MAP.items():
        if data.get(field) in (None, ""):
            raise ValueError(f"ThingSpeak has no value for {field}")
        values[feature] = float(data[field])

    return data, values

st.title("🌱 Smart Irrigation Dashboard")
st.caption("Live ESP32 readings + trained ML prediction")

if st.button("Refresh sensor readings", type="primary"):
    st.cache_data.clear()

try:
    model, metadata = load_model()
    data, values = get_latest_reading()

    X_live = pd.DataFrame(
        [[values[f] for f in metadata["feature_columns_in_required_order"]]],
        columns=metadata["feature_columns_in_required_order"]
    )

    prediction = str(model.predict(X_live)[0])

    st.subheader("Predicted irrigation level")
    st.metric("Model output", prediction.upper())

    col1, col2 = st.columns(2)
    col1.metric("Temperature", f"{values['Temperature']:.1f} °C")
    col2.metric("Humidity", f"{values['Humidity']:.1f}%")

    col3, col4 = st.columns(2)
    col3.metric(
        "Soil moisture",
        f"{values['Soil Moisture Percentage']:.1f}%"
    )
    col4.metric("Raw LDR", f"{values['RawLDR']:.0f}")

    st.write("**Sensor reading timestamp:**", data.get("created_at", "Unknown"))

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_live)[0]
        st.subheader("Model scores")
        st.dataframe(
            pd.DataFrame({
                "Class": model.classes_,
                "Probability": [round(float(p), 4) for p in probs]
            }),
            hide_index=True,
            use_container_width=True
        )

    st.info(
        "Prediction only: verify soil condition before irrigation. "
        "This dashboard does not control a water pump."
    )

except Exception as e:
    st.error(f"Could not generate prediction: {e}")
