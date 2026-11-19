
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="centered"
)

# Load model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "flight_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# Model features
FEATURES = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN",
    "DEST",
    "DISTANCE",
    "DEP_DELAY"
]


# Application title
st.title("✈️ Flight Delay Prediction")

st.write(
    "Enter flight details below to predict the expected arrival delay."
)

st.divider()


# Input form
with st.form("flight_prediction_form"):

    st.subheader("Flight Details")

    FL_DATE = st.date_input(
        "Flight Date"
    )

    AIRLINE = st.text_input(
        "AIRLINE",
        placeholder="Example: AA"
    )

    col1, col2 = st.columns(2)

    with col1:
        ORIGIN = st.text_input(
            "ORIGIN",
            placeholder="Example: ATL"
        )

    with col2:
        DEST = st.text_input(
            "DEST",
            placeholder="Example: LAX"
        )

    DISTANCE = st.number_input(
        "DISTANCE",
        min_value=0.0,
        value=500.0,
        step=10.0
    )

    DEP_DELAY = st.number_input(
        "DEPT_DELAY",
        value=0.0,
        step=1.0
    )

    submit = st.form_submit_button(
        "Predict Flight Delay",
        use_container_width=True
    )


# Prediction
if submit:

    if not AIRLINE.strip() or not ORIGIN.strip() or not DEST.strip():

        st.error(
            "Please enter AIRLINE, ORIGIN and DEST codes."
        )

    else:

        try:

            # Prepare input data
            input_data = {
                "YEAR": FL_DATE.year,
                "MONTH": FL_DATE.month,
                "DAY": FL_DATE.day,
                "DAY_OF_WEEK": FL_DATE.weekday(),
                "AIRLINE": AIRLINE.strip().upper(),
                "ORIGIN": ORIGIN.strip().upper(),
                "DEST": DEST.strip().upper(),
                "DISTANCE": float(DISTANCE),
                "DEP_DELAY": float(DEP_DELAY)
            }

            # Convert dictionary into DataFrame
            input_df = pd.DataFrame(
                [input_data],
                columns=FEATURES
            )

            st.subheader("Input Details")

            st.dataframe(
                input_df,
                use_container_width=True
            )

            # Prediction
            prediction = model.predict(input_df)[0]

            prediction = float(prediction)

            st.divider()

            st.subheader("Prediction Result")

            st.metric(
                label="Predicted Arrival Delay",
                value=f"{prediction:.2f} minutes"
            )

            if prediction > 0:

                st.warning(
                    f"Expected arrival delay: {prediction:.2f} minutes"
                )

            elif prediction < 0:

                st.success(
                    f"Flight may arrive {abs(prediction):.2f} minutes early."
                )

            else:

                st.success(
                    "Flight is predicted to arrive on time."
                )

        except Exception as e:

            st.error(f"Prediction Error: {e}")
