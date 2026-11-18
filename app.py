
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(title="Flight Delay Prediction")

BASE_DIR = Path(__file__).resolve().parent

# Load model
model = joblib.load(BASE_DIR / "flight_model.pkl")

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

# Check model columns
model_features = getattr(model, "feature_names_in_", None)

if model_features is not None and list(model_features) != FEATURES:
    raise ValueError(
        f"Expected {FEATURES}, got {list(model_features)}"
    )


# Home page - directly serve HTML
@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = BASE_DIR / "templates" / "index.html"

    if not html_path.exists():
        return HTMLResponse(
            "<h2>index.html not found. Check your templates folder.</h2>",
            status_code=404
        )

    return HTMLResponse(
        content=html_path.read_text(encoding="utf-8")
    )


# Prediction
@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()

        required = [
            "FL_DATE", "AIRLINE", "ORIGIN",
            "DEST", "DISTANCE", "DEP_DELAY"
        ]

        missing = [
            col for col in required
            if col not in data or data[col] is None or data[col] == ""
        ]

        if missing:
            return JSONResponse(
                status_code=400,
                content={"error": f"Missing fields: {missing}"}
            )

        flight_date = pd.to_datetime(
            data["FL_DATE"], errors="coerce"
        )

        if pd.isna(flight_date):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid flight date"}
            )

        input_data = {
            "YEAR": FL_DATE.year,
            "MONTH": FL_DATE.month,
            "DAY": FL_DATE.day,
            "DAY_OF_WEEK": FL_DATE.dayofweek,
            "AIRLINE": str(data["AIRLINE"]).strip(),
            "ORIGIN": str(data["ORIGIN"]).strip(),
            "DEST": str(data["DEST"]).strip(),
            "DISTANCE": float(data["DISTANCE"]),
            "DEP_DELAY": float(data["DEP_DELAY"])
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=FEATURES
        )

        prediction = model.predict(input_df)[0]

        return {
            "status": "Success",
            "predicted_arrival_delay": round(float(prediction), 2),
            "unit": "minutes",
            "message": (
                "Predicted flight delay"
                if prediction > 0
                else "Predicted flight is on time or early"
            )
        }

    except (ValueError, TypeError) as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/health")
async def health():
    return {"status": "Application running successfully"}


