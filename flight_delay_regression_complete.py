import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
DATA_PATH = r"C:\Users\bakki\Downloads\archive\flights_sample_3m.csv"

df = pd.read_csv(DATA_PATH, nrows=50000)

print("First five rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nDataset information:")
df.info()
print("\nStatistical summary:")
print(df.describe(include="all").T)
print("\nDuplicate rows:", df.duplicated().sum())
print("\nMissing values:")
print(df.isnull().sum())


# ---------------------------------------------------------
# 2. BASIC CLEANING AND TARGET PREPARATION
# ---------------------------------------------------------
target = "ARR_DELAY"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' was not found in the CSV.")

# Convert target to numeric and remove rows with missing target.
# Do this BEFORE imputing any feature columns.
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target]).copy()

# Remove cancelled and diverted flights. They do not represent
# normal completed-flight arrival-delay observations.
for col in ["CANCELLED", "DIVERTED"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df[col].fillna(0).eq(0)].copy()

print("\nShape after target/flight-status cleaning:", df.shape)
print("Missing target values:", df[target].isna().sum())


# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------
df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
df = df.dropna(subset=["FL_DATE"]).copy()

df["YEAR"] = df["FL_DATE"].dt.year
df["MONTH"] = df["FL_DATE"].dt.month
df["DAY"] = df["FL_DATE"].dt.day
df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek

# Optional cyclical time features can help represent time-of-day.
# CRS_DEP_TIME is HHMM, so convert it into minutes after midnight.
if "CRS_DEP_TIME" in df.columns:
    scheduled_dep = pd.to_numeric(df["CRS_DEP_TIME"], errors="coerce")
    hours = np.floor(scheduled_dep / 100)
    minutes = scheduled_dep % 100
    valid_time = hours.between(0, 23) & minutes.between(0, 59)
    dep_minutes = (hours * 60 + minutes).where(valid_time)
    df["DEP_TIME_SIN"] = np.sin(2 * np.pi * dep_minutes / 1440)
    df["DEP_TIME_COS"] = np.cos(2 * np.pi * dep_minutes / 1440)

df = df.drop(columns=["FL_DATE"], errors="ignore")


# ---------------------------------------------------------
# 4. FEATURE SELECTION
# ---------------------------------------------------------
# These features are known before the flight departs.
# Do NOT use DEP_DELAY, DEP_TIME, TAXI_OUT, WHEELS_OFF,
# ARR_TIME, ARR_DELAY, ELAPSED_TIME, AIR_TIME, actual delay
# causes, or other post-departure information as predictors.
candidate_features = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN",
    "DEST",
    "DISTANCE",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DEP_TIME_SIN",
    "DEP_TIME_COS",
]

features = [col for col in candidate_features if col in df.columns]

if not features:
    raise ValueError("None of the selected feature columns exist in the dataset.")

X = df[features].copy()
y = df[target].copy()

# Convert numeric-looking feature columns to numeric safely.
for col in X.columns:
    if col not in ["AIRLINE", "ORIGIN", "DEST"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")

print("\nSelected features:", features)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nFeature preview:")
print(X.head())


# ---------------------------------------------------------
# 5. SAVE CLEANED DATA
# ---------------------------------------------------------
# This is an exploratory cleaned copy. Model preprocessing is
# still fitted only on training data in the pipeline below.
cleaned_output = df.copy()
cleaned_output.to_csv("cleaned_flight_data.csv", index=False)
print("\nCleaned dataset saved as cleaned_flight_data.csv")


# ---------------------------------------------------------
# 6. MISSING VALUE SUMMARY
# ---------------------------------------------------------
print("\nMissing values in selected features:")
print(X.isnull().sum())
print("\nMissing percentage:")
print((X.isnull().mean() * 100).round(2))


# ---------------------------------------------------------
# 7. OUTLIER ANALYSIS (DIAGNOSTIC ONLY)
# ---------------------------------------------------------
numeric_feature_cols = X.select_dtypes(include=np.number).columns.tolist()

print("\nOutlier analysis using IQR:")
for col in numeric_feature_cols:
    values = X[col].dropna()
    if values.empty:
        print(f"{col}: no usable values")
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = ((values < lower) | (values > upper)).sum()
    print(f"{col}: {outlier_count} potential outliers")

    plt.figure(figsize=(8, 3))
    plt.boxplot(values, vert=False)
    plt.title(f"Outlier Analysis - {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Outliers are not automatically deleted or capped because valid
# flight distances/times can naturally have extreme values.


# ---------------------------------------------------------
# 8. CORRELATION ANALYSIS
# ---------------------------------------------------------
corr_data = X.select_dtypes(include=np.number).copy()
corr_data[target] = y.loc[corr_data.index]

plt.figure(figsize=(11, 7))
sns.heatmap(corr_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Numerical Feature Correlation")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 9. TRAIN-TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ---------------------------------------------------------
# 10. PREPROCESSING PIPELINE
# ---------------------------------------------------------
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nNumerical columns:", num_cols)
print("Categorical columns:", cat_cols)

num_process = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

cat_process = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []
if num_cols:
    transformers.append(("num", num_process, num_cols))
if cat_cols:
    transformers.append(("cat", cat_process, cat_cols))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# ---------------------------------------------------------
# 11. BASELINE MODEL - LINEAR REGRESSION
# ---------------------------------------------------------
base_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

base_model.fit(X_train, y_train)
base_pred = base_model.predict(X_test)

print("\n========== LINEAR REGRESSION RESULTS ==========")
print("MAE:", mean_absolute_error(y_test, base_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, base_pred)))
print("R2 Score:", r2_score(y_test, base_pred))


# ---------------------------------------------------------
# 12. CROSS-VALIDATION
# ---------------------------------------------------------
cv_scores = cross_val_score(
    base_model,
    X_train,
    y_train,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\nCross-validation R2 scores:", cv_scores)
print("Mean CV R2:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())


# ---------------------------------------------------------
# 13. RANDOM FOREST REGRESSION
# ---------------------------------------------------------
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n========== RANDOM FOREST RESULTS ==========")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R2 Score:", r2_score(y_test, rf_pred))


# ---------------------------------------------------------
# 14. HYPERPARAMETER TUNING - GRIDSEARCHCV
# ---------------------------------------------------------
param_grid = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5]
}

grid = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
print("Best cross-validation R2:", grid.best_score_)

final_model = grid.best_estimator_
final_pred = final_model.predict(X_test)


# ---------------------------------------------------------
# 15. FINAL MODEL EVALUATION
# ---------------------------------------------------------
final_mae = mean_absolute_error(y_test, final_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
final_r2 = r2_score(y_test, final_pred)

print("\n========== TUNED RANDOM FOREST RESULTS ==========")
print("MAE:", final_mae)
print("RMSE:", final_rmse)
print("R2 Score:", final_r2)


# ---------------------------------------------------------
# 16. MODEL COMPARISON
# ---------------------------------------------------------
results = []

model_predictions = {
    "Linear Regression": base_pred,
    "Random Forest": rf_pred,
    "Tuned Random Forest": final_pred,
}

for model_name, predictions in model_predictions.items():
    results.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions),
    })

results_df = pd.DataFrame(results).sort_values(
    by="R2 Score",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

plt.figure(figsize=(9, 5))
sns.barplot(data=results_df, x="R2 Score", y="Model")
plt.title("Model Comparison - R2 Score")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 17. FEATURE IMPORTANCE
# ---------------------------------------------------------
best_rf = final_model.named_steps["model"]
feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
importances = best_rf.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 15 important features:")
print(feature_importance.head(15).to_string(index=False))

top_features = feature_importance.head(10).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importance - Tuned Random Forest")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 18. ACTUAL VS PREDICTED PLOT
# ---------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, final_pred, alpha=0.4)
plt.xlabel("Actual Arrival Delay (minutes)")
plt.ylabel("Predicted Arrival Delay (minutes)")
plt.title("Actual vs Predicted Arrival Delay")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 19. SAVE FINAL MODEL AND FEATURE LIST
# ---------------------------------------------------------
joblib.dump(final_model, "flight_model.pkl")
joblib.dump(features, "flight_model_features.pkl")

print("\nModel saved successfully as flight_model.pkl")
print("Feature list saved as flight_model_features.pkl")


# ---------------------------------------------------------
# 20. SAMPLE PREDICTION
# ---------------------------------------------------------
sample_input = X_test.iloc[[0]].copy()
sample_prediction = final_model.predict(sample_input)[0]

print("\n========== SAMPLE PREDICTION ==========")
print("Input:")
print(sample_input)
print(f"Predicted arrival delay: {sample_prediction:.2f} minutes")
