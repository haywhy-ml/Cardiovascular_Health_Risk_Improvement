import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "cvd_model_bundle.pkl")

bundle = joblib.load(MODEL_PATH)
PIPELINE = bundle["pipeline"]
SCALER = bundle["scaler"]
LABEL_ENCODER = bundle["label_encoder"]
FEATURE_COLUMNS = bundle["feature_columns"]
THRESHOLD = bundle["chosen_threshold"]

GENERAL_HEALTH_MAP = {"Poor": 0, "Fair": 1, "Good": 2, "Very Good": 3, "Excellent": 4}
CHECKUP_MAP = {
    "Never": 0,
    "5 or more years ago": 1,
    "Within the past 5 years": 2,
    "Within the past 2 years": 3,
    "Within the past year": 4,
}
AGE_CATEGORY_ORDER = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
    "55-59", "60-64", "65-69", "70-74", "75-79", "80+",
]
AGE_CATEGORY_MAP = {label: i for i, label in enumerate(AGE_CATEGORY_ORDER)}
AGE_MID_MAP = {
    "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37, "40-44": 42,
    "45-49": 47, "50-54": 52, "55-59": 57, "60-64": 62, "65-69": 67,
    "70-74": 72, "75-79": 77, "80+": 85,
}
AGE_BIN_LABELS = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
AGE_BIN_MAP = {label: i for i, label in enumerate(AGE_BIN_LABELS)}
SEX_MAP = {"Female": 0, "Male": 1}
DIABETES_OPTIONS = [
    "No",
    "No, pre-diabetes or borderline diabetes",
    "Yes",
    "Yes, but female told only during pregnancy",
]
YES_NO_FIELDS = ["Exercise", "Skin_Cancer", "Other_Cancer", "Depression", "Arthritis", "Smoking_History"]

FORM_DEFAULTS = {
    "General_Health": "Good",
    "Checkup": "Within the past year",
    "Exercise": "Yes",
    "Skin_Cancer": "No",
    "Other_Cancer": "No",
    "Depression": "No",
    "Arthritis": "No",
    "Diabetes": "No",
    "Sex": "Female",
    "Age_Category": "45-49",
    "Height_cm": "165",
    "Weight_kg": "70",
    "Smoking_History": "No",
    "Alcohol_Consumption": "0",
    "Fruit_Consumption": "30",
    "Green_Vegetables_Consumption": "20",
    "FriedPotato_Consumption": "4",
}


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def age_bin_for_mid(age_mid):
    if age_mid < 30:
        return "<30"
    if age_mid < 40:
        return "30-39"
    if age_mid < 50:
        return "40-49"
    if age_mid < 60:
        return "50-59"
    if age_mid < 70:
        return "60-69"
    return "70+"


def build_feature_row(form):
    height_cm = float(form["Height_cm"])
    weight_kg = float(form["Weight_kg"])
    bmi = weight_kg / ((height_cm / 100) ** 2)
    bmi_cat = bmi_category(bmi)

    age_category = form["Age_Category"]
    age_mid = AGE_MID_MAP[age_category]
    age_bin = age_bin_for_mid(age_mid)

    diabetes = form["Diabetes"]
    comorbidity_count = sum(
        [
            diabetes == "Yes",
            form["Arthritis"] == "Yes",
            form["Depression"] == "Yes",
            form["Skin_Cancer"] == "Yes",
            form["Other_Cancer"] == "Yes",
        ]
    )

    row = {
        "General_Health": GENERAL_HEALTH_MAP[form["General_Health"]],
        "Checkup": CHECKUP_MAP[form["Checkup"]],
        "Exercise": 1 if form["Exercise"] == "Yes" else 0,
        "Skin_Cancer": 1 if form["Skin_Cancer"] == "Yes" else 0,
        "Other_Cancer": 1 if form["Other_Cancer"] == "Yes" else 0,
        "Depression": 1 if form["Depression"] == "Yes" else 0,
        "Arthritis": 1 if form["Arthritis"] == "Yes" else 0,
        "Sex": SEX_MAP[form["Sex"]],
        "Age_Category": AGE_CATEGORY_MAP[age_category],
        "Height_(cm)": height_cm,
        "Weight_(kg)": weight_kg,
        "BMI": bmi,
        "Smoking_History": 1 if form["Smoking_History"] == "Yes" else 0,
        "Alcohol_Consumption": float(form["Alcohol_Consumption"]),
        "Fruit_Consumption": float(form["Fruit_Consumption"]),
        "Green_Vegetables_Consumption": float(form["Green_Vegetables_Consumption"]),
        "FriedPotato_Consumption": float(form["FriedPotato_Consumption"]),
        "BMI_Category": {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}[bmi_cat],
        "Comorbidity_Count": comorbidity_count,
        "Age_Mid": age_mid,
        "Age_Bin": AGE_BIN_MAP[age_bin],
        "Diabetes_No, pre-diabetes or borderline diabetes": 1 if diabetes == "No, pre-diabetes or borderline diabetes" else 0,
        "Diabetes_Yes": 1 if diabetes == "Yes" else 0,
        "Diabetes_Yes, but female told only during pregnancy": 1 if diabetes == "Yes, but female told only during pregnancy" else 0,
    }
    return row, bmi, bmi_cat


def predict(form):
    row, bmi, bmi_cat = build_feature_row(form)
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    X_scaled = SCALER.transform(X)
    proba = float(PIPELINE.predict_proba(X_scaled)[:, 1][0])
    pred_class = int(proba >= THRESHOLD)
    label = LABEL_ENCODER.inverse_transform([pred_class])[0]
    return {
        "probability": proba,
        "percent": round(proba * 100, 1),
        "label": label,
        "at_risk": pred_class == 1,
        "bmi": round(bmi, 1),
        "bmi_category": bmi_cat,
        "threshold": round(THRESHOLD * 100, 1),
    }


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        form=FORM_DEFAULTS,
        general_health_options=list(GENERAL_HEALTH_MAP.keys()),
        checkup_options=list(CHECKUP_MAP.keys()),
        age_category_options=AGE_CATEGORY_ORDER,
        diabetes_options=DIABETES_OPTIONS,
        result=None,
        error=None,
    )


@app.route("/predict", methods=["POST"])
def predict_view():
    form = {**FORM_DEFAULTS, **request.form.to_dict()}
    error = None
    result = None
    try:
        result = predict(form)
    except (ValueError, KeyError) as exc:
        error = f"Couldn't process that input: {exc}"

    return render_template(
        "index.html",
        form=form,
        general_health_options=list(GENERAL_HEALTH_MAP.keys()),
        checkup_options=list(CHECKUP_MAP.keys()),
        age_category_options=AGE_CATEGORY_ORDER,
        diabetes_options=DIABETES_OPTIONS,
        result=result,
        error=error,
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
