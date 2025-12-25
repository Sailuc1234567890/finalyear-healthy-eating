import joblib
import pandas as pd

# Load trained model
model = joblib.load("nutrition_risk_model.pkl")

# -------------------------------
# Risk Prediction + Risk Score
# -------------------------------
def predict_risk(input_df):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    risk_map = {0: "Healthy", 1: "Moderate", 2: "Unhealthy"}
    return risk_map[prediction], round(max(proba) * 100, 2)

# -------------------------------
# Nutrient Highlighting
# -------------------------------
def nutrient_issues(row):
    issues = []
    if row["Sugars (g)"] > 15:
        issues.append("High Sugar")
    if row["Fat (g)"] > 20:
        issues.append("High Fat")
    if row["Sodium (mg)"] > 500:
        issues.append("High Sodium")
    if row["Fiber (g)"] < 5:
        issues.append("Low Fiber")
    if row["Calories (kcal)"] > 400:
        issues.append("High Calories")
    return issues

# -------------------------------
# Diet Suggestions
# -------------------------------
def diet_suggestions(risk):
    if risk == "Unhealthy":
        return ["Reduce sugar", "Avoid fried foods", "Increase fiber intake"]
    elif risk == "Moderate":
        return ["Limit sodium", "Maintain balanced meals"]
    else:
        return ["Continue healthy eating habits"]

# -------------------------------
# What-if Analysis
# -------------------------------
def what_if_analysis(input_df, reduce_sugar=20):
    modified = input_df.copy()
    modified["Sugars (g)"] = max(0, modified["Sugars (g)"].iloc[0] - reduce_sugar)
    modified = modified[model.feature_names_in_]
    new_risk, _ = predict_risk(modified)
    return new_risk

# -------------------------------
# BMI Calculation
# -------------------------------
def calculate_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

# -------------------------------
# BMI-Based Risk Adjustment
# -------------------------------
def bmi_adjusted_risk(bmi, risk):
    if bmi >= 30 and risk != "Unhealthy":
        return "Unhealthy"
    elif bmi >= 25 and risk == "Healthy":
        return "Moderate"
    return risk

# -------------------------------
# Health Condition Based Diet
# -------------------------------
def condition_based_diet(condition):
    plans = {
        "Diabetes": ["Low sugar diet", "High fiber foods"],
        "BP": ["Low sodium diet", "Avoid processed food"],
        "Heart": ["Low fat diet", "Omega-3 rich foods"],
        "Obesity": ["Calorie-controlled diet", "High protein foods"],
        "None": ["Balanced diet"]
    }
    return plans.get(condition, plans["None"])

