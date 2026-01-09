from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ===============================
# LOAD ML OBJECTS
# ===============================
model = joblib.load("final_nutrition_model.pkl")
scaler = joblib.load("final_scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ===============================
# FOOD DATABASE (per 100g / serving)
# ===============================
FOOD_DB = {
    "rice": {"cal": 130, "carbs": 28, "protein": 2.7, "fat": 0.3, "fiber": 0.4, "sugar": 0.1, "sodium": 1},
    "dal": {"cal": 116, "carbs": 20, "protein": 9, "fat": 0.4, "fiber": 8, "sugar": 1.5, "sodium": 2},
    "tea": {"cal": 40, "carbs": 6, "protein": 1, "fat": 1, "fiber": 0, "sugar": 4, "sodium": 10},
    "chapati": {"cal": 120, "carbs": 18, "protein": 3, "fat": 2, "fiber": 3, "sugar": 0.4, "sodium": 120}
}

# ===============================
# DASHBOARD – DISEASE SELECTION
# ===============================
@app.route("/", methods=["GET"])
def dashboard():
    diseases = ["Diabetes", "Hypertension", "Obesity", "Heart Disease"]
    return render_template("dashboard.html", diseases=diseases)

# ===============================
# FOOD INPUT PAGE
# ===============================
@app.route("/food", methods=["POST"])
def food_page():
    disease = request.form["disease"]
    return render_template("food_input.html", disease=disease)

# ===============================
# PREDICT RISK
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    disease = request.form["disease"]

    total = {
        "cal": 0,
        "protein": 0,
        "carbs": 0,
        "fat": 0,
        "fiber": 0,
        "sugar": 0,
        "sodium": 0
    }

    # Calculate nutrition internally
    for food in FOOD_DB:
        qty = request.form.get(food)
        if qty and qty.strip() != "":
            qty = float(qty) / 100
            for n in total:
                total[n] += FOOD_DB[food][n] * qty

    # ML input
    ml_input = pd.DataFrame([{
        "Calories (kcal)": total["cal"],
        "Protein (g)": total["protein"],
        "Carbohydrates (g)": total["carbs"],
        "Fat (g)": total["fat"],
        "Fiber (g)": total["fiber"],
        "Sugars (g)": total["sugar"],
        "Sodium (mg)": total["sodium"],
        "Cholesterol (mg)": 0,
        "Category": 0,
        "Meal_Type": 0,
        "Disease": 0
    }])

    scaled = scaler.transform(ml_input)
    pred = model.predict(scaled)
    risk = label_encoder.inverse_transform(pred)[0]

    return render_template(
        "result.html",
        disease=disease,
        risk=risk,
        nutrients=total
    )

# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
