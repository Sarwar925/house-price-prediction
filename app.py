from pathlib import Path
import pickle
import numpy as np
from flask import Flask, render_template, request

# Paths
BASE = Path(r"B:\Projects\House Price Prediction")
MODEL_PATH = BASE / "models" / "model.pkl"
PREP_PATH = BASE / "models" / "preprocess.pkl"

# Flask App
app = Flask(__name__)

# Load best model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load preprocessing info
with open(PREP_PATH, "rb") as f:
    prep = pickle.load(f)

FEATURES = prep["feature_names"]
MEDIANS = prep["medians"]
BEST_MODEL_NAME = prep.get("best_model_name", "Unknown")
METRICS = prep.get("metrics", {})

# Only show a subset of fields in form (you can expand if you want)
FORM_FIELDS = [
    "GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF",
    "YearBuilt", "FullBath", "BedroomAbvGr", "LotArea"
]
FORM_FIELDS = [f for f in FORM_FIELDS if f in FEATURES]

def build_feature_vector(form):
    """Fill missing with medians, overwrite with user input"""
    x = {feat: MEDIANS.get(feat, 0.0) for feat in FEATURES}
    for field in FORM_FIELDS:
        val = form.get(field, "").strip()
        if val != "":
            try:
                x[field] = float(val)
            except:
                pass
    return np.array([x[f] for f in FEATURES], dtype=float).reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction, error = None, None

    if request.method == "POST":
        try:
            X = build_feature_vector(request.form)
            y_pred = model.predict(X)[0]
            prediction = round(float(y_pred), 2)
        except Exception as e:
            error = f"Prediction error: {e}"

    # Format metrics for display
    metrics_display = {
        name: {
            "R2": f"{m['R2']:.4f}",
            "RMSE": f"{m['RMSE']:.2f}",
            "MAE": f"{m['MAE']:.2f}"
        } for name, m in METRICS.items()
    }

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        best_model=BEST_MODEL_NAME,
        metrics=metrics_display,
        fields=FORM_FIELDS
    )

if __name__ == "__main__":
    app.run(debug=True)
