from flask import Flask, request, jsonify
import pandas as pd
import pickle
from datetime import datetime

app = Flask(__name__)

# Load your model
with open("loadshedding_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Loadshedding Prediction API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        date_str = data["date"]
        time_str = data["time"]
        block = int(data["block"])
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid input format. Required: date, time, block"}), 400

    try:
        datetime_obj = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        features = pd.DataFrame({
            "year": [datetime_obj.year],
            "month": [datetime_obj.month],
            "day": [datetime_obj.day],
            "hour": [datetime_obj.hour],
            "minute": [datetime_obj.minute],
            "block": [block]
        })
        prediction = int(model.predict(features)[0])
        return jsonify({"predicted_stage": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
