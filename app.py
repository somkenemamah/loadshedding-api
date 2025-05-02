import joblib
import pandas as pd
from flask import Flask, request, jsonify

import threading

# Load the model
model = joblib.load("loadshedding_model.pkl")

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Load Shedding Predictor API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        # Extract inputs
        date = pd.to_datetime(data['date'], format='%Y%m%d')
        time = int(data['time'])
        block = int(data['block'])

        # Extract features
        hour = time // 100
        minute = time % 100
        dayofweek = date.dayofweek
        month = date.month

        features = [[hour, minute, dayofweek, month, block]]
        prediction = model.predict(features)[0]

        return jsonify({'predicted_stage': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

# Start Flask in a thread
def run_flask():
    app.run(port=5000)

thread = threading.Thread(target=run_flask)
thread.start()
