from flask import Flask, request, jsonify

import joblib
import pandas as pd

# Load the model
model = joblib.load('loadshedding_model.pkl')

# Initialize Flask app
app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when the app is run

@app.route('/')
def home():
    return "Load Shedding Predictor API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract input values
    date = pd.to_datetime(data['date'], format='%Y%m%d')
    time = int(data['time'])
    load_block = int(data['block'])

    # Parse time and extract features
    hour = time // 100
    minute = time % 100
    day_of_week = date.dayofweek
    month = date.month

    # Prepare input for the model
    input_features = [[hour, minute, day_of_week, month, load_block]]
    prediction = model.predict(input_features)[0]

    return jsonify({'predicted_stage': int(prediction)})

