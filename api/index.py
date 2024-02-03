import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the trained LSTM model and the scaler
model = load_model('lstms.keras')
scaler_features = joblib.load('scaler_features.gz')
scaler_target = joblib.load('scaler_target.gz')

@app.route('/predict', methods=['GET'])
def home():
    look_back = 3
    # Extract query parameter for the start date (format: 'YYYY-MM-DD')
    start_date_str = request.args.get('start_date', default=datetime.now().strftime('%Y-%m-%d'), type=str)
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    # Prepare input features for the next year using the start_date
    future_dates = [start_date + timedelta(days=x) for x in range(365)]
    future_features = np.array([[date.day, date.month, date.year] for date in future_dates])
    scaled_features = scaler_features.transform(future_features)

    future_predictions = []
    last_known_sequence = np.zeros((1, look_back, 3))  # Initialize with zeros or last known values

    for i in range(365):
        if i < look_back:
            # For initial days, use the scaled_features directly
            last_known_sequence[:, i, :] = scaled_features[i]
        else:
            # Prepare the input sequence for prediction
            input_seq = last_known_sequence.reshape(1, look_back, 3)
            
            # Predict the next step
            predicted_scaled = model.predict(input_seq)
            
            # Update the last_known_sequence for next prediction
            last_known_sequence = np.roll(last_known_sequence, -1, axis=1)
            last_known_sequence[:, -1, :] = scaled_features[i].reshape(1, 3)
            
            # Inverse transform the predicted 'Close' price
            predicted_close = scaler_target.inverse_transform(predicted_scaled)[0, 0]
            future_predictions.append(predicted_close)
            future_predictions = [float(value) for value in future_predictions]

    return jsonify({'start_date': start_date_str, 'predictions': future_predictions})

    return jsonify({'start_date': start_date_str, 'predictions': future_predictions})


