from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

# Initialize Flask
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Load saved model and scaler
MODEL_PATH = 'water_quality_model.h5'
SCALER_PATH = 'scaler.pkl'
THRESHOLD_PATH = 'best_threshold.npy'

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
BEST_THRESHOLD = float(np.load(THRESHOLD_PATH))

print("✅ Model and scaler loaded successfully.")

# Home route - render frontend
@app.route('/')
def home():
    return render_template('frontend.html')

# WHO Rule-based check + Model Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json(force=True)
        if 'features' not in input_data:
            return jsonify({"error": "Missing 'features' in request data"}), 400

        features = input_data['features']

        # Convert to DataFrame for scaling
        input_df = pd.DataFrame([features], columns=[
            'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ])

        # WHO check
        unsafe_conditions = (
            (features['ph'] < 6.5 or features['ph'] > 8.5) or
            (features['Hardness'] > 500) or
            (features['Solids'] > 50000) or
            (features['Chloramines'] > 4) or
            (features['Sulfate'] > 400) or
            (features['Conductivity'] > 2000) or
            (features['Organic_carbon'] < 2.2 or features['Organic_carbon'] > 15) or
            (features['Trihalomethanes'] < 0.738 or features['Trihalomethanes'] > 100) or
            (features['Turbidity'] > 5)
        )

        if unsafe_conditions:
            return jsonify({
                "prediction": "⚠️ Water is not safe to drink (violates WHO standards).",
                "potability": 0
            })

        # Scale & predict
        input_scaled = scaler.transform(input_df)
        prediction_prob = model.predict(input_scaled)[0][0]
        prediction_binary = int(prediction_prob > BEST_THRESHOLD)

        message = "💧 Water is safe to drink!" if prediction_binary == 1 else "⚠️ Water is not safe to drink."

        return jsonify({
            "prediction": message,
            "potability": prediction_binary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Render uses $PORT environment variable
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
