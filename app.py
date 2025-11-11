from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# -----------------------------------------------------
# 1️⃣ Initialize Flask App
# -----------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TensorFlow warnings
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# -----------------------------------------------------
# 2️⃣ Home Route
# -----------------------------------------------------
@app.route('/')
def home():
    return render_template('frontend.html')

# -----------------------------------------------------
# 3️⃣ Load Model, Scaler, and Threshold
# -----------------------------------------------------
MODEL_PATH = 'water_quality_model.h5'
SCALER_PATH = 'scaler.pkl'
THRESHOLD_PATH = 'best_threshold.npy'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found. Please run train_model_balanced.py first.")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("❌ Scaler not found. Please run train_model_balanced.py first.")
if not os.path.exists(THRESHOLD_PATH):
    raise FileNotFoundError("❌ Threshold not found. Please run train_model_balanced.py first.")

print("✅ Loading trained model and scaler...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
BEST_THRESHOLD = float(np.load(THRESHOLD_PATH))
print(f"✅ Model loaded successfully! Using threshold: {BEST_THRESHOLD:.2f}")

# -----------------------------------------------------
# 4️⃣ Prediction Route (Hybrid WHO + AI)
# -----------------------------------------------------
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

        # WHO rule-based safety check
        unsafe_reasons = []
        if features['ph'] < 6.5 or features['ph'] > 8.5:
            unsafe_reasons.append("pH out of range (6.5–8.5)")
        if features['Hardness'] > 500:
            unsafe_reasons.append("Hardness > 500 mg/L")
        if features['Solids'] > 50000:
            unsafe_reasons.append("Solids > 50,000 mg/L")
        if features['Chloramines'] > 4:
            unsafe_reasons.append("Chloramines > 4 mg/L")
        if features['Sulfate'] > 400:
            unsafe_reasons.append("Sulfate > 400 mg/L")
        if features['Conductivity'] > 2000:
            unsafe_reasons.append("Conductivity > 2000 µS/cm")
        if features['Organic_carbon'] < 2.2 or features['Organic_carbon'] > 15:
            unsafe_reasons.append("Organic Carbon out of range (2.2–15 mg/L)")
        if features['Trihalomethanes'] < 0.738 or features['Trihalomethanes'] > 100:
            unsafe_reasons.append("Trihalomethanes out of range (0.738–100 µg/L)")
        if features['Turbidity'] > 5:
            unsafe_reasons.append("Turbidity > 5 NTU")

        if unsafe_reasons:
            message = f"⚠️ Water is not safe to drink (violates WHO limits: {', '.join(unsafe_reasons)})."
            prediction_binary = 0
        else:
            # Scale input for model
            input_scaled = scaler.transform(input_df)

            # Model prediction
            prediction_prob = model.predict(input_scaled)[0][0]
            prediction_binary = int(prediction_prob > BEST_THRESHOLD)

            # Final decision based on model
            if prediction_binary == 1:
                message = "💧 Water is safe to drink!"
            else:
                message = "⚠️ Water is not safe to drink."

        return jsonify({
            "prediction": message,
            "potability": prediction_binary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------------------------------
# 5️⃣ Run Flask App
# -----------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
