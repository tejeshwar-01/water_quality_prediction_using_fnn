from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Serve frontend
@app.route('/')
def home():
    return render_template('frontend.html')


# -------------------------------
# Load and prepare dataset
# -------------------------------
data = pd.read_csv('water_potability.csv')


def apply_who_standards(data):
    """Apply WHO safety labeling"""
    data['Potability'] = 0
    data.loc[
        (data['ph'] >= 6.5) & (data['ph'] <= 8.5) &
        (data['Hardness'] <= 500) &
        (data['Solids'] <= 50000) &
        (data['Chloramines'] <= 4) &
        (data['Sulfate'] <= 400) &
        (data['Conductivity'] <= 2000) &
        (data['Organic_carbon'] >= 2.2) & (data['Organic_carbon'] <= 15) &
        (data['Trihalomethanes'] >= 0.738) & (data['Trihalomethanes'] <= 100) &
        (data['Turbidity'] <= 5),
        'Potability'
    ] = 1
    return data


# Clean data
data = apply_who_standards(data)
for col in ['ph', 'Sulfate', 'Trihalomethanes']:
    data[col] = data[col].fillna(data[col].mean())

X = data.drop(columns=['Potability'])
y = data['Potability']

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# -------------------------------
# Build Model
# -------------------------------
def build_model():
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_save_model():
    model = build_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)
    loss, acc = model.evaluate(X_test, y_test)
    print(f"✅ Model trained successfully. Accuracy: {acc:.3f}")
    model.save('water_quality_model.h5')


def load_or_train_model():
    if os.path.exists('water_quality_model.h5'):
        print("✅ Model loaded successfully.")
        return tf.keras.models.load_model('water_quality_model.h5')
    else:
        print("⚙️ Training new model...")
        train_and_save_model()
        return tf.keras.models.load_model('water_quality_model.h5')


model = load_or_train_model()
scaler = joblib.load('scaler.pkl')

# Adjusted threshold
BEST_THRESHOLD = 0.25


# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json(force=True)
        if 'features' not in input_data:
            return jsonify({"error": "Missing 'features' in request"}), 400

        features = input_data['features']

        # ✅ Convert all to float for reliable comparison
        features = {k: float(v) for k, v in features.items()}

        # WHO validation
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

        # Rule-based unsafe check first
        if unsafe_conditions:
            message = "⚠️ Water is not safe to drink (violates WHO standards)."
            return jsonify({"prediction": message, "potability": 0})

        # Model prediction
        input_df = pd.DataFrame([features], columns=[
            'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ])

        input_scaled = scaler.transform(input_df)
        prediction_prob = model.predict(input_scaled)[0][0]
        prediction_binary = int(prediction_prob > BEST_THRESHOLD)

        if prediction_binary == 1:
            message = "💧 Water is safe to drink!"
        else:
            message = "⚠️ Water is not safe to drink."

        return jsonify({
            "prediction": message,
            "potability": prediction_binary
        })

    except Exception as e:
        print("❌ Error in prediction:", e)
        return jsonify({"error": str(e)}), 400


# -------------------------------
# Run Server
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
