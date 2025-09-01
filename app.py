from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Suppress TensorFlow warnings except errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__, template_folder='templates')  # Ensure template folder is set correctly
CORS(app)

# Serve the frontend.html
@app.route('/')
def home():
    return render_template('frontend.html')

# Load dataset
data = pd.read_csv('water_potability.csv')

# Apply WHO guidelines to label the water as safe or unsafe
def apply_who_standards(data):
    # Safe drinking water criteria based on WHO standards:
    data['Potability'] = 0  # Default to unsafe (0)
    
    # Apply WHO Standards: Safe if all parameters are within the safe thresholds
    data.loc[
        (data['ph'] >= 6.5) & (data['ph'] <= 8.5) &
        (data['Hardness'] <= 500) &
        (data['Solids'] <= 500) & 
        (data['Chloramines'] <= 4) &
        (data['Sulfate'] <= 250) &
        (data['Conductivity'] <= 1000) &
        (data['Organic_carbon'] >= 2.2) & (data['Organic_carbon'] <= 15) &
        (data['Trihalomethanes'] >= 0.738) & (data['Trihalomethanes'] <= 100) &
        (data['Turbidity'] <= 5),
        'Potability'
    ] = 1  # Water is safe (label as 1)
    
    return data

# Apply WHO standards to label the data
data = apply_who_standards(data)

# Handle missing values
data['ph'] = data['ph'].fillna(data['ph'].mean())
data['Sulfate'] = data['Sulfate'].fillna(data['Sulfate'].mean())
data['Trihalomethanes'] = data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean())

# Define features and target
target_column = 'Potability'
X = data.drop(target_column, axis=1)
y = data[target_column]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and save model
def train_and_save_model():
    model = build_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy on Test Data: {accuracy}")
    model.save('water_quality_model.h5')
    print("Model saved successfully!")

# Load or train model
def load_or_train_model():
    if os.path.exists('water_quality_model.h5'):
        model = tf.keras.models.load_model('water_quality_model.h5')
        print("Model loaded successfully!")
    else:
        print("Model not found, training a new one...")
        train_and_save_model()
        model = tf.keras.models.load_model('water_quality_model.h5')
    return model

# Load model at start
model = load_or_train_model()

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json(force=True)

        if 'features' not in input_data:
            return jsonify({"error": "Missing 'features' in the request data."}), 400

        input_df = pd.DataFrame([input_data['features']], columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                                                                 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])

        if input_df.shape[1] != X_train.shape[1]:
            return jsonify({"error": f"Incorrect number of features. Expected {X_train.shape[1]} features."}), 400

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prediction_binary = (prediction > 0.5).astype(int)

        if prediction_binary[0] == 1:
            return jsonify({"prediction": "Water is safe to drink!"})
        else:
            return jsonify({"prediction": "Water is not safe to drink."})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

