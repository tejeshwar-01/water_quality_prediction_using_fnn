import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------
# STEP 1: Load and preprocess dataset
# ------------------------------
print("🔹 Loading dataset...")
data = pd.read_csv('water_potability.csv')

# Apply WHO labeling
def apply_who_standards(data):
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

data = apply_who_standards(data)

# Fill missing values
for col in ['ph', 'Sulfate', 'Trihalomethanes']:
    data[col] = data[col].fillna(data[col].mean())

# Split features/target
X = data.drop(columns=['Potability'])
y = data['Potability']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved (scaler.pkl)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------
# STEP 2: Handle imbalance with class weights
# ------------------------------
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))
print("✅ Class Weights:", class_weights)

# ------------------------------
# STEP 3: Build and train model
# ------------------------------
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("⚙️ Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ------------------------------
# STEP 4: Evaluate model and find best threshold
# ------------------------------
y_pred_prob = model.predict(X_test).ravel()

thresholds = np.arange(0.0, 1.0, 0.01)
f1_scores = [f1_score(y_test, (y_pred_prob > t).astype(int), zero_division=0) for t in thresholds]
best_t = thresholds[np.argmax(f1_scores)]

print(f"\n✅ Training complete.")
print(f"Best threshold for F1: {best_t:.2f}")

# Save model and threshold
model.save('water_quality_model.h5')
np.save('best_threshold.npy', best_t)
print("✅ Model saved (water_quality_model.h5)")
print("✅ Best threshold saved (best_threshold.npy)")
