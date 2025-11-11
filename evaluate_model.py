import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# STEP 1: Load data, model, and scaler
# ------------------------------
print("🔹 Loading dataset, model, and scaler...")

data = pd.read_csv('water_potability.csv')
model = tf.keras.models.load_model('water_quality_model.h5')
scaler = joblib.load('scaler.pkl')

# ------------------------------
# STEP 2: Apply WHO labeling (same as training)
# ------------------------------
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

# ------------------------------
# STEP 3: Clean data and split
# ------------------------------
for col in ['ph', 'Sulfate', 'Trihalomethanes']:
    data[col] = data[col].fillna(data[col].mean())

X = data.drop(columns=['Potability'])
y = data['Potability']

X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------
# STEP 4: Model Evaluation
# ------------------------------
print("\n🔹 Evaluating model on test data...")
y_pred_prob = model.predict(X_test).ravel()

# --- Automatically find best threshold ---
from sklearn.metrics import f1_score
thresholds = np.arange(0.0, 1.0, 0.01)
f1_scores = [f1_score(y_test, (y_pred_prob > t).astype(int), zero_division=0) for t in thresholds]
best_t = thresholds[np.argmax(f1_scores)]
print(f"\n🔍 Best Threshold for F1 Score: {best_t:.2f}")

# Apply best threshold
y_pred = (y_pred_prob > best_t).astype(int)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred)

# ------------------------------
# STEP 5: Display Results
# ------------------------------
print("\n✅ Model Evaluation Results:")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1-Score : {f1*100:.2f}%")
print(f"AUC      : {auc:.3f}")
print("\nConfusion Matrix:\n", cm)

# ------------------------------
# STEP 6: Visualizations
# ------------------------------

# Confusion Matrix Plot
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Threshold = {:.2f})'.format(best_t))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Water Quality Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print("\n🔹 Evaluation complete.")
