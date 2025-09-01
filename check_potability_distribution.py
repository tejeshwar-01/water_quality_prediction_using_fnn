
import pandas as pd
import os

# Suppress TensorFlow warnings (not strictly needed for this script, but good practice)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define the apply_who_standards function as currently in app.py
def apply_who_standards(data):
    data['Potability'] = 0  # Default to unsafe (0)
    
    # Apply WHO Standards: Safe if all parameters are within the safe thresholds
    data.loc[
        (data['ph'] >= 6.0) & (data['ph'] <= 9.0) &
        (data['Hardness'] <= 300),
        'Potability'
    ] = 1  # Water is safe (label as 1)
    
    return data

# Load dataset
data = pd.read_csv('water_potability.csv')

# Handle missing values (as in app.py)
data['ph'] = data['ph'].fillna(data['ph'].mean())
data['Sulfate'] = data['Sulfate'].fillna(data['Sulfate'].mean())
data['Trihalomethanes'] = data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean())

# Apply WHO standards to label the data
data = apply_who_standards(data)

# Print the value counts of the 'Potability' column
print("Potability Label Distribution after applying WHO Standards:")
print(data['Potability'].value_counts())
