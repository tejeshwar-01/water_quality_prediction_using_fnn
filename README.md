🌊 AI-Powered Water Quality Predictor
<p align="center"> <img src="https://i.imgur.com/0eC9cVu.png" width="80%" alt="Water Quality Prediction Banner"/> </p> <p align="center"> <b>💧 Predict whether water is safe to drink using AI + Flask + WHO standards 💧</b><br> Built by <a href="https://github.com/tejeshwar-01">Tejeshwar Reddy</a> </p>
🚀 Overview

An AI-based web application that predicts whether a given water sample is safe or unsafe to drink using a Feedforward Neural Network (FNN).
It integrates Flask, TensorFlow, and WHO guideline validation to ensure accuracy and reliability.

🧠 Trained using real-world Water Potability Dataset
🎨 Includes a beautiful responsive UI
⚡ Built for speed, precision, and presentation

🧠 Features

✅ Feedforward Neural Network (Keras/TensorFlow)

✅ WHO safety standard validation

✅ Flask backend for real-time predictions

✅ Modern colorful responsive UI

✅ Clear “Safe / Unsafe” result display

✅ Ready for deployment on Render / Vercel / Heroku

🧰 Tech Stack
Layer	Technology
Frontend	HTML5, CSS3 (with gradients & animations), JavaScript
Backend	Flask (Python)
Machine Learning	TensorFlow / Keras
Data Preprocessing	Pandas, NumPy, Scikit-learn
Model Persistence	.h5, .pkl, .npy
Deployment	Render / Localhost
📂 Project Structure
water_quality_prediction_using_fnn/
│
├── app.py                        # Flask backend
├── train_model_balanced.py       # Model training & threshold tuning
├── evaluate_model.py             # Evaluation metrics
├── check_potability_distribution.py
│
├── templates/
│   └── frontend.html             # Responsive modern UI
│
├── static/
│   └── app.js                    # Optional JavaScript
│
├── water_potability.csv          # Dataset
├── water_quality_model.h5        # Saved trained model
├── scaler.pkl                    # MinMaxScaler for normalization
├── best_threshold.npy            # Optimized decision threshold
│
└── README.md                     # Project documentation

🧪 Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/tejeshwar-01/water_quality_prediction_using_fnn.git
cd water_quality_prediction_using_fnn

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # macOS / Linux

3️⃣ Install Required Packages
pip install -r requirements.txt

4️⃣ Run the Flask App
python app.py


Then open your browser at 👉 http://127.0.0.1:5000

💡 WHO Safety Standards
Parameter	Safe Range	Unit
pH	6.5 – 8.5	-
Hardness	≤ 500	mg/L
Solids	≤ 50,000	mg/L
Chloramines	≤ 4	mg/L
Sulfate	≤ 400	mg/L
Conductivity	≤ 2000	µS/cm
Organic Carbon	2.2 – 15	mg/L
Trihalomethanes	≤ 100	µg/L
Turbidity	≤ 5	NTU

⚠️ If any value exceeds these limits → the water is not safe.

🧬 AI Model Details
Layer	Neurons	Activation
Dense (Input)	64	ReLU
Dense (Hidden)	32	ReLU
Dense (Output)	1	Sigmoid

Optimizer: Adam
Loss: Binary Crossentropy
Metrics: Accuracy, F1-Score, Precision, Recall
Early Stopping: Enabled

💻 Example Predictions
Sample Input	Result
pH=7.3, Hardness=220, Solids=15000, Chloramines=2.5, Sulfate=250, Conductivity=1500, Organic Carbon=10, Trihalomethanes=60, Turbidity=3	💧 Water is Safe
pH=5.5, Hardness=700, Solids=60000, Chloramines=5, Sulfate=500, Conductivity=2500, Organic Carbon=20, Trihalomethanes=120, Turbidity=8	⚠️ Water is Not Safe to Drink
🧾 License

Licensed under the MIT License — free to use, modify, and distribute with attribution.

👤 Author

Tejeshwar Reddy
📍 AI & ML Enthusiast | Python Developer
🔗 GitHub Profile

⭐ Support

If you like this project, please ⭐ the repo —
it motivates continued improvements and inspires innovation 💙

<p align="center"> <b>💧 Empowering clean water through Artificial Intelligence 💧</b> </p>
