# ⚡ Power Tower Fault Risk Prediction & Maintenance Routing System

This project simulates a **power transmission infrastructure monitoring system** that predicts **failure risk for electricity towers** using **machine learning** and optimizes **field inspection routing** using **Dijkstra’s algorithm**.

The system includes:

✅ **Synthetic dataset generation** with real-world inspired tower metadata and daily weather conditions  
✅ **Logistic Regression fault prediction model** with ROC/AUC evaluation  
✅ **Streamlit web application** to simulate daily sensor readings and identify **high-risk towers**  
✅ **Graph-based routing (Dijkstra)** to compute the **optimal maintenance visit path**

---

## 🚀 Features

| Component | Description |
|-----------|------------|
| 🏗 Dataset Generator | Creates 60 days of data for 25 towers with weather, aging, corrosion, maintenance history |
| 🤖 Machine Learning Model | Logistic Regression classifier with hyperparameter tuning & feature importance |
| 📊 Model Evaluation | ROC Curve, Confusion Matrix, Accuracy, Precision, Recall, F1-Score |
| 🌦 Dynamic Risk Prediction | Accepts live (or simulated) daily sensor readings to re-evaluate risk |
| 🛣 Optimized Routing | Uses Dijkstra’s Algorithm to find least-cost path to inspect risky towers |

---

## 📁 Project Structure

📁 your-repo/
│── generate_dataset_and_model.py # Data generation + model training
│── streamlit_app.py # Interactive prediction & routing UI
│── requirements.txt # Dependencies
│── README.md # This file
│
├── (Auto-generated after running script)
│ ├── tower_data.csv
│ ├── towers_metadata.csv
│ ├── graph_edges.csv
│ ├── fault_prediction_model.pkl
│ ├── model_features.pkl
│ ├── roc_curve.png
│ ├── confusion_matrix.png

yaml
Copy code

---

## 🛠 Installation & Setup

1️⃣ **Clone Repository**

```bash
git clone https://github.com/yourname/your-repo.git
cd your-repo
2️⃣ Install Dependencies

bash
Copy code
pip install -r requirements.txt
3️⃣ Generate Dataset & Train Model

bash
Copy code
python generate_dataset_and_model.py
4️⃣ Launch Web App

bash
Copy code
streamlit run streamlit_app.py
🧠 Model Performance (Sample Output)
Metric	Value
Accuracy	0.87
Precision	0.82
Recall	0.79
F1-Score	0.80
ROC-AUC	0.91

Visual ROC Curve and Confusion Matrix are saved in project root.

🌍 Future Improvements
Deploy to Streamlit Cloud / Azure / AWS

Upgrade to XGBoost / Random Forest

Integrate real IoT sensor inputs instead of random

Use A* Algorithm or TSP variant for multi-path optimization

🤝 Contributing
Pull requests are welcome! Feel free to:

Add realistic substation/tower geolocation

Plug in real maintenance crew travel constraints

Suggest UI improvements

📜 License
This project is licensed under the MIT License — free to use and modify.

⚡ Built for Smart Grid Automation & Predictive Maintenance 🚀
Let me know if you want a logo banner, GIF demo recording, or deployment badge — I can help add that too!

yaml
Copy code

---

Would you like me to also:

✅ **Generate shields/badges** (Build Passing, Python, MIT License, etc.)  
✅ **Include a GIF of the Streamlit UI?** (I can give you the code to record with `streamlit-webrtc` or ffmpeg)  
✅ **Auto-generate changelog?**

Just say the word!






