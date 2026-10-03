# 🛡️ MachineGuard

## AI-Powered Predictive Maintenance & Machine Failure Detection

MachineGuard is a Machine Learning-based predictive maintenance application that analyzes machine operating parameters and estimates the probability of machine failure.

The system uses a **Random Forest Classifier** trained on machine operational data and provides an interactive **Streamlit web application** through which users can enter machine parameters and obtain a real-time failure prediction, probability estimate, and risk assessment.

---

## 📌 Overview

Unexpected machine failures can result in production downtime, maintenance costs, and operational losses.

MachineGuard demonstrates how Machine Learning can be used to support predictive maintenance by analyzing parameters such as:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

Based on these parameters, the trained model predicts whether the machine is likely to experience a failure and calculates the estimated probability of failure.

The Streamlit dashboard then converts the model output into an easy-to-understand risk assessment.

---

## ✨ Key Features

- 🤖 Machine Failure Prediction using Machine Learning
- 🌲 Random Forest Classification
- 📊 Failure and Normal Probability
- ⚠️ Automated Risk Assessment
- 🟢 Low Risk Detection
- 🟡 Moderate Risk Detection
- 🔴 High Risk Detection
- 🖥️ Interactive Streamlit Dashboard
- 📈 Real-time prediction from user inputs
- 🔄 Pre-trained ML model loading
- 🧩 One-Hot Encoding for categorical features
- ⚙️ End-to-end ML pipeline using Scikit-learn
- 📱 User-friendly machine monitoring interface

---

# 🏗️ System Architecture

```text
                         USER
                           │
                           ▼
              ┌────────────────────────┐
              │   Streamlit Dashboard  │
              │                        │
              │ Machine Parameters     │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Data Preprocessing   │
              │                        │
              │ One-Hot Encoding       │
              │ Feature Transformation │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Random Forest Model  │
              │                        │
              │     300 Estimators     │
              └────────────┬───────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
       ┌────────────────┐    ┌────────────────┐
       │ Failure        │    │ Normal         │
       │ Probability    │    │ Probability    │
       └────────┬───────┘    └────────┬───────┘
                │                     │
                └──────────┬──────────┘
                           ▼
              ┌────────────────────────┐
              │    Risk Assessment     │
              │                        │
              │ Low / Moderate / High  │
              └────────────┬───────────┘
                           │
                           ▼
                    FINAL RESULT
```

---

# 🧠 Machine Learning Model

MachineGuard uses a **Random Forest Classifier** for machine failure prediction.

The model is implemented using a Scikit-learn pipeline consisting of:

1. Feature preprocessing
2. One-Hot Encoding of the machine `Type`
3. Random Forest Classification
4. Probability estimation

### Model Configuration

| Parameter | Configuration |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of Trees | 300 |
| Max Features | `sqrt` |
| Minimum Samples per Leaf | 2 |
| Class Weight | `balanced` |
| Random State | 42 |
| Parallel Processing | `n_jobs=-1` |
| Prediction Type | Binary Classification |

---

# 🎯 Prediction Classes

The model performs binary classification:

| Class | Meaning |
|---:|---|
| `0` | Normal Operation |
| `1` | Machine Failure |

The application additionally displays the probability associated with each class.

For example:

```text
Normal Probability  : 49.71%
Failure Probability : 50.29%
```

Because this is a binary classification problem:

```text
Normal Probability + Failure Probability ≈ 100%
```

---

# ⚠️ Risk Assessment

MachineGuard applies application-defined thresholds to the model's predicted failure probability.

| Failure Probability | Risk Level |
|---:|---|
| `< 40%` | 🟢 LOW RISK |
| `40% – 69.99%` | 🟡 MODERATE RISK |
| `≥ 70%` | 🔴 HIGH RISK |

### Important

The Random Forest model predicts the machine failure class and its estimated probabilities.

The labels **Low Risk, Moderate Risk, and High Risk** are application-level categories created using the probability thresholds above.

---

# 📥 Input Parameters

The application accepts the following machine parameters:

| Parameter | Description | Unit |
|---|---|---|
| Type | Machine quality/type category | L / M / H |
| Air Temperature | Ambient air temperature | Kelvin (K) |
| Process Temperature | Machine process temperature | Kelvin (K) |
| Rotational Speed | Machine rotational speed | RPM |
| Torque | Applied machine torque | Nm |
| Tool Wear | Tool usage duration | Minutes |

---

# 📊 Prediction Workflow

The application follows the following workflow:

```text
User enters machine parameters
            ↓
Input validation
            ↓
Feature preprocessing
            ↓
One-Hot Encoding
            ↓
Random Forest prediction
            ↓
Probability estimation
            ↓
Risk classification
            ↓
Dashboard displays result
```

---

# 🗂️ Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains machine operating parameters and failure information used to train the predictive model.

### Main Features

```text
Type
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
```

### Target

```text
Machine failure
```

---

# 🛠️ Technology Stack

### Programming Language

- Python 3

### Machine Learning

- Scikit-learn
- Random Forest Classifier
- One-Hot Encoding

### Data Processing

- Pandas
- NumPy

### Model Persistence

- Joblib

### Web Application

- Streamlit

### Development Tools

- Git
- GitHub
- VS Code

---

# 📁 Project Structure

```text
MachineGuard/
│
├── app.py
│
├── train_model.py
│
├── machine-dataset.csv
│
├── machine_failure_model.pkl
│
├── requirements.txt
│
├── .gitignore
│
└── assets/
    └── maintenance-bg.png
```

### File Description

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `train_model.py` | Model training script |
| `machine-dataset.csv` | Training dataset |
| `machine_failure_model.pkl` | Saved trained ML model |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignored files |
| `assets/maintenance-bg.png` | Dashboard background image |

---

# ⚙️ Installation & Setup

Follow the steps below to run MachineGuard locally.

## 1. Clone the Repository

Open Command Prompt / PowerShell / Terminal:

```bash
git clone https://github.com/Pushkarcodes166/MachineGuard.git
```

---

## 2. Open the Project Directory

```bash
cd MachineGuard
```

---

## 3. Create a Virtual Environment

Creating a virtual environment is recommended to keep project dependencies isolated.

### Windows

```bash
python -m venv venv
```

### macOS / Linux

```bash
python3 -m venv venv
```

---

## 4. Activate the Virtual Environment

### Windows Command Prompt

```bash
venv\Scripts\activate
```

### Windows PowerShell

```powershell
venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source venv/bin/activate
```

After activation, your terminal should show something similar to:

```text
(venv) C:\...\MachineGuard>
```

---

# 📦 Install Dependencies

Make sure the virtual environment is activated.

Then run:

```bash
pip install -r requirements.txt
```

This installs the required Python libraries, including:

```text
pandas
numpy
scikit-learn
joblib
streamlit
```

---

# 🚀 Running the Application

## Option 1 — Run the Existing Trained Model

If `machine_failure_model.pkl` is already present in the project directory, you **do not need to train the model again**.

Simply run:

```bash
streamlit run app.py
```

You should see output similar to:

```text
Local URL: http://localhost:8501
Network URL: http://192.xxx.xxx.xxx:8501
```

Open the following URL in your browser:

```text
http://localhost:8501
```

The MachineGuard dashboard will then open.

---

# 🧠 Training the Model

If you want to retrain the model using the dataset, run:

```bash
python train_model.py
```

The training script will:

```text
Load Dataset
     ↓
Separate Features and Target
     ↓
Preprocess Categorical Features
     ↓
Train Random Forest
     ↓
Evaluate Model
     ↓
Save Trained Model
```

After successful training, the model is saved as:

```text
machine_failure_model.pkl
```

Then start the application:

```bash
streamlit run app.py
```

---

# 🔄 Complete Setup From Scratch

If you are setting up the project on a new computer:

```bash
git clone https://github.com/Pushkarcodes166/MachineGuard.git

cd MachineGuard

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

### If the model file does not exist

Run:

```bash
python train_model.py
```

Then:

```bash
streamlit run app.py
```

---

# 🧪 Example Test Cases

You can use the following values to test the dashboard.

## 🟢 Low-Risk Example

```text
Type                : L
Air Temperature     : 298.10 K
Process Temperature : 308.60 K
Rotational Speed    : 1500 rpm
Torque              : 40 Nm
Tool Wear           : 20 min
```

Expected result:

```text
Low Risk
```

---

## 🟡 Moderate-Risk Example

```text
Type                : L
Air Temperature     : 300.00 K
Process Temperature : 310.00 K
Rotational Speed    : 1500 rpm
Torque              : 70 Nm
Tool Wear           : 200 min
```

Expected result:

```text
Failure Probability ≈ 50%
Risk Assessment     : MODERATE RISK
```

---

## 🔴 High-Risk Example

```text
Type                : L
Air Temperature     : 300.00 K
Process Temperature : 310.00 K
Rotational Speed    : 2500 rpm
Torque              : 10 Nm
Tool Wear           : 150 min
```

Expected result:

```text
High Risk
```

> The exact prediction and probability may vary depending on the trained model version.

---

# 🔍 Understanding the Prediction

The application uses:

```python
model.predict()
```

to obtain the predicted class.

For probability estimation it uses:

```python
model.predict_proba()
```

For a binary classifier:

```text
Class 0 → Normal
Class 1 → Failure
```

Therefore:

```python
probabilities = model.predict_proba(input_data)[0]

normal_probability = probabilities[0]
failure_probability = probabilities[1]
```

The failure probability is then converted into the application's risk category.

---

# 💡 Why Predictive Maintenance?

Traditional maintenance approaches often include:

### Reactive Maintenance

Machine is repaired after failure.

```text
Machine Failure
      ↓
Production Stop
      ↓
Inspection
      ↓
Repair
```

### Predictive Maintenance

Machine data is analyzed before failure.

```text
Machine Data
      ↓
Machine Learning
      ↓
Failure Probability
      ↓
Risk Assessment
      ↓
Maintenance Decision
```

MachineGuard demonstrates the second approach using machine learning.

---

# 🔐 Git & Version Control

The project uses Git for version control.

To create a new commit:

```bash
git add .
git commit -m "Update MachineGuard application"
git push
```

To check the current status:

```bash
git status
```

To view commit history:

```bash
git log --oneline
```

---

# 🧹 Recommended .gitignore

The project includes a `.gitignore` file to prevent unnecessary files from being committed.

Example:

```gitignore
# Python
__pycache__/
*.py[cod]

# Virtual Environment
venv/
.venv/
env/

# IDE
.vscode/

# Python Cache
.pytest_cache/

# Streamlit Secrets
.streamlit/secrets.toml

# OS
.DS_Store
Thumbs.db
```

---

# 🐛 Troubleshooting

## `python` is not recognized

Try:

```bash
python --version
```

or:

```bash
py --version
```

If `py` works, use:

```bash
py -m venv venv
```

---

## `pip` is not recognized

Use:

```bash
python -m pip install -r requirements.txt
```

---

## Streamlit command not found

Use:

```bash
python -m streamlit run app.py
```

---

## Model file not found

If you see an error related to:

```text
machine_failure_model.pkl
```

make sure the model exists in the project root.

If it does not exist, train it:

```bash
python train_model.py
```

---

## Port 8501 is already in use

Run Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

---

# 🔮 Future Enhancements

Potential future improvements include:

- 📡 Real-time IoT sensor integration
- 🏭 Live industrial machine monitoring
- 📊 Historical machine performance tracking
- 🔔 Automated maintenance alerts
- 🗄️ Database integration
- ☁️ Cloud deployment
- 📈 Advanced predictive analytics
- 🔧 Remaining Useful Life (RUL) prediction
- 🤖 Comparison of multiple ML algorithms
- 📱 Mobile-friendly monitoring interface
- 📧 Automated maintenance notifications

---

# 🎯 Project Objective

The objective of MachineGuard is to demonstrate an end-to-end Machine Learning workflow for predictive maintenance:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Streamlit Deployment
   ↓
Machine Failure Prediction
```

The project combines **Machine Learning, Python, Data Processing, and Web Application Development** into a practical predictive maintenance system.

---

# 👨‍💻 Author

## Pushkar Potnis

Computer Engineering Student

GitHub:

https://github.com/Pushkarcodes166

---

# 📄 License

This project is developed for educational and academic purposes.

---

## ⭐ Acknowledgement

The project is based on the **AI4I 2020 Predictive Maintenance Dataset** and uses open-source Python and Machine Learning libraries.
