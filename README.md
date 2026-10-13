# 🚌 Pune Public Transport — NLP Grievance Classification System

An NLP system developed for the **Pune Municipal Transport Organization (PMPML & Pune Metro)** to automatically classify passenger complaints, predict sentiment, and triage priority.

---

## 📌 Features

1. **✍️ Live Complaint Classifier:** Predicts Service Category (6 classes), Sentiment (Negative/Neutral/Positive), and Priority (High/Medium/Low).
2. **📊 Model Evaluation & Benchmark:** Compares **4 NLP Models** (Logistic Regression, Nearest Centroid, Linear SVM, and Naive Bayes) using **Accuracy, Precision, Recall, F1-Score**, and interactive **Confusion Matrix Heatmaps**.
3. **📈 Dataset & Analytics:** Overview of grievance categories, priority distributions, and a searchable dataset viewer.

---

## 🚀 How to Deploy on Streamlit Community Cloud (Free 24/7 Access)

You can deploy this app online so you can open it anytime from your mobile phone or laptop.

### Step 1: Create a GitHub Repository
1. Go to [github.com](https://github.com) and create a new repository (e.g., `pune-transport-nlp`).
2. Upload all files from this directory:
   - `app.py`
   - `nlp_engine.py`
   - `requirements.txt`
   - `pune_transport_complaints.csv`
   - `models/` directory (containing `.pkl` and `evaluation_metrics.json`)

### Step 2: Deploy on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `pune-transport-nlp`.
4. Main file path: `app.py`.
5. Click **"Deploy!"**.

### Result
Streamlit will generate a permanent link (e.g. `https://pune-transport-nlp.streamlit.app`). You can open and bookmark this link directly on your mobile phone to access the dashboard 24/7 without needing your laptop running!

---

## 💻 Local Run
To run locally on your laptop:
```bash
python -m streamlit run app.py
```
*(Or double-click `run_dashboard.bat`)*
