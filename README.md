# 📍 Pune NLP Case Studies Repository

This repository hosts two independent NLP systems for Pune Municipal & Traffic Operations, kept strictly separated:

1. **🚦 [Case Study 4 (Branch `cs4`)]: Pune Traffic Management Organization — Grievance Classification & Location NER System**
   - **Directory:** [`cs4_traffic/`](file:///c:/Users/Perfect/OneDrive/Desktop/NLP-case-study/cs4_traffic)
   - **Topics Covered:** 6-Category Traffic Classification, Location Named Entity Recognition (NER), Informal Text Nuances (Typos, Lowercase, Missing Words, Multiple Locations, Multiple Problems), Reproducible Evaluation Metrics.
   - **Launch Command:** `run_cs4_traffic.bat` or `python -m streamlit run cs4_traffic/app.py`
   - **Detailed Docs:** [`cs4_traffic/README.md`](file:///c:/Users/Perfect/OneDrive/Desktop/NLP-case-study/cs4_traffic/README.md)

2. **🚌 [Case Study 1]: Pune Public Transport — Grievance Classification System**
   - **Directory:** Root repository (`app.py`, `pune_transport_complaints.csv`, `models/`)
   - **Topics Covered:** PMPML & Pune Metro Passenger Grievances, Sentiment Analysis, Priority Triage.
   - **Launch Command:** `run_dashboard.bat` or `python -m streamlit run app.py`

---

## 🚦 Case Study 4 Quick Reference (Pune Traffic Management)

### Problem Statement
A Pune traffic-management organization receives a large number of free-text complaints every day. Sorting them by hand needs many staff and gives inconsistent results. The location, which is the most important detail for action, is buried inside the sentence and must be found manually.
Complaints are informal: they contain spelling mistakes, missing words, all-lowercase text, local place names and sometimes more than one problem or more than one place. An automated system is therefore required to:
1. Classify each complaint into one of **six categories**
2. Extract the location mentioned using **NER**
3. Be evaluated with **standard, reproducible metrics**

### Quick Launch Options (Case Study 4)
- **Windows Batch Launcher:** Double-click [`run_cs4_traffic.bat`](file:///c:/Users/Perfect/OneDrive/Desktop/NLP-case-study/run_cs4_traffic.bat)
- **Streamlit App:** `python -m streamlit run cs4_traffic/app.py`
- **Reproducible Evaluation CLI:** `python cs4_traffic/evaluate_models.py`
- **Model Training Pipeline:** `python cs4_traffic/train_models.py`
- **Dataset Generator:** `python cs4_traffic/generate_traffic_dataset.py`

### 6 Target Categories
1. `Traffic Congestion & Bottlenecks`
2. `Traffic Signal Breakdown & Timing`
3. `Illegal & Obstructive Parking`
4. `Reckless & Wrong-Side Driving`
5. `Road Hazards, Potholes & Waterlogging`
6. `Encroachment & Footpath Obstruction`

### Reproducible Metrics Summary
- **Classifier Models Evaluated:** Multinomial Naive Bayes (F1: 1.0000), Logistic Regression (F1: 1.0000), Linear SVM (F1: 1.0000), Random Forest (F1: 0.9921)
- **Location NER Performance:** Precision: `0.9610` | Recall: `0.9933` | F1-Score: `0.9769` | Exact Match: `0.9444`
- **Informal Nuances Robustness:** Typos F1: `0.9669` | Lowercase F1: `0.9717` | Multi-Location F1: `1.0000`

---

## 🚌 Original Public Transport Grievance Classifier

To run the original transport grievance classification app:
```bash
python -m streamlit run app.py
```
*(Or double-click `run_dashboard.bat`)*
