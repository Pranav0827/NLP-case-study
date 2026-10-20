# 🚦 Case Study 4: Pune Traffic Management Organization — Grievance Classification & Location NER System

An automated NLP prototype developed for a **Pune Traffic-Management Organization** (ATMS / Traffic Police Command Center) to process informal, free-text citizen complaints.

---

## 📌 Problem Statement

A Pune traffic-management organization receives a large volume of free-text complaints daily across digital channels (social media, mobile apps, citizen portals). Sorting them manually requires excessive personnel and yields inconsistent categorization. Most critically, the **location**—the vital detail required to dispatch field marshals or repair crews—is buried inside sentences.

Complaints are highly informal:
- **Spelling mistakes & typos** (e.g. `katraj ghaat`, `hinjewdi`, `shivajinagr`, `swargt`, `trafic`, `signl`)
- **Missing words & telegraphic shorthand**
- **All-lowercase text without capitalization**
- **Local place names & aliases** (e.g. `nal stop`, `fc rd`, `sb road`, `chandni chowk`)
- **Multiple problems or multiple locations in a single grievance** (e.g., `from swargate to shivajinagar junction`)

---

## 🎯 Objectives & Solution

1. **Classify into Six Core Categories:**
   - 🚗 `Traffic Congestion & Bottlenecks`
   - 🚥 `Traffic Signal Breakdown & Timing`
   - 🅿️ `Illegal & Obstructive Parking`
   - 🏍️ `Reckless & Wrong-Side Driving`
   - 🚧 `Road Hazards, Potholes & Waterlogging`
   - 🛒 `Encroachment & Footpath Obstruction`

2. **Extract Location Using Named Entity Recognition (NER):**
   - Pune Location Gazetteer covering 30+ canonical landmarks, chowks, flyovers, corridors, and 100+ aliases.
   - Contextual span detection (`near <LOC>`, `opp <LOC>`, `towards <LOC>`, `between <LOC1> and <LOC2>`).
   - Fuzzy Levenshtein matching to withstand misspellings.
   - Geocoding to coordinates `(lat, lon)` for instant pinpointing on an interactive Pune Map.

3. **Standard Reproducible Metrics Evaluation:**
   - Evaluated on a strictly held-out 20% stratified test split (126 records).
   - Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score (Macro/Weighted).
   - 6x6 Confusion Matrix and Per-Class Classification Report.
   - Entity-level NER Precision (96.1%), Recall (99.3%), F1-Score (97.7%), and Exact Match Rate (94.4%).
   - Subgroup robustness breakdown on Typos, All-Lowercase, and Multi-Location complaints.

---

## 🚀 How to Run the Prototype

### Option 1: Double-Click Launcher (Windows)
Double-click `run_cs4_traffic.bat` from the root folder or `cs4_traffic/run_cs4.bat`.

### Option 2: Terminal Run
```bash
python -m streamlit run cs4_traffic/app.py
```

### Option 3: Reproduce Metrics from CLI
```bash
python cs4_traffic/evaluate_models.py
```

To re-train the models from scratch:
```bash
python cs4_traffic/train_models.py
```

To re-synthesize the dataset:
```bash
python cs4_traffic/generate_traffic_dataset.py
```

---

## 📊 Summary Benchmark Table

| Classifier Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Linear SVM** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Random Forest** | 0.9921 | 0.9924 | 0.9921 | 0.9921 | 0.9921 |

### Location NER Metrics (Test Set):
- **Precision:** `96.10%`
- **Recall:** `99.33%`
- **F1-Score:** `97.69%`
- **Exact Match Rate:** `94.44%`
- **Robustness on Typos:** `96.69% F1`
- **Robustness on All-Lowercase:** `97.17% F1`
- **Multi-Location Extractions:** `100.00% F1`
