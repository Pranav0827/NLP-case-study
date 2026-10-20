"""
Case Study 4: Model Training & Evaluation Pipeline for Pune Traffic Complaints.
Evaluates 4 NLP Classifiers across 6 Traffic Management Categories:
  - Accuracy, Precision (Macro & Weighted), Recall (Macro & Weighted), F1-Score (Macro & Weighted)
  - 6x6 Confusion Matrix & Per-Class Classification Report
Evaluates Location Named Entity Recognition (NER):
  - Precision, Recall, F1-Score, Exact Match Rate, Informal Text Robustness
Saves models, metrics, and evaluation summaries to cs4_traffic/models/.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from nlp_engine import CATEGORIES, PuneLocationNER, evaluate_ner_performance

def run_training_pipeline(
    data_path=None,
    output_dir=None,
    test_size=0.20,
    random_state=42
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if data_path is None:
        data_path = os.path.join(base_dir, "pune_traffic_complaints.csv")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "models")
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")

    # 1. Stratified Train / Test Split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['category']
    )
    print(f"Split: Train = {len(train_df)} samples, Test = {len(test_df)} samples")

    # 2. Fit TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2500,
        sublinear_tf=True,
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(train_df['complaint_text'])
    X_test_tfidf = vectorizer.transform(test_df['complaint_text'])
    
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    print("TF-IDF Vectorizer fitted and saved.")

    # 3. Model Definitions
    model_factories = {
        "Multinomial Naive Bayes": lambda: MultinomialNB(alpha=0.3),
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, C=2.0, class_weight='balanced', random_state=random_state
        ),
        "Linear SVM": lambda: CalibratedClassifierCV(
            LinearSVC(dual='auto', C=1.5, class_weight='balanced', random_state=random_state),
            method='sigmoid'
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=150, max_depth=25, random_state=random_state
        )
    }

    y_train = train_df['category']
    y_test = test_df['category']
    classes = CATEGORIES

    all_evaluations = {
        "categories": classes,
        "classification_models": {},
        "best_model": None,
        "dataset_statistics": {
            "total_records": len(df),
            "train_records": len(train_df),
            "test_records": len(test_df),
            "typo_percentage": round(df['has_typos'].mean() * 100, 2),
            "all_lowercase_percentage": round(df['is_all_lowercase'].mean() * 100, 2),
            "multi_location_percentage": round(df['is_multi_location'].mean() * 100, 2),
            "multi_problem_percentage": round(df['is_multi_problem'].mean() * 100, 2)
        }
    }

    best_f1 = -1.0
    best_model_name = None
    fitted_models = {}

    print("\n" + "="*50)
    print("  TRAINING & EVALUATING 6-CATEGORY CLASSIFIERS")
    print("="*50)

    for model_name, factory in model_factories.items():
        clf = factory()
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        fitted_models[model_name] = clf

        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
        prec_weighted = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
        rec_weighted = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_mac = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        f1_wt = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        report = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

        all_evaluations["classification_models"][model_name] = {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_macro": round(rec_macro, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_macro": round(f1_mac, 4),
            "f1_weighted": round(f1_wt, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }

        # Save model artifact
        slug = model_name.lower().replace(" ", "_")
        joblib.dump(clf, os.path.join(output_dir, f"category_{slug}.joblib"))

        print(f"[{model_name:25s}] Acc: {acc:.4f} | Prec(wt): {prec_weighted:.4f} | Rec(wt): {rec_weighted:.4f} | F1(wt): {f1_wt:.4f}")

        if f1_wt > best_f1:
            best_f1 = f1_wt
            best_model_name = model_name

    all_evaluations["best_model"] = best_model_name
    print(f"\n--> BEST CLASSIFIER: {best_model_name} (F1 Weighted = {best_f1:.4f})")

    # 4. Evaluate Location NER on the Test Split
    print("\n" + "="*50)
    print("  EVALUATING LOCATION NER ON INFORMAL TEST SET")
    print("="*50)
    ner_model = PuneLocationNER()
    ner_metrics = evaluate_ner_performance(test_df, ner_model=ner_model)
    all_evaluations["location_ner_metrics"] = ner_metrics
    
    print(f"Location NER Precision: {ner_metrics['precision']:.4f}")
    print(f"Location NER Recall:    {ner_metrics['recall']:.4f}")
    print(f"Location NER F1-Score:  {ner_metrics['f1_score']:.4f}")
    print(f"Exact Match Rate:       {ner_metrics['exact_match_rate']:.4f}")
    print(f"Subgroup Robustness on Typos:     F1 = {ner_metrics['subgroup_robustness']['has_typos']['f1']:.4f}")
    print(f"Subgroup Robustness on Lowercase: F1 = {ner_metrics['subgroup_robustness']['is_all_lowercase']['f1']:.4f}")

    # 5. Save Test Split Predictions for Live Verification
    best_clf = fitted_models[best_model_name]
    y_test_pred = best_clf.predict(X_test_tfidf)
    
    test_summary = test_df.copy()
    test_summary["predicted_category"] = y_test_pred
    test_summary["correct_classification"] = (test_summary["category"] == test_summary["predicted_category"])
    
    # Add predicted locations
    pred_locations_list = []
    for txt in test_summary["complaint_text"]:
        preds = [e["canonical"] for e in ner_model.extract_locations(txt)]
        pred_locations_list.append(json.dumps(preds))
    test_summary["predicted_locations"] = pred_locations_list
    
    test_summary.to_csv(os.path.join(output_dir, "test_split_records.csv"), index=False)
    print("\nTest split records with predictions saved.")

    # 6. Save Evaluation Metrics JSON
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(all_evaluations, f, indent=2)
    print("Saved evaluation metrics to cs4_traffic/models/evaluation_metrics.json")

    return all_evaluations

if __name__ == "__main__":
    run_training_pipeline()
