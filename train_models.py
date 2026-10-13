"""
Model Training and Comparative Evaluation Pipeline for Pune Public Transport Complaints.
Evaluates 4 NLP Classifiers across:
  - Accuracy
  - Precision (Macro & Weighted)
  - Recall (Macro & Weighted)
  - F1-Score (Macro & Weighted)
  - Confusion Matrix & Classification Report
Trains on 3 targets: Service Category, Sentiment, Priority.
Saves models, vectorizer, evaluation metrics, and sample predictions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def run_training_pipeline(data_path="pune_transport_complaints.csv", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df)} records from {data_path}")

    # Shared TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2500,
        sublinear_tf=True,
        stop_words='english'
    )
    
    # Stratified split based on service_category
    X_train_raw, X_test_raw, y_train_df, y_test_df = train_test_split(
        df['complaint_text'],
        df[['service_category', 'sentiment', 'priority']],
        test_size=0.20,
        random_state=42,
        stratify=df['service_category']
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    print("TF-IDF Vectorizer trained and saved.")

    # Candidate Models to compare
    model_factories = {
        "Multinomial Naive Bayes": lambda: MultinomialNB(alpha=0.5),
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42),
        "Linear SVM": lambda: LinearSVC(dual='auto', C=1.0, class_weight='balanced', random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=120, max_depth=20, random_state=42)
    }

    targets = ['service_category', 'sentiment', 'priority']
    all_evaluations = {}
    best_models = {}

    for target in targets:
        print(f"\n==========================================")
        print(f"  EVALUATING MODELS FOR: {target.upper()}")
        print(f"==========================================")
        
        y_train = y_train_df[target]
        y_test = y_test_df[target]
        classes = sorted(list(df[target].unique()))
        
        all_evaluations[target] = {
            "classes": classes,
            "models": {}
        }

        best_f1 = -1
        best_model_name = None

        for model_name, factory in model_factories.items():
            clf = factory()
            clf.fit(X_train_tfidf, y_train)
            y_pred = clf.predict(X_test_tfidf)

            acc = float(accuracy_score(y_test, y_pred))
            prec_macro = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
            prec_weighted = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            rec_macro = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
            rec_weighted = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_mac = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
            f1_wt = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            cm = confusion_matrix(y_test, y_pred, labels=classes)
            report = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

            all_evaluations[target]["models"][model_name] = {
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

            # Save each trained model checkpoint
            model_slug = f"{target}_{model_name.lower().replace(' ', '_')}"
            joblib.dump(clf, os.path.join(output_dir, f"{model_slug}.joblib"))

            print(f"[{model_name}] -> Acc: {acc:.4f} | Prec(wt): {prec_weighted:.4f} | Rec(wt): {rec_weighted:.4f} | F1(wt): {f1_wt:.4f}")

            if f1_wt > best_f1:
                best_f1 = f1_wt
                best_model_name = model_name
                best_models[target] = clf

        all_evaluations[target]["best_model"] = best_model_name
        print(f"--> BEST MODEL for {target}: {best_model_name} (F1 Weighted: {best_f1:.4f})")

    # Save evaluation summary to JSON
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(all_evaluations, f, indent=2)
    print("\nAll evaluation metrics saved to models/evaluation_metrics.json")

    # Save test set for interactive validation and viewing
    test_summary = pd.DataFrame({
        "complaint_text": X_test_raw.values,
        "true_category": y_test_df['service_category'].values,
        "true_sentiment": y_test_df['sentiment'].values,
        "true_priority": y_test_df['priority'].values
    })
    test_summary.to_csv(os.path.join(output_dir, "test_split_records.csv"), index=False)
    print("Test split records saved.")

    return all_evaluations

if __name__ == "__main__":
    run_training_pipeline()
