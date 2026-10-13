"""
Complete Training and Model Comparison Pipeline for Pune Transit NLP.
Trains 4 classifiers across:
  - Service Category (6 classes)
  - Sentiment (Negative, Neutral, Positive)
  - Priority (High, Medium, Low)
Compares using: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
Saves model artifacts and evaluation metrics.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from nlp_engine import (
    PuneTfidfVectorizer,
    MultinomialNaiveBayesClassifier,
    SoftmaxLogisticRegression,
    LinearSVMClassifier,
    NearestCentroidClassifier,
    compute_metrics
)

def stratified_split(df, target_col='service_category', test_ratio=0.20, seed=42):
    rng = np.random.RandomState(seed)
    train_indices = []
    test_indices = []

    for label, group in df.groupby(target_col):
        indices = group.index.values.copy()
        rng.shuffle(indices)
        split_point = int(len(indices) * (1 - test_ratio))
        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return df.loc[train_indices].reset_index(drop=True), df.loc[test_indices].reset_index(drop=True)

def run_pipeline(data_path="pune_transport_complaints.csv", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")

    # Stratified split based on service category
    train_df, test_df = stratified_split(df, target_col='service_category', test_ratio=0.20, seed=42)
    print(f"Train set: {len(train_df)} | Test set: {len(test_df)}")

    # Vectorizer
    vectorizer = PuneTfidfVectorizer(max_features=1800, min_df=1, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['complaint_text'].values)
    X_test = vectorizer.transform(test_df['complaint_text'].values)

    with open(os.path.join(output_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} n-grams")

    models_config = {
        "Multinomial Naive Bayes": lambda: MultinomialNaiveBayesClassifier(alpha=0.5),
        "Logistic Regression": lambda: SoftmaxLogisticRegression(lr=0.15, epochs=350, l2_reg=0.005),
        "Linear SVM": lambda: LinearSVMClassifier(C=1.2, epochs=350, lr=0.08),
        "Nearest Centroid": lambda: NearestCentroidClassifier()
    }

    targets = ['service_category', 'sentiment', 'priority']
    evaluation_results = {}
    best_models = {}

    for target in targets:
        print(f"\n{'='*55}")
        print(f"  TARGET: {target.upper()}")
        print(f"{'='*55}")

        classes = sorted(list(df[target].unique()))
        y_train = train_df[target].values
        y_test = test_df[target].values

        target_eval = {
            "classes": classes,
            "models": {}
        }

        best_f1 = -1
        best_name = None

        for model_name, model_fn in models_config.items():
            clf = model_fn()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            metrics = compute_metrics(y_test, y_pred, classes)
            target_eval["models"][model_name] = metrics

            # Save model checkpoint
            safe_target = target.replace(" ", "_")
            safe_name = model_name.lower().replace(" ", "_")
            with open(os.path.join(output_dir, f"{safe_target}_{safe_name}.pkl"), "wb") as f:
                pickle.dump(clf, f)

            print(f"{model_name:<26} | Acc: {metrics['accuracy']:.4f} | Prec(wt): {metrics['precision_weighted']:.4f} | Rec(wt): {metrics['recall_weighted']:.4f} | F1(wt): {metrics['f1_weighted']:.4f}")

            if metrics['f1_weighted'] > best_f1:
                best_f1 = metrics['f1_weighted']
                best_name = model_name
                best_models[target] = clf

        target_eval["best_model"] = best_name
        evaluation_results[target] = target_eval
        print(f"--> BEST PERFORMER for {target}: {best_name} (F1 Weighted: {best_f1:.4f})")

    # Save evaluation summary
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print("\nSaved evaluation metrics to models/evaluation_metrics.json")

    # Save test set with true values for live test set inspection
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    print("Saved test set to models/test_split.csv")

    return evaluation_results

if __name__ == "__main__":
    run_pipeline()
