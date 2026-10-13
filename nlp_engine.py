"""
Pure Python & NumPy NLP Engine for Pune Public Transport Complaints Classification.
Optimized implementation with:
  - TF-IDF Vectorizer with Sublinear Scaling & Pune stopword handling
  - Model 1: Multinomial Naive Bayes (MNB with tuned Laplace smoothing)
  - Model 2: Class-Weighted Softmax Logistic Regression (Gradient descent with Adam/Momentum)
  - Model 3: Linear Support Vector Machine (Pegasos margin solver)
  - Model 4: Nearest Centroid Classifier (Cosine Similarity)
  - Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report
"""

import re
import math
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter

STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 
    'aren', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 
    'but', 'by', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 
    'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 
    'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 
    'me', 'more', 'most', 'my', 'myself', 'of', 'off', 'on', 'once', 
    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 
    'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 
    'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 
    'up', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 
    'why', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
}

def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[a-z0-9]+\b', text)
    filtered = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return filtered

class PuneTfidfVectorizer:
    def __init__(self, max_features=2500, min_df=1, ngram_range=(1, 2)):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vocabulary_ = {}
        self.feature_names_ = []
        self.idf_ = None

    def _extract_ngrams(self, tokens):
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i+n])
                ngrams.append(ngram)
        return ngrams

    def fit(self, raw_documents):
        doc_count = len(raw_documents)
        df_counter = Counter()

        for doc in raw_documents:
            tokens = tokenize(doc)
            ngrams = set(self._extract_ngrams(tokens))
            for ng in ngrams:
                df_counter[ng] += 1

        filtered_items = [
            (term, count) for term, count in df_counter.items()
            if count >= self.min_df
        ]
        filtered_items.sort(key=lambda x: x[1], reverse=True)
        if self.max_features and len(filtered_items) > self.max_features:
            filtered_items = filtered_items[:self.max_features]

        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(filtered_items)}
        self.feature_names_ = [term for term, _ in filtered_items]

        idf = np.zeros(len(self.feature_names_))
        for idx, term in enumerate(self.feature_names_):
            df = df_counter[term]
            idf[idx] = math.log((1 + doc_count) / (1 + df)) + 1.0
        self.idf_ = idf
        return self

    def transform(self, raw_documents):
        n_docs = len(raw_documents)
        n_features = len(self.feature_names_)
        X = np.zeros((n_docs, n_features), dtype=np.float32)

        for i, doc in enumerate(raw_documents):
            tokens = tokenize(doc)
            ngrams = self._extract_ngrams(tokens)
            term_counts = Counter(ngrams)

            for term, count in term_counts.items():
                if term in self.vocabulary_:
                    j = self.vocabulary_[term]
                    tf = 1.0 + math.log(count)
                    X[i, j] = tf * self.idf_[j]

            norm = np.linalg.norm(X[i])
            if norm > 0:
                X[i] /= norm

        return X

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)


class MultinomialNaiveBayesClassifier:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for c_idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            total_docs = len(y)
            self.class_log_prior_[c_idx] = math.log(len(X_c) / total_docs)

            feature_counts = X_c.sum(axis=0) + self.alpha
            smoothed_total = feature_counts.sum()
            self.feature_log_prob_[c_idx] = np.log(feature_counts / smoothed_total)

        return self

    def predict_log_proba(self, X):
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_proba(self, X):
        log_prob = self.predict_log_proba(X)
        log_prob_max = np.max(log_prob, axis=1, keepdims=True)
        exp_prob = np.exp(log_prob - log_prob_max)
        return exp_prob / np.sum(exp_prob, axis=1, keepdims=True)

    def predict(self, X):
        log_prob = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_prob, axis=1)]


class SoftmaxLogisticRegression:
    def __init__(self, lr=0.2, epochs=450, l2_reg=0.001, random_state=42):
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.random_state = random_state
        self.classes_ = None
        self.weights_ = None
        self.bias_ = None

    def _softmax(self, z):
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Calculate class weights for balance
        class_counts = Counter(y)
        sample_weights = np.array([n_samples / (n_classes * class_counts[label]) for label in y])

        Y = np.zeros((n_samples, n_classes))
        for c_idx, c in enumerate(self.classes_):
            Y[y == c, c_idx] = 1.0

        self.weights_ = rng.randn(n_features, n_classes) * 0.01
        self.bias_ = np.zeros(n_classes)

        # Adam optimizer for fast, smooth convergence
        m_w = np.zeros_like(self.weights_)
        v_w = np.zeros_like(self.weights_)
        m_b = np.zeros_like(self.bias_)
        v_b = np.zeros_like(self.bias_)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.epochs + 1):
            logits = X @ self.weights_ + self.bias_
            probs = self._softmax(logits)
            error = (probs - Y) * sample_weights[:, None]

            grad_w = (X.T @ error) / n_samples + self.l2_reg * self.weights_
            grad_b = np.mean(error, axis=0)

            # Adam updates
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            self.weights_ -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            self.bias_ -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

        return self

    def predict_proba(self, X):
        logits = X @ self.weights_ + self.bias_
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class LinearSVMClassifier:
    """Multi-class One-vs-Rest Linear SVM with Pegasos subgradient updates."""
    def __init__(self, C=1.5, epochs=400, lr=0.08, random_state=42):
        self.C = C
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.classes_ = None
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.weights_ = np.zeros((n_classes, n_features))
        self.bias_ = np.zeros(n_classes)

        for c_idx, c in enumerate(self.classes_):
            y_bin = np.where(y == c, 1.0, -1.0)
            w = rng.randn(n_features) * 0.01
            b = 0.0

            pos_weight = (len(y) - np.sum(y == c)) / max(1, np.sum(y == c))

            for epoch in range(1, self.epochs + 1):
                eta = self.lr / (1 + 0.001 * epoch)
                scores = X @ w + b
                margins = y_bin * scores
                
                # Weight positive errors more to counter class imbalance
                weights = np.where(y_bin > 0, pos_weight, 1.0)
                viol = (margins < 1.0)

                grad_w = 0.01 * w - self.C * (X[viol].T @ (y_bin[viol] * weights[viol])) / n_samples
                grad_b = - self.C * np.sum(y_bin[viol] * weights[viol]) / n_samples

                w -= eta * grad_w
                b -= eta * grad_b

            self.weights_[c_idx] = w
            self.bias_[c_idx] = b

        return self

    def decision_function(self, X):
        return X @ self.weights_.T + self.bias_

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores_max = np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


class NearestCentroidClassifier:
    """Rocchio / Nearest Centroid with L2 normalized Class Prototypes."""
    def __init__(self):
        self.classes_ = None
        self.centroids_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.centroids_ = np.zeros((n_classes, n_features))

        for c_idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            centroid = np.mean(X_c, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            self.centroids_[c_idx] = centroid

        return self

    def predict_proba(self, X):
        sims = X @ self.centroids_.T  # Cosine similarities
        # Temperature scaled softmax
        temperature = 0.2
        sims_scaled = sims / temperature
        sims_max = np.max(sims_scaled, axis=1, keepdims=True)
        exp_s = np.exp(sims_scaled - sims_max)
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def predict(self, X):
        sims = X @ self.centroids_.T
        return self.classes_[np.argmax(sims, axis=1)]


def compute_metrics(y_true, y_pred, classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total = len(y_true)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in class_to_idx and p in class_to_idx:
            cm[class_to_idx[t], class_to_idx[p]] += 1

    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support)

        per_class[c] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "support": support
        }

    prec_macro = float(np.mean(precisions))
    rec_macro = float(np.mean(recalls))
    f1_macro = float(np.mean(f1s))

    tot_supp = sum(supports)
    if tot_supp > 0:
        prec_weighted = float(sum(p * s for p, s in zip(precisions, supports)) / tot_supp)
        rec_weighted = float(sum(r * s for r, s in zip(recalls, supports)) / tot_supp)
        f1_weighted = float(sum(f * s for f, s in zip(f1s, supports)) / tot_supp)
    else:
        prec_weighted = rec_weighted = f1_weighted = 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm.tolist(),
        "classes": list(classes),
        "classification_report": per_class
    }
