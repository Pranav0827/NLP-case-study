"""
Case Study 4: Standalone Reproducible Evaluation CLI.
Loads cs4_traffic/models/evaluation_metrics.json and prints standard reproducible metrics.
"""

import os
import json

def display_evaluation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_dir, "models", "evaluation_metrics.json")
    
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Please run train_models.py first.")
        return
        
    with open(metrics_path, "r") as f:
        data = json.load(f)
        
    print("\n" + "="*85)
    print("      PUNE TRAFFIC COMPLAINTS NLP BENCHMARK (CASE STUDY 4)")
    print("="*85)
    
    stats = data.get("dataset_statistics", {})
    print(f"Dataset Total Records : {stats.get('total_records')} (Train: {stats.get('train_records')}, Test: {stats.get('test_records')})")
    print(f"Informal Text Nuances : Typos={stats.get('typo_percentage')}% | Lowercase={stats.get('all_lowercase_percentage')}% | Multi-Loc={stats.get('multi_location_percentage')}%")
    print(f"Target Classes (6)    : {', '.join(data.get('categories', []))}")
    print("-"*85)
    
    print("\n1. 6-CATEGORY CLASSIFICATION BENCHMARK (ON 20% TEST SPLIT):")
    header = f"{'Model':<26} | {'Accuracy':<9} | {'Prec (Macro)':<12} | {'Rec (Macro)':<12} | {'F1 (Macro)':<10} | {'F1 (Wt)':<8}"
    print(header)
    print("-" * len(header))
    
    models = data.get("classification_models", {})
    for mname, m in models.items():
        print(f"{mname:<26} | {m['accuracy']:<9.4f} | {m['precision_macro']:<12.4f} | {m['recall_macro']:<12.4f} | {m['f1_macro']:<10.4f} | {m['f1_weighted']:<8.4f}")
        
    print(f"\n--> Recommended Best Classifier: {data.get('best_model')}")
    
    print("\n2. LOCATION NAMED ENTITY RECOGNITION (NER) EVALUATION:")
    ner = data.get("location_ner_metrics", {})
    print(f"  - Entity Precision    : {ner.get('precision', 0):.4f}")
    print(f"  - Entity Recall       : {ner.get('recall', 0):.4f}")
    print(f"  - Entity F1-Score     : {ner.get('f1_score', 0):.4f}")
    print(f"  - Exact Match Rate    : {ner.get('exact_match_rate', 0):.4f}")
    print(f"  - Ground Truth Entities: {ner.get('total_gold', 0)} | Extracted Entities: {ner.get('total_pred', 0)}")
    
    robust = ner.get("subgroup_robustness", {})
    print("\n3. NER ROBUSTNESS ON INFORMAL TEXT SUBGROUPS:")
    print(f"  - Typos / Misspellings (F1) : {robust.get('has_typos', {}).get('f1', 0):.4f}")
    print(f"  - All-Lowercase Text (F1)   : {robust.get('is_all_lowercase', {}).get('f1', 0):.4f}")
    print(f"  - Multi-Location Text (F1)  : {robust.get('is_multi_location', {}).get('f1', 0):.4f}")
    print("="*85 + "\n")

if __name__ == "__main__":
    display_evaluation()
