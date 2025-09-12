"""
Utility script to generate evaluation plots referenced in the LNCS report.
Run AFTER model training & evaluation artifacts exist.

Usage:
  python reports/generate_figures.py \
      --model_dir models \
      --data_dir data \
      --reports_dir reports
"""
import argparse
import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix
)
from scipy.sparse import csr_matrix

def load_test(data_dir):
    x_test_path = os.path.join(data_dir, "x_test.npy")
    y_test_path = os.path.join(data_dir, "y_test.npy")
    return np.load(x_test_path, allow_pickle=True), np.load(y_test_path, allow_pickle=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)

    # Load model and vectorizer
    model = joblib.load(os.path.join(args.model_dir, "logistic_regression_model.joblib"))
    vectorizer = joblib.load(os.path.join(args.model_dir, "vectorizer.joblib"))

    # Load test data
    # X_test_raw, y_test = load_test(args.data_dir)

    # # Fix: Convert X_test_raw to proper string list
    # # Ensure raw test data is a flat list of strings
    # X_test_raw = X_test_raw.tolist()
    # X_test_raw = [x if isinstance(x, str) else x[0] for x in X_test_raw]
    # # Flatten 2D array of shape (N,1) to list

    # # Vectorize
    # X_test = vectorizer.transform(X_test_raw)

    X_test_vec, y_test = load_test(args.data_dir)

    # X_test_vec is already a dense numpy array of vectors, convert to sparse matrix if needed
    from scipy.sparse import csr_matrix
    X_test = csr_matrix(X_test_vec)


    # Get prediction probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        # decision_function fallback
        raw = model.decision_function(X_test)
        # min-max normalize
        probs = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    # Confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(args.reports_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4.2, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.reports_dir, "roc_curve.png"), dpi=200)
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(4.2, 4))
    plt.plot(recall, precision, label=f"AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.reports_dir, "pr_curve.png"), dpi=200)
    plt.close()

    # Error distribution (illustrative)
    error_categories = {
        "satire_like": 18,
        "sensational_real": 30,
        "subtle_fake": 12,
        "entity_novelty": 9,
        "code_switch": 6,
        "other": 12
    }
    labels = list(error_categories.keys())
    values = [error_categories[k] for k in labels]
    plt.figure(figsize=(5.2, 3.6))
    plt.bar(labels, values, color="#2d76b5")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count (heuristic)")
    plt.title("Error Category Distribution (Illustrative)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.reports_dir, "error_distribution.png"), dpi=200)
    plt.close()

    # Save summary JSON
    summary = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "error_category_counts": error_categories
    }
    with open(os.path.join(args.reports_dir, "figure_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Figures generated in:", args.reports_dir)

if __name__ == "__main__":
    main()
