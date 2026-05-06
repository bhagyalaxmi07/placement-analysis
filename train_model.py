# train_model.py — ML Pipeline: Train, Evaluate, Save
# Run: python train_model.py
# Output: models/placement_model.pkl + prints full evaluation report

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection    import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.pipeline           import Pipeline
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics            import (classification_report, confusion_matrix,
                                         roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.inspection         import permutation_importance
from imblearn.over_sampling     import SMOTE
import matplotlib.pyplot as plt

from utils.data_loader import load_data, get_encoded_features
from config import MODEL_PATH

os.makedirs("models",  exist_ok=True)
os.makedirs("assets",  exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────
df = load_data()
X, y = get_encoded_features(df)

print(f"✅ Features: {X.shape[1]} | Samples: {X.shape[0]}")
print(f"   Class balance: {y.value_counts().to_dict()}")

# ── 2. Train / Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Handle Class Imbalance with SMOTE ──────────────────────
try:
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"✅ After SMOTE: {dict(pd.Series(y_train_res).value_counts())}")
except Exception:
    X_train_res, y_train_res = X_train, y_train
    print("⚠️  SMOTE skipped (no imbalance or error)")

# ── 4. Define Models ───────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=200, max_depth=8,
                                           random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                               max_depth=4, random_state=42))
    ]),
}

# ── 5. Cross-Validation Comparison ────────────────────────────
print("\n" + "="*55)
print("📊 5-Fold Cross Validation (ROC-AUC)")
print("="*55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_name  = None
best_model = None

cv_results = {}
for name, pipeline in models.items():
    scores = cross_val_score(pipeline, X_train_res, y_train_res,
                              cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    mean_s, std_s = scores.mean(), scores.std()
    print(f"  {name:<25} AUC = {mean_s:.4f} ± {std_s:.4f}")
    if mean_s > best_score:
        best_score = mean_s
        best_name  = name
        best_model = pipeline

print(f"\n🏆 Best Model: {best_name} (AUC = {best_score:.4f})")

# ── 6. Train Best Model on Full Train Set ─────────────────────
best_model.fit(X_train_res, y_train_res)

# ── 7. Test Set Evaluation ────────────────────────────────────
y_pred      = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
auc         = roc_auc_score(y_test, y_pred_prob)

print("\n" + "="*55)
print(f"📋 Test Set Evaluation — {best_name}")
print("="*55)
print(f"\nROC-AUC : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

# ── 8. Save Plots ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Model Evaluation — {best_name}", fontsize=13, fontweight="bold")

# 8a. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Not Placed", "Placed"]).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix")

# 8b. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color="#2ecc71", lw=2, label=f"AUC = {auc:.3f}")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

# 8c. Feature Importance
try:
    # Works for tree-based models
    clf        = best_model.named_steps["clf"]
    feat_names = X.columns.tolist()
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        importances = np.abs(clf.coef_[0])

    top_idx   = np.argsort(importances)[-15:]
    axes[2].barh([feat_names[i] for i in top_idx],
                  importances[top_idx], color="#3498db")
    axes[2].set_title("Top 15 Feature Importances")
except Exception as e:
    axes[2].text(0.5, 0.5, f"Feature importance\nnot available:\n{e}",
                  ha="center", va="center")

plt.tight_layout()
plt.savefig("assets/06_model_evaluation.png", dpi=150, bbox_inches="tight")
print("✅ Saved: assets/06_model_evaluation.png")
plt.close()

# 8d. CV Score comparison bar
fig, ax = plt.subplots(figsize=(8, 4))
names   = list(cv_results.keys())
means   = [v.mean() for v in cv_results.values()]
stds    = [v.std()  for v in cv_results.values()]
bars    = ax.bar(names, means, yerr=stds, color=["#3498db","#2ecc71","#e74c3c"],
                  capsize=6, edgecolor="white")
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("ROC-AUC")
ax.set_title("Cross-Validation Model Comparison")
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/07_model_comparison.png", dpi=150, bbox_inches="tight")
print("✅ Saved: assets/07_model_comparison.png")
plt.close()

# ── 9. Save Model + Metadata ──────────────────────────────────
model_artifact = {
    "model":        best_model,
    "model_name":   best_name,
    "feature_cols": X.columns.tolist(),
    "auc":          auc,
    "cv_score":     best_score,
}
joblib.dump(model_artifact, MODEL_PATH)
print(f"\n✅ Model saved → {MODEL_PATH}")
print("\n🎉 Training complete!")
