# eda.py — Exploratory Data Analysis
# Run: python eda.py
# Outputs: saves plots to assets/ folder and prints summary stats

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import warnings
warnings.filterwarnings("ignore")

from utils.data_loader import load_data
from config import TARGET_COL

# ── Setup ──────────────────────────────────────────────────────
os.makedirs("assets", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
COLORS = {"Placed": "#2ecc71", "Not Placed": "#e74c3c", "main": "#3498db"}

df = load_data()
print(f"✅ Dataset loaded: {df.shape[0]} students, {df.shape[1]} columns")
print(f"\n📊 Target distribution:\n{df[TARGET_COL].value_counts().rename({1:'Placed', 0:'Not Placed'})}")
print(f"\n🧩 Segments:\n{df['Segment'].value_counts()}")


# ─────────────────────────────────────────────────────────────
# PLOT 1 — Target Distribution + Segment Breakdown
# ─────────────────────────────────────────────────────────────
figs = []
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Student Overview", fontsize=14, fontweight="bold")

# 1a. Placement pie
labels = ["Placed", "Not Placed"]
vals   = df[TARGET_COL].value_counts().reindex([1, 0]).values
axes[0].pie(vals, labels=labels, autopct="%1.1f%%",
            colors=[COLORS["Placed"], COLORS["Not Placed"]], startangle=90)
axes[0].set_title("Placement Distribution")

# 1b. Segment bar
seg_counts = df["Segment"].value_counts()
seg_counts.plot(kind="barh", ax=axes[1], color="#3498db", edgecolor="white")
axes[1].set_title("Student Segments")
axes[1].set_xlabel("Count")

# 1c. Department vs Placement
dept_place = df.groupby("Department")[TARGET_COL].mean().sort_values()
dept_place.plot(kind="barh", ax=axes[2], color="#9b59b6", edgecolor="white")
axes[2].set_title("Placement Rate by Department")
axes[2].set_xlabel("Placement Rate")

plt.tight_layout()
plt.savefig("assets/01_overview.png", dpi=150, bbox_inches="tight")
figs.append(fig)
print("✅ Saved: assets/01_overview.png")
plt.close()


# ─────────────────────────────────────────────────────────────
# PLOT 2 — Engagement Score vs Placement (Key Insight)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Core Engagement Metrics vs Placement", fontsize=14, fontweight="bold")

metrics = [
    ("Attendance_%",        "Attendance %"),
    ("Login_Frequency",     "Logins / Week"),
    ("Time_Spent_Hours",    "Time Spent (hrs/week)"),
    ("Avg_Quiz_Score",      "Avg Quiz Score"),
    ("Video_Completion_%",  "Video Completion %"),
    ("Doubts_Raised",       "Doubts Raised"),
]

for ax, (col, label) in zip(axes.flat, metrics):
    placed     = df[df[TARGET_COL] == 1][col]
    not_placed = df[df[TARGET_COL] == 0][col]
    ax.hist(not_placed, bins=20, alpha=0.6, color=COLORS["Not Placed"], label="Not Placed", density=True)
    ax.hist(placed,     bins=20, alpha=0.6, color=COLORS["Placed"],     label="Placed",     density=True)
    ax.axvline(placed.mean(),     color=COLORS["Placed"],     linestyle="--", lw=1.5,
               label=f"Placed mean: {placed.mean():.1f}")
    ax.axvline(not_placed.mean(), color=COLORS["Not Placed"], linestyle="--", lw=1.5,
               label=f"Not Placed mean: {not_placed.mean():.1f}")
    ax.set_title(label)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("assets/02_engagement_vs_placement.png", dpi=150, bbox_inches="tight")
figs.append(fig)
print("✅ Saved: assets/02_engagement_vs_placement.png")
plt.close()


# ─────────────────────────────────────────────────────────────
# PLOT 3 — Composite Scores Distribution
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Composite Score Distributions", fontsize=14, fontweight="bold")

scores = ["Engagement_Score", "Learning_Effectiveness", "Placement_Readiness"]
titles = ["Engagement Score (0–100)", "Learning Effectiveness", "Placement Readiness"]

for ax, col, title in zip(axes, scores, titles):
    for val, label, color in [(1, "Placed", COLORS["Placed"]), (0, "Not Placed", COLORS["Not Placed"])]:
        subset = df[df[TARGET_COL] == val][col]
        ax.hist(subset, bins=20, alpha=0.6, color=color, label=label, density=True)
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig("assets/03_composite_scores.png", dpi=150, bbox_inches="tight")
figs.append(fig)
print("✅ Saved: assets/03_composite_scores.png")
plt.close()


# ─────────────────────────────────────────────────────────────
# PLOT 4 — Correlation Heatmap
# ─────────────────────────────────────────────────────────────
num_df = df.select_dtypes(include=[np.number]).drop(columns=["Student_ID"], errors="ignore")
corr   = num_df.corr()

# Top 15 features correlated with target
top_feats = corr[TARGET_COL].abs().sort_values(ascending=False).head(16).index.tolist()
corr_sub  = num_df[top_feats].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_sub, dtype=bool))
sns.heatmap(corr_sub, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation Heatmap — Top Features vs Placement", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
figs.append(fig)
print("✅ Saved: assets/04_correlation_heatmap.png")
plt.close()


# ─────────────────────────────────────────────────────────────
# PLOT 5 — Attendance Band × Quiz Band vs Placement (Risk Matrix)
# ─────────────────────────────────────────────────────────────
pivot = df.groupby(["Attendance_Band", "Quiz_Band"])[TARGET_COL].mean().unstack()

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn", linewidths=0.5,
            ax=ax, vmin=0, vmax=1)
ax.set_title("Placement Rate: Attendance Band × Quiz Band", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/05_risk_matrix.png", dpi=150, bbox_inches="tight")
figs.append(fig)
print("✅ Saved: assets/05_risk_matrix.png")
plt.close()


# ─────────────────────────────────────────────────────────────
# PRINT — Key Statistical Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📋 KEY INSIGHTS")
print("="*60)

placed     = df[df[TARGET_COL] == 1]
not_placed = df[df[TARGET_COL] == 0]

for col in ["Attendance_%", "Engagement_Score", "Avg_Quiz_Score",
            "Login_Frequency", "Doubts_Raised", "Placement_Readiness"]:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Placed:     {placed[col].mean():.2f}")
        print(f"  Not Placed: {not_placed[col].mean():.2f}")

print("\n" + "="*60)
print(f"🚨 At-Risk Students: {df['At_Risk'].sum()} ({df['At_Risk'].mean()*100:.1f}%)")
print("="*60)
print("\n✅ EDA complete. All plots saved to assets/")

# --- Save a combined PDF report containing all figures + a structured summary page
os.makedirs("assets", exist_ok=True)
pdf_path = "assets/EDA_report.pdf"

# Prepare summary tables
target_dist = df[TARGET_COL].value_counts().rename({1: 'Placed', 0: 'Not Placed'})
target_df = target_dist.reset_index()
target_df.columns = ['Placement', 'Count']

segments = df['Segment'].value_counts()
segments_df = segments.reset_index()
segments_df.columns = ['Segment', 'Count']

metrics = []
metric_cols = ["Attendance_%", "Engagement_Score", "Avg_Quiz_Score",
               "Login_Frequency", "Doubts_Raised", "Placement_Readiness"]
for col in metric_cols:
    if col in df.columns:
        metrics.append([col,
                        f"{placed[col].mean():.2f}",
                        f"{not_placed[col].mean():.2f}"])
metrics_df = None
if metrics:
    import pandas as _pd
    metrics_df = _pd.DataFrame(metrics, columns=['Metric', 'Placed Mean', 'Not Placed Mean'])

with PdfPages(pdf_path) as pp:
    for f in figs:
        pp.savefig(f)

    # 1) Target distribution table page
    fig_t = plt.figure(figsize=(8.27, 11.69))
    ax = fig_t.add_subplot(111)
    ax.axis('off')
    ax.set_title('Target Distribution', fontsize=14, pad=20)
    tbl = ax.table(cellText=target_df.values, colLabels=target_df.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2)
    pp.savefig(fig_t)
    plt.close(fig_t)

    # 2) Segments table page
    fig_s = plt.figure(figsize=(8.27, 11.69))
    ax = fig_s.add_subplot(111)
    ax.axis('off')
    ax.set_title('Segments (counts)', fontsize=14, pad=20)
    tbl = ax.table(cellText=segments_df.values, colLabels=segments_df.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2)
    pp.savefig(fig_s)
    plt.close(fig_s)

    # 3) Metrics table page
    if metrics_df is not None:
        fig_m = plt.figure(figsize=(8.27, 11.69))
        ax = fig_m.add_subplot(111)
        ax.axis('off')
        ax.set_title('Metric Means (Placed vs Not Placed)', fontsize=14, pad=20)
        tbl = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        pp.savefig(fig_m)
        plt.close(fig_m)

print(f"✅ Combined PDF saved: {pdf_path}")
