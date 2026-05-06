# utils/data_loader.py — Load, clean, and engineer features

import pandas as pd
import numpy as np
from config import (
    DATA_PATH, FEATURE_COLS, TARGET_COL,
    ATTENDANCE_HIGH, ATTENDANCE_MED, QUIZ_HIGH, QUIZ_MED,
    LOGIN_HIGH, LOGIN_MED, TIME_HIGH, TIME_PLATEAU
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and return a clean DataFrame."""
    df = pd.read_csv(path)
    # Normalize common alternate column names found in datasets
    rename_map = {}
    if "Quiz_Score" in df.columns and "Avg_Quiz_Score" not in df.columns:
        rename_map["Quiz_Score"] = "Avg_Quiz_Score"
    if "Active_Days" in df.columns and "Active_Days_Per_Week" not in df.columns:
        rename_map["Active_Days"] = "Active_Days_Per_Week"
    if "Hackathons" in df.columns and "Hackathons_Attended" not in df.columns:
        rename_map["Hackathons"] = "Hackathons_Attended"
    if "Workshops" in df.columns and "Workshops_Attended" not in df.columns:
        rename_map["Workshops"] = "Workshops_Attended"
    if "Placement_Status" in df.columns and TARGET_COL != "Placement_Status":
        # keep the configured TARGET_COL, but alias common name
        rename_map["Placement_Status"] = TARGET_COL
    if rename_map:
        df = df.rename(columns=rename_map)
    df = _clean(df)
    df = _engineer_features(df)
    df = _segment_students(df)
    return df


# ── Internal helpers ───────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle nulls, types, and obvious outliers."""
    # Drop full-duplicate rows
    df = df.drop_duplicates()

    # Encode binary columns
    if "Notes_Taken" in df.columns:
        df["Notes_Taken"] = df["Notes_Taken"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Encode target (support both numeric 0/1 and string labels)
    if TARGET_COL in df.columns:
        # If target is textual ('Placed'/'Not Placed'), map to 0/1
        if df[TARGET_COL].dtype == object or df[TARGET_COL].isin(["Placed", "Not Placed"]).any():
            df[TARGET_COL] = df[TARGET_COL].map({"Placed": 1, "Not Placed": 0})
        # Coerce numeric-like targets to integers when possible
        try:
            df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        except Exception:
            pass
        # Fill missing with mode if available, else 0
        if df[TARGET_COL].isna().any():
            modes = df[TARGET_COL].mode()
            if not modes.empty:
                df[TARGET_COL] = df[TARGET_COL].fillna(modes[0])
            else:
                df[TARGET_COL] = df[TARGET_COL].fillna(0)

    # Fill numeric nulls with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical nulls with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create composite engagement scores."""
    # Ensure all expected feature columns exist (fill missing cols with 0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    # 1. Attendance Score (0–25)
    df["Attendance_Score"] = (df["Attendance_%"] / 100 * 25).round(2)

    # 2. Activity Score (0–25) — login + time + active days
    login_norm = df["Login_Frequency"].clip(0, 7) / 7
    time_norm  = df["Time_Spent_Hours"].clip(0, TIME_PLATEAU) / TIME_PLATEAU
    days_norm  = df["Active_Days_Per_Week"].clip(0, 7) / 7
    df["Activity_Score"] = ((login_norm + time_norm + days_norm) / 3 * 25).round(2)

    # 3. Learning Score (0–25) — video + quiz
    video_norm = df["Video_Completion_%"] / 100
    quiz_norm  = df["Avg_Quiz_Score"] / 100
    df["Learning_Score"] = ((video_norm + quiz_norm) / 2 * 25).round(2)

    # 4. Interaction Score (0–25) — doubts + events + peers
    doubts_norm = (df["Doubts_Raised"].clip(0, 20) / 20)
    events_norm = (
        (df["Hackathons_Attended"] + df["Workshops_Attended"] +
         df["Live_Sessions_Joined"] + df["Project_Demo_Events"]).clip(0, 20) / 20
    )
    peer_norm = df["Peer_Discussion_Count"].clip(0, 20) / 20
    df["Interaction_Score"] = ((doubts_norm + events_norm + peer_norm) / 3 * 25).round(2)

    # 5. Composite Engagement Score (0–100)
    df["Engagement_Score"] = (
        df["Attendance_Score"] + df["Activity_Score"] +
        df["Learning_Score"]  + df["Interaction_Score"]
    ).round(2)

    # 6. Learning Effectiveness (quiz × video completion)
    df["Learning_Effectiveness"] = (
        (df["Avg_Quiz_Score"] / 100) * (df["Video_Completion_%"] / 100) * 100
    ).round(2)

    # 7. Placement Readiness Score (0–100)
    df["Placement_Readiness"] = (
        df["Engagement_Score"] * 0.5 +
        df["Learning_Effectiveness"] * 0.3 +
        df["Interaction_Score"] / 25 * 100 * 0.2
    ).round(2)

    # 8. Attendance Band
    df["Attendance_Band"] = pd.cut(
        df["Attendance_%"],
        bins=[0, ATTENDANCE_MED, ATTENDANCE_HIGH, 100],
        labels=["Low (<60%)", "Medium (60–80%)", "High (>80%)"]
    )

    # 9. Quiz Band
    df["Quiz_Band"] = pd.cut(
        df["Avg_Quiz_Score"],
        bins=[0, QUIZ_MED, QUIZ_HIGH, 100],
        labels=["Low (<50)", "Medium (50–75)", "High (>75)"]
    )

    # 10. Risk Flag
    df["At_Risk"] = (
        (df["Attendance_%"] < ATTENDANCE_MED) &
        (df["Avg_Quiz_Score"] < QUIZ_MED)
    ).astype(int)

    return df


def _segment_students(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each student to a behavioral segment."""

    def assign_segment(row):
        high_eng  = row["Engagement_Score"] >= 60
        high_quiz = row["Avg_Quiz_Score"]    >= QUIZ_HIGH
        active    = row["Doubts_Raised"]      >= 3

        if high_eng and high_quiz and active:
            return "High Performer"
        elif high_eng and not active:
            return "Passive Learner"
        elif high_eng and not high_quiz:
            return "Active but Confused"
        else:
            return "Disengaged"

    df["Segment"] = df.apply(assign_segment, axis=1)
    return df


def get_encoded_features(df: pd.DataFrame):
    """Return X (feature matrix) and y (target vector) ready for sklearn."""
    feature_cols = FEATURE_COLS + [
        "Attendance_Score", "Activity_Score", "Learning_Score",
        "Interaction_Score", "Engagement_Score",
        "Learning_Effectiveness", "Placement_Readiness"
    ]
    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y
