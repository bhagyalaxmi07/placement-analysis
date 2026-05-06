# config.py — Central config for the entire project

DATA_PATH = r"C:\Users\Amrut\OneDrive\Desktop\8th_sem_vtu\work\mini_projects\cap_stone_project\student_engagement\data\student_data_engament_Project_8.csv"
MODEL_PATH = "models/placement_model.pkl"

# ── Column Groups ──────────────────────────────────────────────
PROFILE_COLS = ["Student_ID", "College_Tier", "Department", "CGPA", "Gender"]

ATTENDANCE_COLS = [
    "Attendance_%", "Sessions_Attended", "Sessions_Missed", "Weekly_Consistency_Score"
]

PLATFORM_COLS = [
    "Login_Frequency", "Time_Spent_Hours", "Active_Days_Per_Week", "Session_Duration_Avg"
]

LEARNING_COLS = [
    "Videos_Watched", "Video_Completion_%", "Rewatch_Rate", "Notes_Taken"
]

QUIZ_COLS = [
    "Quizzes_Attempted", "Quiz_Submission_Rate", "Avg_Quiz_Score",
    "Assignment_Submissions", "On_Time_Submission_%"
]

DOUBT_COLS = [
    "Doubts_Raised", "Doubts_Resolved", "Doubt_Response_Time", "Peer_Discussion_Count"
]

EVENT_COLS = [
    "Hackathons_Attended", "Workshops_Attended", "Live_Sessions_Joined", "Project_Demo_Events"
]

SKILL_COLS = ["Skills_Learned_Count", "Project_Completion_Rate"]

TARGET_COL = "Placement_Status"

# ── Feature columns for ML (all numeric, excluding ID & target) ──
FEATURE_COLS = (
    ATTENDANCE_COLS + PLATFORM_COLS + LEARNING_COLS +
    QUIZ_COLS + DOUBT_COLS + EVENT_COLS + SKILL_COLS
)

# ── Engagement Segment Labels ──────────────────────────────────
SEGMENTS = {
    "High Performer":      "🏆 Placement Ready",
    "Passive Learner":     "📺 Encourage Interaction",
    "Active but Confused": "🤔 Needs Mentoring",
    "Disengaged":          "🚨 High Dropout Risk",
}

# ── Thresholds ─────────────────────────────────────────────────
ATTENDANCE_HIGH   = 80
ATTENDANCE_MED    = 60
QUIZ_HIGH         = 75
QUIZ_MED          = 50
LOGIN_HIGH        = 5
LOGIN_MED         = 3
TIME_HIGH         = 15
TIME_PLATEAU      = 30
