# app.py — Student Engagement Intelligence Dashboard
# Run: streamlit run app.py

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings("ignore")

from utils.data_loader import load_data, get_encoded_features
from config import TARGET_COL, MODEL_PATH, SEGMENTS

# Detect whether statsmodels is available (plotly's trendline='ols' requires it)
try:
    import statsmodels.api  # noqa: F401
    TRENDLINE_OLS = "ols"
except Exception:
    TRENDLINE_OLS = None

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Student Engagement Intelligence",
    page_icon   = "🎓",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body, .stApp, .main, .block-container {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    .main { background-color: #ffffff !important; }
    .metric-card {
        background: #ffffff;
        border-radius: 12px; padding: 18px 22px;
        border-left: 4px solid #4ade80; margin-bottom: 10px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
    }
    .metric-card.red   { border-left-color: #f87171; }
    .metric-card.blue  { border-left-color: #60a5fa; }
    .metric-card.yel   { border-left-color: #fbbf24; }
    .metric-title  { font-size: 13px; color: #475569; margin: 0; }
    .metric-value  { font-size: 28px; font-weight: 700; color: #0f172a; margin: 4px 0 0; }
    .segment-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 13px; font-weight: 600; margin: 3px;
    }
    .badge-green  { background:#d1fae5; color:#065f46; }
    .badge-yellow { background:#fef3c7; color:#78350f; }
    .badge-blue   { background:#e0f2fe; color:#084298; }
    .badge-red    { background:#fee2e2; color:#991b1b; }
    .section-header {
        font-size: 20px; font-weight: 700; color: #0f172a;
        border-bottom: 2px solid #4ade80; padding-bottom: 6px; margin: 24px 0 16px;
    }
</style>
""", unsafe_allow_html=True)


def style_plot(fig):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#111827",
        legend=dict(font_color="#111827"),
        margin=dict(t=50, b=40, l=40, r=40)
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zerolinecolor="#d1d5db",
        linecolor="#9ca3af",
        tickfont=dict(color="#111827"),
        title_font=dict(color="#111827")
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zerolinecolor="#d1d5db",
        linecolor="#9ca3af",
        tickfont=dict(color="#111827"),
        title_font=dict(color="#111827")
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Data & Model Loading (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    return load_data()

@st.cache_resource(show_spinner="Loading model...")
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

df    = get_data()
model = get_model()

# Notify user if trendline functionality is unavailable
if TRENDLINE_OLS is None:
    try:
        st.sidebar.warning("Optional package 'statsmodels' not installed — trendlines are disabled. Install with: pip install statsmodels")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Sidebar — Filters
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1e2130/4ade80?text=PragyanAI", width=200)
    st.markdown("## 🎛️ Filters")

    dept_opts  = ["All"] + sorted(df["Department"].unique().tolist())
    tier_opts  = ["All"] + sorted(df["College_Tier"].unique().tolist())
    seg_opts   = ["All"] + sorted(df["Segment"].unique().tolist())

    sel_dept   = st.selectbox("Department",    dept_opts)
    sel_tier   = st.selectbox("College Tier",  tier_opts)
    sel_seg    = st.selectbox("Segment",       seg_opts)
    cgpa_range = st.slider("CGPA Range", float(df["CGPA"].min()),
                            float(df["CGPA"].max()),
                            (float(df["CGPA"].min()), float(df["CGPA"].max())))

    st.markdown("---")
    st.markdown("### 📌 Navigation")
    page = st.radio("Go to", [
        "🏠 Overview",
        "📊 Engagement Analysis",
        "🧠 Learning Analytics",
        "💬 Interaction Insights",
        "🚨 Risk Detection",
        "🏆 Leaderboard",
        "🔮 Placement Predictor"
    ])

# ─────────────────────────────────────────────────────────────
# Apply Filters
# ─────────────────────────────────────────────────────────────
fdf = df.copy()
if sel_dept != "All":  fdf = fdf[fdf["Department"]   == sel_dept]
if sel_tier != "All":  fdf = fdf[fdf["College_Tier"]  == sel_tier]
if sel_seg  != "All":  fdf = fdf[fdf["Segment"]       == sel_seg]
fdf = fdf[(fdf["CGPA"] >= cgpa_range[0]) & (fdf["CGPA"] <= cgpa_range[1])]

n_total   = len(fdf)
n_placed  = int(fdf[TARGET_COL].sum())
n_atrisk  = int(fdf["At_Risk"].sum())
avg_eng   = fdf["Engagement_Score"].mean()


# ─────────────────────────────────────────────────────────────
# Helper: Top KPI Cards
# ─────────────────────────────────────────────────────────────
def kpi_cards():
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="metric-card">
        <p class="metric-title">Total Students</p>
        <p class="metric-value">{n_total:,}</p></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card blue">
        <p class="metric-title">Placement Rate</p>
        <p class="metric-value">{n_placed/n_total*100:.1f}%</p></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card yel">
        <p class="metric-title">Avg Engagement Score</p>
        <p class="metric-value">{avg_eng:.1f}/100</p></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-card red">
        <p class="metric-title">At-Risk Students</p>
        <p class="metric-value">{n_atrisk} ({n_atrisk/n_total*100:.1f}%)</p></div>""",
        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🎓 Student Engagement Intelligence System")
    st.markdown("> Tracks behavior → Predicts success → Drives placements")
    st.markdown("---")
    kpi_cards()

    col1, col2, col3 = st.columns(3)

    # Placement pie
    place_counts = fdf[TARGET_COL].value_counts().rename({1:"Placed", 0:"Not Placed"})
    fig_pie = px.pie(values=place_counts.values, names=place_counts.index,
                     color=place_counts.index,
                     color_discrete_map={"Placed":"#4ade80","Not Placed":"#f87171"},
                     title="Placement Distribution", hole=0.4)
    fig_pie = style_plot(fig_pie)
    col1.plotly_chart(fig_pie, use_container_width=True)

    # Segment bar
    seg_counts = fdf["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    seg_color  = {
        "High Performer":      "#4ade80",
        "Passive Learner":     "#60a5fa",
        "Active but Confused": "#fbbf24",
        "Disengaged":          "#f87171"
    }
    fig_seg = px.bar(seg_counts, x="Count", y="Segment", orientation="h",
                     color="Segment", color_discrete_map=seg_color,
                     title="Student Segments")
    fig_seg = style_plot(fig_seg)
    fig_seg.update_layout(showlegend=False)
    col2.plotly_chart(fig_seg, use_container_width=True)

    # Dept placement
    dept_place = fdf.groupby("Department")[TARGET_COL].mean().reset_index()
    dept_place.columns = ["Department","Placement_Rate"]
    fig_dept = px.bar(dept_place, x="Department", y="Placement_Rate",
                      color="Placement_Rate", color_continuous_scale="Greens",
                      title="Placement Rate by Department")
    fig_dept = style_plot(fig_dept)
    col3.plotly_chart(fig_dept, use_container_width=True)

    # Engagement Score distribution
    st.markdown('<p class="section-header">Engagement Score Distribution</p>',
                unsafe_allow_html=True)
    fig_eng = px.histogram(fdf, x="Engagement_Score", color=fdf[TARGET_COL].map({1:"Placed",0:"Not Placed"}),
                            barmode="overlay", nbins=30,
                            color_discrete_map={"Placed":"#4ade80","Not Placed":"#f87171"},
                            title="Engagement Score — Placed vs Not Placed")
    fig_eng = style_plot(fig_eng)
    fig_eng.update_layout(xaxis_title="Engagement Score (0–100)")
    st.plotly_chart(fig_eng, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 2 — ENGAGEMENT ANALYSIS
# ═════════════════════════════════════════════════════════════
elif page == "📊 Engagement Analysis":
    st.markdown("# 📊 Engagement Analysis")
    kpi_cards()

    tab1, tab2, tab3 = st.tabs(["Attendance", "Platform Activity", "Score Breakdown"])

    with tab1:
        c1, c2 = st.columns(2)
        # Attendance band vs placement rate
        band_place = fdf.groupby("Attendance_Band")[TARGET_COL].mean().reset_index()
        band_place.columns = ["Band", "Placement_Rate"]
        fig1 = px.bar(band_place, x="Band", y="Placement_Rate",
                      color="Placement_Rate", color_continuous_scale="RdYlGn",
                      title="Attendance Band → Placement Rate",
                      text=band_place["Placement_Rate"].apply(lambda x: f"{x:.0%}"))
        fig1 = style_plot(fig1)
        c1.plotly_chart(fig1, use_container_width=True)

        # Attendance scatter
        fig2 = px.scatter(fdf, x="Attendance_%", y="Engagement_Score",
                          color=fdf[TARGET_COL].map({1:"Placed",0:"Not Placed"}),
                          color_discrete_map={"Placed":"#4ade80","Not Placed":"#f87171"},
                          title="Attendance % vs Engagement Score",
                          opacity=0.6)
        fig2 = style_plot(fig2)
        c2.plotly_chart(fig2, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        fig3 = px.box(fdf, x=fdf[TARGET_COL].map({1:"Placed",0:"Not Placed"}),
                      y="Login_Frequency",
                      color=fdf[TARGET_COL].map({1:"Placed",0:"Not Placed"}),
                      color_discrete_map={"Placed":"#4ade80","Not Placed":"#f87171"},
                      title="Login Frequency vs Placement")
        fig3 = style_plot(fig3)
        c1.plotly_chart(fig3, use_container_width=True)

        fig4 = px.scatter(fdf, x="Time_Spent_Hours", y="Placement_Readiness",
                  color=fdf[TARGET_COL].map({1:"Placed",0:"Not Placed"}),
                  color_discrete_map={"Placed":"#4ade80","Not Placed":"#f87171"},
                  title="Time Spent (hrs/week) vs Placement Readiness",
                  trendline=TRENDLINE_OLS, opacity=0.6)
        fig4 = style_plot(fig4)
        c2.plotly_chart(fig4, use_container_width=True)

    with tab3:
        score_cols = ["Attendance_Score","Activity_Score","Learning_Score","Interaction_Score"]
        score_means = fdf.groupby(fdf[TARGET_COL].map({1:"Placed",0:"Not Placed"}))[score_cols].mean()
        fig5 = go.Figure()
        for group in score_means.index:
            fig5.add_trace(go.Bar(name=group, x=score_cols, y=score_means.loc[group],
                                   marker_color="#4ade80" if group=="Placed" else "#f87171"))
        fig5.update_layout(barmode="group", title="Score Breakdown: Placed vs Not Placed")
        fig5 = style_plot(fig5)
        st.plotly_chart(fig5, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 3 — LEARNING ANALYTICS
# ═════════════════════════════════════════════════════════════
elif page == "🧠 Learning Analytics":
    st.markdown("# 🧠 Learning Analytics")
    kpi_cards()

    c1, c2 = st.columns(2)
    # Quiz score vs placement
    fig1 = px.violin(fdf, x=fdf[TARGET_COL].map({1:"Placed",0:"Not Placed"}),
                     y="Avg_Quiz_Score",
                     color=fdf[TARGET_COL].map({1:"Placed",0:"Not Placed"}),
                     color_discrete_map={"Placed":"#4ade80","Not Placed":"#f87171"},
                     box=True, title="Quiz Score Distribution — Placed vs Not Placed")
    fig1 = style_plot(fig1)
    c1.plotly_chart(fig1, use_container_width=True)

    # Video completion
    fig2 = px.violin(fdf, x=fdf[TARGET_COL].map({1:"Placed",0:"Not Placed"}),
                     y="Video_Completion_%",
                     color=fdf[TARGET_COL].map({1:"Placed",0:"Not Placed"}),
                     color_discrete_map={"Placed":"#4ade80","Not Placed":"#f87171"},
                     box=True, title="Video Completion % Distribution")
    fig2 = style_plot(fig2)
    c2.plotly_chart(fig2, use_container_width=True)

    # Learning effectiveness
    st.markdown('<p class="section-header">Learning Effectiveness = Quiz Score × Video Completion</p>',
                unsafe_allow_html=True)
    fig3 = px.scatter(fdf, x="Video_Completion_%", y="Avg_Quiz_Score",
                      color="Learning_Effectiveness", size="Skills_Learned_Count",
                      color_continuous_scale="Viridis",
                      title="Video Completion vs Quiz Score (bubble = Skills Learned)",
                      opacity=0.7)
    fig3 = style_plot(fig3)
    st.plotly_chart(fig3, use_container_width=True)

    # Quiz band vs placement
    qb_place = fdf.groupby("Quiz_Band")[TARGET_COL].mean().reset_index()
    qb_place.columns = ["Band","Placement_Rate"]
    fig4 = px.bar(qb_place, x="Band", y="Placement_Rate",
                  color="Placement_Rate", color_continuous_scale="RdYlGn",
                  title="Quiz Band → Placement Rate",
                  text=qb_place["Placement_Rate"].apply(lambda x: f"{x:.0%}"))
    fig4 = style_plot(fig4)
    st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 4 — INTERACTION INSIGHTS
# ═════════════════════════════════════════════════════════════
elif page == "💬 Interaction Insights":
    st.markdown("# 💬 Interaction & Event Insights")
    kpi_cards()

    c1, c2 = st.columns(2)

    # Doubts raised buckets
    fdf["Doubt_Bucket"] = pd.cut(fdf["Doubts_Raised"], bins=[-1,0,3,10,100],
                                  labels=["No doubts","1–3","4–10","10+"])
    db_place = fdf.groupby("Doubt_Bucket")[TARGET_COL].mean().reset_index()
    db_place.columns = ["Doubt_Level","Placement_Rate"]
    fig1 = px.bar(db_place, x="Doubt_Level", y="Placement_Rate",
                  color="Placement_Rate", color_continuous_scale="RdYlGn",
                  title="Doubts Raised → Placement Rate 💡",
                  text=db_place["Placement_Rate"].apply(lambda x: f"{x:.0%}"))
    fig1 = style_plot(fig1)
    c1.plotly_chart(fig1, use_container_width=True)

    # Event participation
    fdf["Total_Events"] = (fdf["Hackathons_Attended"] + fdf["Workshops_Attended"] +
                            fdf["Live_Sessions_Joined"] + fdf["Project_Demo_Events"])
    fdf["Event_Bucket"] = pd.cut(fdf["Total_Events"], bins=[-1,0,2,5,100],
                                  labels=["0 events","1–2","3–5","5+"])
    ev_place = fdf.groupby("Event_Bucket")[TARGET_COL].mean().reset_index()
    ev_place.columns = ["Event_Level","Placement_Rate"]
    fig2 = px.bar(ev_place, x="Event_Level", y="Placement_Rate",
                  color="Placement_Rate", color_continuous_scale="RdYlGn",
                  title="Event Participation → Placement Rate 🎯",
                  text=ev_place["Placement_Rate"].apply(lambda x: f"{x:.0%}"))
    fig2 = style_plot(fig2)
    c2.plotly_chart(fig2, use_container_width=True)

    # Peer discussion vs engagement
    fig3 = px.scatter(fdf, x="Peer_Discussion_Count", y="Engagement_Score",
              color=fdf[TARGET_COL].map({1:"Placed",0:"Not Placed"}),
              color_discrete_map={"Placed":"#4ade80","Not Placed":"#f87171"},
              title="Peer Discussions vs Engagement Score", opacity=0.6,
              trendline=TRENDLINE_OLS)
    fig3 = style_plot(fig3)
    st.plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 5 — RISK DETECTION
# ═════════════════════════════════════════════════════════════
elif page == "🚨 Risk Detection":
    st.markdown("# 🚨 Risk Detection & Early Warning")

    col1, col2, col3 = st.columns(3)
    col1.metric("At-Risk Students",    f"{n_atrisk}",
                f"{n_atrisk/n_total*100:.1f}% of cohort", delta_color="inverse")
    col2.metric("Disengaged Students",
                int((fdf["Segment"]=="Disengaged").sum()))
    col3.metric("No Doubts Raised",
                int((fdf["Doubts_Raised"]==0).sum()))

    st.markdown("---")

    # Risk matrix heatmap
    st.markdown('<p class="section-header">Risk Matrix: Attendance × Quiz Score</p>',
                unsafe_allow_html=True)
    pivot = fdf.groupby(["Attendance_Band","Quiz_Band"])[TARGET_COL].mean().unstack()
    fig_h = px.imshow(pivot, color_continuous_scale="RdYlGn",
                      title="Placement Rate by Attendance & Quiz Band",
                      text_auto=".0%", aspect="auto")
    fig_h = style_plot(fig_h)
    st.plotly_chart(fig_h, use_container_width=True)

    # At-risk student table
    st.markdown('<p class="section-header">🚨 At-Risk Student List</p>',
                unsafe_allow_html=True)
    risk_cols = ["Student_ID","Department","CGPA","Attendance_%",
                 "Avg_Quiz_Score","Engagement_Score","Segment","Placement_Readiness"]
    risk_cols = [c for c in risk_cols if c in fdf.columns]
    risk_df   = fdf[fdf["At_Risk"]==1][risk_cols].sort_values("Engagement_Score")

    st.dataframe(
        risk_df.style
               .background_gradient(subset=["Engagement_Score","Placement_Readiness"],
                                     cmap="RdYlGn")
               .format({"Attendance_%": "{:.1f}%", "Avg_Quiz_Score": "{:.1f}",
                         "CGPA": "{:.2f}", "Engagement_Score": "{:.1f}",
                         "Placement_Readiness": "{:.1f}"}),
        use_container_width=True
    )
    st.download_button("⬇️ Export At-Risk List",
                        risk_df.to_csv(index=False),
                        "at_risk_students.csv", "text/csv")


# ═════════════════════════════════════════════════════════════
# PAGE 6 — LEADERBOARD
# ═════════════════════════════════════════════════════════════
elif page == "🏆 Leaderboard":
    st.markdown("# 🏆 Top Engaged Students")

    top_n = st.slider("Show Top N Students", 5, 50, 20)
    sort_by = st.selectbox("Rank By", ["Engagement_Score","Placement_Readiness",
                                         "Learning_Effectiveness","Avg_Quiz_Score"])

    leader_cols = ["Student_ID","Department","CGPA","Attendance_%",
                    "Engagement_Score","Learning_Effectiveness",
                    "Placement_Readiness","Segment",TARGET_COL]
    leader_cols = [c for c in leader_cols if c in fdf.columns]

    top_df = fdf[leader_cols].sort_values(sort_by, ascending=False).head(top_n).reset_index(drop=True)
    top_df.index += 1
    top_df[TARGET_COL] = top_df[TARGET_COL].map({1:"✅ Placed", 0:"❌ Not Placed"})

    st.dataframe(
        top_df.style
              .background_gradient(subset=["Engagement_Score","Placement_Readiness",
                                             "Learning_Effectiveness"], cmap="Greens")
              .format({"Attendance_%": "{:.1f}%", "Avg_Quiz_Score": "{:.1f}",
                        "CGPA": "{:.2f}", "Engagement_Score": "{:.1f}",
                        "Placement_Readiness": "{:.1f}",
                        "Learning_Effectiveness": "{:.1f}"}),
        use_container_width=True
    )

    # Radar chart for top 5
    st.markdown("### Radar: Top 5 Students")
    top5 = fdf.sort_values(sort_by, ascending=False).head(5)
    dims = ["Attendance_Score","Activity_Score","Learning_Score","Interaction_Score"]
    dims = [d for d in dims if d in top5.columns]

    fig_radar = go.Figure()
    colors = ["#4ade80","#60a5fa","#fbbf24","#a78bfa","#f87171"]
    for i, (_, row) in enumerate(top5.iterrows()):
        vals = [row[d] for d in dims]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=dims + [dims[0]],
            fill="toself", name=f"ID {int(row['Student_ID'])}",
            line_color=colors[i], opacity=0.7
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 25])),
        title="Engagement Radar — Top 5 Students"
    )
    fig_radar = style_plot(fig_radar)
    st.plotly_chart(fig_radar, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 7 — PLACEMENT PREDICTOR
# ═════════════════════════════════════════════════════════════
elif page == "🔮 Placement Predictor":
    st.markdown("# 🔮 Placement Predictor")

    if model is None:
        st.warning("⚠️ Model not found. Run `python train_model.py` first.")
        st.stop()

    st.markdown("Enter a student's engagement data to predict placement probability.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Profile**")
        cgpa        = st.slider("CGPA",                4.0, 10.0, 7.5, 0.01)
        attendance  = st.slider("Attendance %",         0,   100,  75)
        consistency = st.slider("Weekly Consistency",   0,   10,    6)
        sessions    = st.slider("Sessions Attended",    0,   100,  60)

    with col2:
        st.markdown("**🖥️ Platform Activity**")
        login_freq  = st.slider("Login Freq/week",      1,    7,    4)
        time_spent  = st.slider("Time Spent (hrs/wk)",  0,   40,   12)
        active_days = st.slider("Active Days/week",     0,    7,    4)
        session_dur = st.slider("Avg Session (mins)",   5,  120,   45)

    with col3:
        st.markdown("**📚 Learning & Quiz**")
        videos      = st.slider("Videos Watched",       0,  100,  40)
        vid_comp    = st.slider("Video Completion %",   0,  100,  70)
        quiz_score  = st.slider("Avg Quiz Score",       0,  100,  65)
        quizzes     = st.slider("Quizzes Attempted",    0,   50,  20)

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**💬 Interaction**")
        doubts      = st.slider("Doubts Raised",        0,   30,   5)
        peer_disc   = st.slider("Peer Discussions",     0,   30,   8)
        hackathons  = st.slider("Hackathons",           0,   10,   2)
        workshops   = st.slider("Workshops",            0,   10,   2)

    with col5:
        st.markdown("**🎯 Projects & Skills**")
        live_sess   = st.slider("Live Sessions Joined", 0,   20,   5)
        proj_demo   = st.slider("Project Demo Events",  0,   10,   1)
        skills      = st.slider("Skills Learned",       0,   30,  10)
        proj_comp   = st.slider("Project Completion %", 0,  100,  60)

    if st.button("🔮 Predict Placement", use_container_width=True, type="primary"):

        # Build input row matching FEATURE_COLS
        input_data = {
            "Attendance_%":          attendance,
            "Sessions_Attended":     sessions,
            "Sessions_Missed":       100 - sessions,
            "Weekly_Consistency_Score": consistency,
            "Login_Frequency":       login_freq,
            "Time_Spent_Hours":      time_spent,
            "Active_Days_Per_Week":  active_days,
            "Session_Duration_Avg":  session_dur,
            "Videos_Watched":        videos,
            "Video_Completion_%":    vid_comp,
            "Rewatch_Rate":          0.2,
            "Notes_Taken":           1 if vid_comp > 70 else 0,
            "Quizzes_Attempted":     quizzes,
            "Quiz_Submission_Rate":  0.8,
            "Avg_Quiz_Score":        quiz_score,
            "Assignment_Submissions":quizzes,
            "On_Time_Submission_%":  75,
            "Doubts_Raised":         doubts,
            "Doubts_Resolved":       int(doubts * 0.8),
            "Doubt_Response_Time":   24,
            "Peer_Discussion_Count": peer_disc,
            "Hackathons_Attended":   hackathons,
            "Workshops_Attended":    workshops,
            "Live_Sessions_Joined":  live_sess,
            "Project_Demo_Events":   proj_demo,
            "Skills_Learned_Count":  skills,
            "Project_Completion_Rate": proj_comp,
            # Composite features
            "Attendance_Score":      attendance / 100 * 25,
            "Activity_Score":        (login_freq/7 + min(time_spent,30)/30 + active_days/7) / 3 * 25,
            "Learning_Score":        (vid_comp/100 + quiz_score/100) / 2 * 25,
            "Interaction_Score":     (doubts/20 + (hackathons+workshops+live_sess+proj_demo)/20 + peer_disc/20) / 3 * 25,
            "Engagement_Score":      0,  # will compute below
            "Learning_Effectiveness": (quiz_score/100) * (vid_comp/100) * 100,
            "Placement_Readiness":   0,
        }

        input_data["Engagement_Score"] = (
            input_data["Attendance_Score"] + input_data["Activity_Score"] +
            input_data["Learning_Score"]   + input_data["Interaction_Score"]
        )
        input_data["Placement_Readiness"] = (
            input_data["Engagement_Score"] * 0.5 +
            input_data["Learning_Effectiveness"] * 0.3 +
            input_data["Interaction_Score"] / 25 * 100 * 0.2
        )

        feat_cols = model["feature_cols"]
        inp_df    = pd.DataFrame([{c: input_data.get(c, 0) for c in feat_cols}])

        prob      = model["model"].predict_proba(inp_df)[0][1]
        pred      = int(prob >= 0.5)

        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        rc1, rc2, rc3 = st.columns(3)
        color = "#4ade80" if pred == 1 else "#f87171"
        label = "✅ PLACED" if pred == 1 else "❌ NOT PLACED"

        rc1.markdown(f"""<div class="metric-card" style="border-left-color:{color}; text-align:center;">
            <p class="metric-title">Prediction</p>
            <p class="metric-value" style="color:{color};">{label}</p></div>""",
            unsafe_allow_html=True)
        rc2.markdown(f"""<div class="metric-card blue" style="text-align:center;">
            <p class="metric-title">Placement Probability</p>
            <p class="metric-value">{prob*100:.1f}%</p></div>""", unsafe_allow_html=True)
        rc3.markdown(f"""<div class="metric-card yel" style="text-align:center;">
            <p class="metric-title">Engagement Score</p>
            <p class="metric-value">{input_data['Engagement_Score']:.1f}/100</p></div>""",
            unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = prob * 100,
            delta = {"reference": 50},
            gauge = {
                "axis":  {"range": [0, 100]},
                "bar":   {"color": color},
                "steps": [
                    {"range": [0,  40], "color": "#7f1d1d"},
                    {"range": [40, 60], "color": "#78350f"},
                    {"range": [60,100], "color": "#064e3b"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 50}
            },
            title = {"text": "Placement Probability"}
        ))
        fig_gauge = style_plot(fig_gauge)
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Actionable recommendations
        st.markdown("### 📌 Recommendations")
        recs = []
        if attendance < 60:  recs.append("🔴 Attendance is critical — must exceed 60%")
        if quiz_score  < 50: recs.append("🔴 Quiz score too low — practice more MCQs")
        if doubts      == 0: recs.append("🟡 No doubts raised — engage with mentors")
        if time_spent  < 5:  recs.append("🔴 Spend at least 5+ hours/week on platform")
        if hackathons + workshops < 1: recs.append("🟡 Participate in at least 1 event")
        if not recs:         recs.append("🟢 Great engagement! Stay consistent.")

        for r in recs:
            st.markdown(f"- {r}")
