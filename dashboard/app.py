"""
Streamlit Dashboard for Monitoring Harmful Content.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import altair as alt

from services.mongodb_query import get_recent_predictions, get_stats, get_time_series_data
# from services.file_query import get_recent_predictions, get_stats, get_time_series_data
from services.minio_reader import get_video_url
from components.stats_box import render_stats_box
from components.charts import render_recent_trend, render_label_dist
from components.video_player import render_video_player

# Page Config
st.set_page_config(page_title="TikTok Safety Dashboard", layout="wide", page_icon="🛡️")

# Load CSS
with open("dashboard/styles/theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Filters")
label_filter = st.sidebar.selectbox("Filter by Label", ["All", "Harmful", "Safe"])
refresh_rate = st.sidebar.slider("Refresh Rate (sec)", 5, 60, 10)

# Header
st.title("🛡️ TikTok Harmful Content Detection")
st.markdown("Real-time monitoring of video ingestion streams.")

# Data Fetch
stats = get_stats()
recent_preds = get_recent_predictions(limit=50, label_filter=label_filter)
trend_df = get_time_series_data()

# Top Stats Row
c1, c2, c3, c4 = st.columns(4)
total = stats.get("total", 0)
harmful = stats.get("counts", {}).get("Harmful", 0)
safe = stats.get("counts", {}).get("Safe", 0)
harmful_pct = (harmful / total * 100) if total > 0 else 0

with c1:
    render_stats_box("Total Processed", str(total))
with c2:
    render_stats_box("Harmful Detected", str(harmful), color="inverse")
with c3:
    render_stats_box("Safe", str(safe))
with c4:
    render_stats_box("Risk Ratio", f"{harmful_pct:.1f}%")

st.markdown("---")

# Main Content Grid
col_charts, col_table = st.columns([1, 2])

with col_charts:
    st.subheader("Analytics")
    st.write("#### Recent Trend")
    render_recent_trend(trend_df)
    
    st.write("#### Label Distribution")
    render_label_dist(stats)

with col_table:
    st.subheader("Recent Alerts")
    
    if recent_preds:
        # Convert to DF for clean display
        df_display = pd.DataFrame(recent_preds)
        
        # Select/Rename cols
        cols_to_show = ["video_id", "label", "confidence", "ingested_at"]
        df_show = df_display[cols_to_show].copy()
        
        # Interactive Selection
        # Use selection to show video player
        event = st.dataframe(
            df_show,
            use_container_width=True,
            on_select="rerun", # Streamlit 1.35+ feature
            selection_mode="single-row",
            hide_index=True
        )
        
        # If row selected, show video player below
        if event and event.selection and event.selection.rows:
            idx = event.selection.rows[0]
            selected_row = df_show.iloc[idx]
            vid = selected_row["video_id"]
            
            st.write(f"### Playback: {vid}")
            url = get_video_url(vid)
            if url:
                render_video_player(url)
            else:
                st.warning("Video file not found in Bronze layer.")
    else:
        st.info("No recent predictions found matching criteria.")

# Auto Refresh logic (simple)
# Note: Streamlit normally handles this via st.empty loop or autorefresh text
# For now, simplistic manual refresh button is standard unless we use st_autorefresh component (external)
if st.sidebar.button("Refresh Data"):
    st.rerun()
