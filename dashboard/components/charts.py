import streamlit as st
import pandas as pd
import altair as alt

def render_recent_trend(df: pd.DataFrame):
    """
    Render a time series line chart.
    """
    if df.empty:
        st.info("No data for trend chart.")
        return

    # Count per minute/hour
    # Assuming df has 'ingested_at' and 'label'
    
    # Resample to 5min
    chart_data = df.set_index("ingested_at").resample("5T")["label"].count().reset_index()
    chart_data.columns = ["time", "count"]
    
    c = alt.Chart(chart_data).mark_line(point=True).encode(
        x="time",
        y="count",
        tooltip=["time", "count"]
    ).interactive()
    
    st.altair_chart(c, use_container_width=True)

def render_label_dist(stats: dict):
    """
    Render aggregation (Pie/Bar).
    """
    counts = stats.get("counts", {})
    if not counts:
        return
        
    data = pd.DataFrame(list(counts.items()), columns=["Label", "Count"])
    
    c = alt.Chart(data).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Label", type="nominal"),
        tooltip=["Label", "Count"]
    )
    st.altair_chart(c, use_container_width=True)
