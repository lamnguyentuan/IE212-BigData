import streamlit as st

def render_stats_box(label: str, value: str, delta: str = None, color: str = "primary"):
    """
    Render a metric box.
    """
    st.metric(label=label, value=value, delta=delta)
