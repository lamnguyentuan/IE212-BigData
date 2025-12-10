import streamlit as st

def render_video_player(url: str):
    """
    Render video player if URL is valid.
    """
    if url:
        st.video(url)
    else:
        st.warning("Video not found or accessible.")
