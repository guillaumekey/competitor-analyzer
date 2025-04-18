import streamlit as st

def setup_page_config():
    """Configure la mise en page principale de l'application Streamlit."""
    st.set_page_config(
        page_title="Analyseur SEMrush",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )