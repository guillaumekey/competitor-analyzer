import streamlit as st
import pandas as pd


def apply_custom_css():
    """Applique un CSS personnalisé minimaliste à l'application Streamlit."""
    st.markdown("""
    <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton>button {
            width: 100%;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
        }
    </style>
    """, unsafe_allow_html=True)


def format_number(num):
    """Formate un nombre avec séparateur de milliers.

    Args:
        num: Nombre à formater

    Returns:
        Chaîne formatée
    """
    if pd.isna(num):
        return "0"
    try:
        return f"{int(num):,}".replace(',', ' ')
    except (ValueError, TypeError):
        return "0"