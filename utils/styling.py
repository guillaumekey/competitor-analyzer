import streamlit as st
import pandas as pd


def apply_custom_css():
    """Applique un CSS personnalisé à l'application Streamlit."""
    st.markdown("""
    <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton>button {
            width: 100%;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
        }
        .custom-metric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .custom-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        div[data-testid="stDataFrameResizable"] {
            width: 100% !important;
            max-width: none !important;
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
    return f"{num:,.0f}".replace(',', ' ')