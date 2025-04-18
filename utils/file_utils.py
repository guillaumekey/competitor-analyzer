import streamlit as st
import pandas as pd
import chardet


def detect_encoding(file):
    """Détecte l'encodage du fichier."""
    raw_data = file.read(10000)
    file.seek(0)
    result = chardet.detect(raw_data)
    return result['encoding']


def read_csv_safely(file):
    """Lit un fichier CSV avec gestion du séparateur point-virgule.

    Args:
        file: Fichier uploadé via Streamlit

    Returns:
        DataFrame pandas ou None en cas d'erreur
    """
    try:
        return pd.read_csv(file, sep=';', encoding='utf-8')
    except Exception as first_error:
        try:
            file.seek(0)
            encoding = detect_encoding(file)
            file.seek(0)
            return pd.read_csv(file, sep=';', encoding=encoding or 'utf-8')
        except Exception as second_error:
            st.error(f"Erreur lors de la lecture du fichier: {str(second_error)}")
            return None


def normalize_column_names(df):
    """Normalise les noms de colonnes du DataFrame.

    Args:
        df: DataFrame pandas à normaliser

    Returns:
        DataFrame avec noms de colonnes normalisés
    """
    if df is None:
        return None

    # Mapping de noms de colonnes possibles
    column_mapping = {
        'keyword': ['Keyword', 'Mot clé', 'Mot-clé', 'keyword'],
        'position': ['Position', 'Pos', 'position'],
        'traffic': ['Traffic', 'Trafic', 'traffic'],
        'url': ['URL', 'Url', 'url'],
        'position_type': ['Position Type', 'Type de position', 'type'],
        'search_volume': ['Search Volume', 'Volume de recherche', 'Volume'],
        'difficulty': ['Keyword Difficulty', 'Difficulté', 'KD', 'Difficulty']
    }

    # Normaliser les noms de colonnes
    normalized_df = df.copy()
    normalized_df.columns = [col.strip() for col in normalized_df.columns]

    # Rechercher et renommer les colonnes basées sur les mappings
    for standard_name, possible_names in column_mapping.items():
        for col in normalized_df.columns:
            if col in possible_names:
                normalized_df = normalized_df.rename(columns={col: standard_name})
                break

    return normalized_df