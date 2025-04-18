import streamlit as st
import pandas as pd
from utils.file_utils import normalize_column_names


def analyze_common_keywords(all_dataframes, min_occurrences, client_df=None, debug_mode=False):
    """Analyse les mots clés communs entre les fichiers.

    Args:
        all_dataframes: Liste de DataFrames pandas à analyser (concurrents)
        min_occurrences: Nombre minimum d'occurrences pour qu'un mot-clé soit considéré
        client_df: DataFrame pandas du client (optionnel)
        debug_mode: Afficher des informations de débogage

    Returns:
        DataFrame pandas contenant les mots-clés communs
    """
    # Déboggage optionnel
    if debug_mode:
        st.write(f"Nombre de DataFrames concurrents: {len(all_dataframes)}")

    # Normaliser tous les DataFrames
    normalized_dataframes = []
    for df in all_dataframes:
        if df is not None and not df.empty:
            normalized_df = normalize_column_names(df)
            normalized_dataframes.append(normalized_df)

    normalized_client_df = normalize_column_names(client_df) if client_df is not None and not client_df.empty else None

    # Dictionnaire pour stocker les informations par mot clé et par fichier
    keywords_data = {}

    # Dictionnaire pour suivre dans quels fichiers chaque mot-clé apparaît
    keyword_files = {}

    # Parcourir chaque DataFrame concurrent
    for idx, df in enumerate(normalized_dataframes):
        # Obtenir la liste des mots-clés (même sans filtrer par position_type)
        if 'keyword' not in df.columns:
            continue

        # Essayer d'abord de filtrer par position_type si disponible
        try:
            if 'position_type' in df.columns:
                organic_df = df[df['position_type'].str.contains('Organic', case=False, na=False)]
                if organic_df.empty:
                    organic_df = df
            else:
                organic_df = df
        except Exception:
            organic_df = df

        # Collecter tous les mots-clés uniques dans ce fichier
        file_keywords = set()

        for keyword in organic_df['keyword'].dropna().astype(str):
            keyword_lower = keyword.lower().strip()  # Normalisation du mot-clé
            file_keywords.add(keyword_lower)

            # Initialiser les données du mot-clé s'il n'existe pas encore
            if keyword_lower not in keywords_data:
                keywords_data[keyword_lower] = {
                    'occurrences': 0,
                    'search_volumes': [],
                    'difficulties': [],
                    'in_client': False
                }
                keyword_files[keyword_lower] = set()

            # Ajouter l'ID du fichier actuel à l'ensemble des fichiers où ce mot-clé apparaît
            keyword_files[keyword_lower].add(idx)

            # Ajouter les métriques (volume et difficulté) si disponibles
            for col_key, col_names in [
                ('search_volume', ['search_volume', 'volume', 'volume de recherche']),
                ('difficulty', ['difficulty', 'kd', 'keyword_difficulty'])
            ]:
                for col_name in col_names:
                    if col_name in df.columns:
                        try:
                            value = df[df['keyword'] == keyword][col_name].mean()
                            if pd.notna(value):
                                if col_key == 'search_volume':
                                    keywords_data[keyword_lower]['search_volumes'].append(value)
                                else:
                                    keywords_data[keyword_lower]['difficulties'].append(value)
                        except Exception:
                            pass
                        break

    # Mettre à jour le nombre d'occurrences basé sur le nombre de fichiers distincts
    for keyword in keywords_data:
        keywords_data[keyword]['occurrences'] = len(keyword_files[keyword])

    # Vérifier la présence dans le fichier client
    if normalized_client_df is not None:
        client_keywords = set()

        # Méthode 1: Essayer d'extraire les mots-clés organiques si possible
        if 'keyword' in normalized_client_df.columns and 'position_type' in normalized_client_df.columns:
            try:
                client_organic_df = normalized_client_df[
                    normalized_client_df['position_type'].str.contains('Organic', case=False, na=False)
                ]
                client_keywords = set(client_organic_df['keyword'].dropna().astype(str).str.lower().str.strip())
            except Exception:
                pass

        # Méthode 2: Si la première méthode ne donne pas de résultats, utiliser tous les mots-clés
        if not client_keywords and 'keyword' in normalized_client_df.columns:
            client_keywords = set(normalized_client_df['keyword'].dropna().astype(str).str.lower().str.strip())

        # Marquer les mots-clés présents chez le client
        for keyword in keywords_data:
            keywords_data[keyword]['in_client'] = keyword in client_keywords

    # Filtrer les mots clés selon le seuil minimum d'occurrences
    filtered_keywords = {
        k: v for k, v in keywords_data.items()
        if v['occurrences'] >= min_occurrences
    }

    # Afficher les statistiques en mode débogage
    if debug_mode:
        st.write(f"Nombre total de mots-clés collectés: {len(keywords_data)}")
        st.write(f"Mots-clés avec au moins {min_occurrences} occurrences: {len(filtered_keywords)}")
        st.write(f"Nombre maximum de fichiers: {len(normalized_dataframes)}")

    # Créer le DataFrame de résultats
    results = []
    for keyword, data in filtered_keywords.items():
        avg_search_volume = round(sum(data['search_volumes']) / len(data['search_volumes'])) if data[
            'search_volumes'] else 0
        avg_difficulty = round(sum(data['difficulties']) / len(data['difficulties'])) if data['difficulties'] else 0

        results.append({
            'Mot clé': keyword,
            'Nombre de fichiers': data['occurrences'],
            'Volume de recherche': avg_search_volume,
            'Difficulté': avg_difficulty,
            'Présent chez le client': 'Oui' if data['in_client'] else 'Non'
        })

    # Si aucun résultat, retourner un DataFrame vide mais avec les colonnes définies
    if not results:
        return pd.DataFrame(
            columns=['Mot clé', 'Nombre de fichiers', 'Volume de recherche', 'Difficulté', 'Présent chez le client'])

    # Trier par nombre d'occurrences décroissant
    return pd.DataFrame(results).sort_values(['Nombre de fichiers', 'Volume de recherche'], ascending=[False, False])