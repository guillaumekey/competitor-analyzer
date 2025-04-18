import streamlit as st
import pandas as pd
from utils.file_utils import normalize_column_names


def analyze_common_keywords(all_dataframes, min_occurrences, client_df=None):
    """Analyse les mots clés communs entre les fichiers.

    Args:
        all_dataframes: Liste de DataFrames pandas à analyser (concurrents)
        min_occurrences: Nombre minimum d'occurrences pour qu'un mot-clé soit considéré
        client_df: DataFrame pandas du client (optionnel)

    Returns:
        DataFrame pandas contenant les mots-clés communs
    """
    # Afficher des informations de débogage
    st.write(f"Débogage: Nombre de DataFrames concurrents: {len(all_dataframes)}")

    # Vérifier si les DataFrames concurrents sont vides
    if len(all_dataframes) == 0:
        st.error("Aucun DataFrame concurrent n'a été fourni pour l'analyse des mots-clés communs.")
        return pd.DataFrame(
            columns=['Mot clé', 'Nombre de fichiers', 'Volume de recherche', 'Difficulté', 'Présent chez le client'])

    # Afficher des informations sur chaque DataFrame concurrent
    for i, df in enumerate(all_dataframes):
        if df is None:
            st.warning(f"DataFrame concurrent #{i + 1} est None.")
            continue
        st.write(f"DataFrame concurrent #{i + 1}: {df.shape[0]} lignes, colonnes: {', '.join(df.columns.tolist())}")

    # Informations sur le DataFrame client
    if client_df is not None:
        if client_df.empty:
            st.warning("DataFrame client est vide.")
        else:
            st.write(
                f"DataFrame client: {client_df.shape[0]} lignes, colonnes: {', '.join(client_df.columns.tolist())}")
    else:
        st.warning("DataFrame client est None.")

    # Normaliser tous les DataFrames
    normalized_dataframes = []
    for df in all_dataframes:
        if df is not None and not df.empty:
            normalized_df = normalize_column_names(df)
            normalized_dataframes.append(normalized_df)
        else:
            st.warning("Un DataFrame concurrent vide ou None a été ignoré.")

    normalized_client_df = normalize_column_names(client_df) if client_df is not None and not client_df.empty else None

    # Vérifier si nous avons encore des DataFrames après la normalisation
    if len(normalized_dataframes) == 0:
        st.error("Aucun DataFrame concurrent valide après normalisation.")
        return pd.DataFrame(
            columns=['Mot clé', 'Nombre de fichiers', 'Volume de recherche', 'Difficulté', 'Présent chez le client'])

    # Dictionnaire pour stocker les informations par mot clé
    keywords_data = {}

    # Parcourir chaque DataFrame concurrent
    for idx, df in enumerate(normalized_dataframes):
        # Obtenir la liste des mots-clés (même sans filtrer par position_type)
        if 'keyword' not in df.columns:
            st.warning(f"La colonne 'keyword' est manquante dans le DataFrame concurrent #{idx + 1}.")
            continue

        # Essayer d'abord de filtrer par position_type si disponible
        try:
            if 'position_type' in df.columns:
                organic_df = df[df['position_type'].str.contains('Organic', case=False, na=False)]
                if organic_df.empty:
                    st.warning(
                        f"Aucun mot-clé organique trouvé dans le DataFrame concurrent #{idx + 1}. Utilisation de tous les mots-clés.")
                    organic_df = df
            else:
                st.warning(
                    f"Colonne 'position_type' manquante dans le DataFrame concurrent #{idx + 1}. Utilisation de tous les mots-clés.")
                organic_df = df
        except Exception as e:
            st.error(
                f"Erreur lors du filtrage des mots-clés organiques dans le DataFrame concurrent #{idx + 1}: {str(e)}")
            organic_df = df

        # Collecter les mots-clés
        keywords = organic_df['keyword'].dropna().astype(str)
        st.write(f"DataFrame concurrent #{idx + 1}: {len(keywords)} mots-clés trouvés.")

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()  # Normalisation du mot-clé

            if keyword_lower not in keywords_data:
                keywords_data[keyword_lower] = {
                    'occurrences': 0,
                    'search_volumes': [],
                    'difficulties': [],
                    'in_client': False
                }

            # Incrémenter le compteur d'occurrences
            keywords_data[keyword_lower]['occurrences'] += 1

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
                        except Exception as e:
                            st.warning(f"Erreur lors de l'extraction de {col_name} pour '{keyword}': {str(e)}")
                        break

    # Afficher le nombre total de mots-clés collectés
    st.write(f"Nombre total de mots-clés collectés: {len(keywords_data)}")

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
                st.write(f"Méthode 1: {len(client_keywords)} mots-clés trouvés chez le client.")
            except Exception as e:
                st.error(f"Erreur lors de l'extraction des mots-clés organiques du client: {str(e)}")

        # Méthode 2: Si la première méthode ne donne pas de résultats, utiliser tous les mots-clés
        if not client_keywords and 'keyword' in normalized_client_df.columns:
            client_keywords = set(normalized_client_df['keyword'].dropna().astype(str).str.lower().str.strip())
            st.write(f"Méthode 2: {len(client_keywords)} mots-clés trouvés chez le client.")

        # Marquer les mots-clés présents chez le client
        for keyword in keywords_data:
            keywords_data[keyword]['in_client'] = keyword in client_keywords
    else:
        st.warning("Aucun DataFrame client valide pour vérifier la présence des mots-clés.")

    # Filtrer les mots clés selon le seuil minimum d'occurrences
    filtered_keywords = {
        k: v for k, v in keywords_data.items()
        if v['occurrences'] >= min_occurrences
    }

    st.write(f"Mots-clés avec au moins {min_occurrences} occurrences: {len(filtered_keywords)}")

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

    st.write(f"Nombre de résultats à afficher: {len(results)}")

    # Si aucun résultat, retourner un DataFrame vide mais avec les colonnes définies
    if not results:
        st.warning("Aucun mot-clé commun n'a été trouvé selon les critères spécifiés.")
        return pd.DataFrame(
            columns=['Mot clé', 'Nombre de fichiers', 'Volume de recherche', 'Difficulté', 'Présent chez le client'])

    # Trier par nombre d'occurrences décroissant
    return pd.DataFrame(results).sort_values('Nombre de fichiers', ascending=False)