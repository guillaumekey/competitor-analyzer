import streamlit as st
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict


def safe_round(value):
    """Convertit de manière sécurisée une valeur en entier, en gérant les NaN."""
    if pd.isna(value) or np.isnan(value):
        return 0
    try:
        return round(float(value))
    except (ValueError, TypeError):
        return 0


def analyze_common_keywords_with_progress(all_dataframes, min_occurrences, client_df=None, debug_mode=False):
    """Analyse optimisée des mots clés communs entre les fichiers avec barre de progression.

    Args:
        all_dataframes: Liste de DataFrames pandas à analyser (concurrents)
        min_occurrences: Nombre minimum d'occurrences pour qu'un mot-clé soit considéré
        client_df: DataFrame pandas du client (optionnel)
        debug_mode: Afficher des informations de débogage

    Returns:
        DataFrame pandas contenant les mots-clés communs
    """
    # Créer un placeholder pour la barre de progression
    progress_placeholder = st.empty()

    # Initialiser la barre de progression
    progress_bar = progress_placeholder.progress(0)

    # Afficher un message de statut
    status_placeholder = st.empty()
    status_placeholder.info("Démarrage de l'analyse des mots-clés communs...")

    # Fonction pour mettre à jour la progression
    def update_progress(step_name, progress_value):
        progress_bar.progress(progress_value)
        status_placeholder.info(f"En cours : {step_name} ({int(progress_value * 100)}%)")

    # 1. Normalisation des données - 10%
    update_progress("Normalisation des données", 0.1)

    # Optimisation: utilisation de sets pour les recherches rapides
    keywords_by_file = {}
    metrics_by_keyword = defaultdict(lambda: {'volumes': [], 'difficulties': []})

    # 2. Extraction des mots-clés - 30%
    total_files = len(all_dataframes)

    # Fonction d'extraction pour un seul DataFrame
    def extract_keywords_from_df(idx, df):
        if df is None or df.empty:
            return idx, set(), {}

        # Normaliser les noms de colonnes
        df.columns = [col.lower().strip() for col in df.columns]

        # Déterminer les noms des colonnes
        keyword_col = next((col for col in df.columns if col in ['keyword', 'mot clé', 'mot-clé']), None)
        volume_col = next((col for col in df.columns if col in ['search volume', 'volume', 'volume de recherche']),
                          None)
        difficulty_col = next(
            (col for col in df.columns if col in ['difficulty', 'difficulté', 'kd', 'keyword difficulty']), None)

        if keyword_col is None:
            return idx, set(), {}

        # Extraire les mots-clés uniques
        keywords = set(df[keyword_col].dropna().astype(str).str.lower().str.strip())

        # Extraire les métriques pour chaque mot-clé
        keyword_metrics = {}
        for keyword in keywords:
            metrics = {}
            if volume_col:
                rows = df[df[keyword_col].str.lower().str.strip() == keyword]
                if not rows.empty:
                    volume_value = rows[volume_col].mean()
                    if not pd.isna(volume_value):
                        metrics['volume'] = volume_value

            if difficulty_col:
                rows = df[df[keyword_col].str.lower().str.strip() == keyword]
                if not rows.empty:
                    difficulty_value = rows[difficulty_col].mean()
                    if not pd.isna(difficulty_value):
                        metrics['difficulty'] = difficulty_value

            keyword_metrics[keyword] = metrics

        # Simuler un traitement plus long pour voir la progression
        time.sleep(0.05)

        return idx, keywords, keyword_metrics

    # Traitement des DataFrames avec mise à jour de la progression
    for i, df in enumerate(all_dataframes):
        idx, keywords, metrics = extract_keywords_from_df(i, df)
        keywords_by_file[idx] = keywords

        # Collecter les métriques
        for keyword, metric_values in metrics.items():
            if 'volume' in metric_values and not pd.isna(metric_values['volume']):
                metrics_by_keyword[keyword]['volumes'].append(metric_values['volume'])
            if 'difficulty' in metric_values and not pd.isna(metric_values['difficulty']):
                metrics_by_keyword[keyword]['difficulties'].append(metric_values['difficulty'])

        # Mettre à jour la progression
        progress_value = 0.1 + (0.2 * (i + 1) / total_files)
        update_progress(f"Traitement du fichier {i + 1}/{total_files}", progress_value)

    # 3. Comptage des occurrences - 50%
    update_progress("Analyse des occurrences", 0.5)

    all_keywords = set()
    for keywords in keywords_by_file.values():
        all_keywords.update(keywords)

    # Compter les occurrences de chaque mot-clé
    keyword_counts = {}
    keyword_count = len(all_keywords)
    for i, keyword in enumerate(all_keywords):
        count = sum(1 for keywords in keywords_by_file.values() if keyword in keywords)
        if count >= min_occurrences:
            keyword_counts[keyword] = count

        # Mise à jour de la progression pour cette étape
        if i % 100 == 0 or i == keyword_count - 1:  # Mise à jour tous les 100 mots-clés
            progress_value = 0.5 + (0.2 * (i + 1) / keyword_count)
            update_progress("Analyse des occurrences", min(0.7, progress_value))

    # 4. Vérification de la présence chez le client - 70%
    update_progress("Vérification de la présence chez le client", 0.7)

    client_keywords = set()
    if client_df is not None and not client_df.empty:
        # Normaliser les noms de colonnes
        client_df.columns = [col.lower().strip() for col in client_df.columns]

        # Trouver la colonne contenant les mots-clés
        keyword_col = next((col for col in client_df.columns if col in ['keyword', 'mot clé', 'mot-clé']), None)

        if keyword_col:
            client_keywords = set(client_df[keyword_col].dropna().astype(str).str.lower().str.strip())

    # 5. Création du DataFrame résultat - 80%
    update_progress("Création du tableau de résultats", 0.8)

    results = []
    result_count = len(keyword_counts)
    for i, (keyword, count) in enumerate(keyword_counts.items()):
        # Calcul des moyennes avec gestion sécurisée des NaN
        avg_volume = 0
        if metrics_by_keyword[keyword]['volumes']:
            try:
                # Filtrer les valeurs non-numériques ou NaN
                valid_volumes = [v for v in metrics_by_keyword[keyword]['volumes']
                                 if pd.notna(v) and isinstance(v, (int, float))]
                if valid_volumes:
                    avg_volume = safe_round(sum(valid_volumes) / len(valid_volumes))
            except (TypeError, ValueError):
                avg_volume = 0

        avg_difficulty = 0
        if metrics_by_keyword[keyword]['difficulties']:
            try:
                # Filtrer les valeurs non-numériques ou NaN
                valid_difficulties = [d for d in metrics_by_keyword[keyword]['difficulties']
                                      if pd.notna(d) and isinstance(d, (int, float))]
                if valid_difficulties:
                    avg_difficulty = safe_round(sum(valid_difficulties) / len(valid_difficulties))
            except (TypeError, ValueError):
                avg_difficulty = 0

        results.append({
            'Mot clé': keyword,
            'Nombre de fichiers': int(count),  # Assurer que c'est un entier
            'Volume de recherche': int(avg_volume),  # Assurer que c'est un entier
            'Difficulté': int(avg_difficulty),  # Assurer que c'est un entier
            'Présent chez le client': 'Oui' if keyword in client_keywords else 'Non'
        })

        # Mise à jour de la progression pour cette étape
        if i % 100 == 0 or i == result_count - 1:  # Mise à jour tous les 100 mots-clés
            progress_value = 0.8 + (0.15 * (i + 1) / result_count)
            update_progress("Création du tableau de résultats", min(0.95, progress_value))

    # 6. Tri et finalisation - 95%
    update_progress("Finalisation des résultats", 0.95)

    # Trier par nombre d'occurrences et volume
    results_df = pd.DataFrame(results).sort_values(['Nombre de fichiers', 'Volume de recherche'],
                                                   ascending=[False, False])

    # 7. Terminé - 100%
    update_progress("Analyse terminée", 1.0)

    # Supprimer la barre de progression et le message après un court délai
    time.sleep(0.5)
    progress_placeholder.empty()
    status_placeholder.success(f"Analyse terminée ! {len(results_df)} mots-clés trouvés.")

    return results_df


# Pour compatibilité avec l'interface originale
def analyze_common_keywords(all_dataframes, min_occurrences, client_df=None, debug_mode=False):
    """
    Version de compatibilité qui redirige vers la version avec barre de progression.
    """
    return analyze_common_keywords_with_progress(all_dataframes, min_occurrences, client_df, debug_mode)