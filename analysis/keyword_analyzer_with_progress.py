import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


class AnalysisLogger:
    """Classe pour gérer les logs d'analyse avec horodatage."""

    def __init__(self, debug_mode=False):
        """Initialise le logger.

        Args:
            debug_mode: Si True, affiche plus de détails
        """
        self.logs = []
        self.debug_mode = debug_mode
        self.start_time = time.time()

        # S'assurer que analysis_logs existe dans la session
        if 'analysis_logs' not in st.session_state:
            st.session_state.analysis_logs = []

    def log(self, message, level="INFO"):
        """Ajoute un message au log avec horodatage.

        Args:
            message: Message à logger
            level: Niveau de log (INFO, DEBUG, WARNING, ERROR)
        """
        elapsed = time.time() - self.start_time
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{level}] [{elapsed:.2f}s] {message}"
        self.logs.append(entry)

        # Ajouter directement à la session state
        if level != "DEBUG" or self.debug_mode:
            st.session_state.analysis_logs.append(entry)

        # Pour le mode debug, afficher immédiatement le message
        if self.debug_mode and (level != "DEBUG" or self.debug_mode):
            print(entry)

    def debug(self, message):
        """Ajoute un message de debug."""
        if self.debug_mode:
            self.log(message, "DEBUG")

    def info(self, message):
        """Ajoute un message d'information."""
        self.log(message, "INFO")

    def warning(self, message):
        """Ajoute un avertissement."""
        self.log(message, "WARNING")

    def error(self, message):
        """Ajoute une erreur."""
        self.log(message, "ERROR")

    def get_logs(self):
        """Retourne tous les logs."""
        return "\n".join(self.logs)


def safe_round(value):
    """Convertit de manière sécurisée une valeur en entier, en gérant les NaN."""
    if pd.isna(value) or (isinstance(value, (float, np.float64)) and np.isnan(value)):
        return 0
    try:
        return round(float(value))
    except (ValueError, TypeError):
        return 0


def analyze_common_keywords_with_progress(all_dataframes, min_occurrences, client_df=None, debug_mode=False):
    """Analyse optimisée des mots clés communs entre les fichiers avec suivi détaillé.

    Args:
        all_dataframes: Liste de DataFrames pandas à analyser (concurrents)
        min_occurrences: Nombre minimum d'occurrences pour qu'un mot-clé soit considéré
        client_df: DataFrame pandas du client (optionnel)
        debug_mode: Afficher des informations de débogage

    Returns:
        DataFrame pandas contenant les mots-clés communs
    """
    # Réinitialiser les logs dans la session state
    st.session_state.analysis_logs = []

    # Initialisation du logger
    logger = AnalysisLogger(debug_mode)
    logger.info(f"Démarrage de l'analyse avec {len(all_dataframes)} fichiers concurrents")

    # Configuration du suivi sur l'interface Streamlit
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    status_placeholder = st.empty()
    status_placeholder.info("Démarrage de l'analyse des mots-clés communs...")

    # Fonction pour mettre à jour la progression et les logs
    def update_progress(step_name, progress_value, force_update=False):
        # Limiter les mises à jour trop fréquentes pour améliorer les performances
        current_time = time.time()
        last_update = getattr(update_progress, 'last_update', 0)

        if force_update or (current_time - last_update) > 0.2:  # Limite à une mise à jour toutes les 0.2 secondes
            progress_bar.progress(progress_value)
            status_message = f"En cours : {step_name} ({int(progress_value * 100)}%)"
            status_placeholder.info(status_message)
            logger.info(status_message)
            update_progress.last_update = current_time

    try:
        # 1. Normalisation des données - 10%
        update_progress("Initialisation de l'analyse", 0.05, True)
        logger.info("Normalisation des noms de colonnes et préparation des données")

        # Optimisation: utilisation de sets pour les recherches rapides
        keywords_by_file = {}
        metrics_by_keyword = defaultdict(lambda: {'volumes': [], 'difficulties': []})

        # 2. Extraction des mots-clés - 30%
        logger.info(f"Début de l'extraction des mots-clés de {len(all_dataframes)} fichiers")
        update_progress("Préparation de l'extraction des mots-clés", 0.1, True)

        # Fonction d'extraction optimisée
        def extract_keywords_from_df(idx, df):
            file_logger = {}  # Pour collecter les statistiques par fichier

            if df is None or df.empty:
                file_logger["status"] = "échec"
                file_logger["error"] = "DataFrame vide ou None"
                return idx, set(), {}, file_logger

            # Normaliser les noms de colonnes
            df.columns = [col.lower().strip() for col in df.columns]

            # Détecter les colonnes d'intérêt
            keyword_col = next((col for col in df.columns if col in ['keyword', 'mot clé', 'mot-clé']), None)
            volume_col = next((col for col in df.columns if col in ['search volume', 'volume', 'volume de recherche']),
                              None)
            difficulty_col = next(
                (col for col in df.columns if col in ['difficulty', 'difficulté', 'kd', 'keyword difficulty']), None)

            file_logger["colonnes"] = {
                "keyword_col": keyword_col,
                "volume_col": volume_col,
                "difficulty_col": difficulty_col
            }

            if keyword_col is None:
                file_logger["status"] = "échec"
                file_logger["error"] = "Colonne de mots-clés non trouvée"
                return idx, set(), {}, file_logger

            # Extraire les mots-clés uniques
            keywords = set()
            try:
                keywords = set(df[keyword_col].dropna().astype(str).str.lower().str.strip())
                file_logger["nb_keywords"] = len(keywords)
            except Exception as e:
                file_logger["status"] = "échec partiel"
                file_logger["error"] = f"Erreur lors de l'extraction des mots-clés: {str(e)}"

            # Extraire les métriques pour chaque mot-clé
            keyword_metrics = {}
            metrics_errors = 0

            for keyword in keywords:
                metrics = {}
                try:
                    if volume_col:
                        rows = df[df[keyword_col].str.lower().str.strip() == keyword]
                        if not rows.empty:
                            volume_value = pd.to_numeric(rows[volume_col], errors='coerce').mean()
                            if not pd.isna(volume_value):
                                metrics['volume'] = volume_value

                    if difficulty_col:
                        rows = df[df[keyword_col].str.lower().str.strip() == keyword]
                        if not rows.empty:
                            difficulty_value = pd.to_numeric(rows[difficulty_col], errors='coerce').mean()
                            if not pd.isna(difficulty_value):
                                metrics['difficulty'] = difficulty_value
                except Exception:
                    metrics_errors += 1

                keyword_metrics[keyword] = metrics

            file_logger["status"] = "succès" if metrics_errors == 0 else "succès partiel"
            if metrics_errors > 0:
                file_logger["metrics_errors"] = metrics_errors

            return idx, keywords, keyword_metrics, file_logger

        # Extraction parallèle des mots-clés avec suivi amélioré
        total_files = len(all_dataframes)
        files_processed = 0

        # Traiter les fichiers en parallèle
        with ThreadPoolExecutor(max_workers=min(4, total_files)) as executor:
            futures = [executor.submit(extract_keywords_from_df, i, df) for i, df in enumerate(all_dataframes)]

            for future in as_completed(futures):
                try:
                    idx, keywords, metrics, file_logger = future.result()

                    # Collecter les résultats
                    keywords_by_file[idx] = keywords
                    for keyword, metric_values in metrics.items():
                        if 'volume' in metric_values and not pd.isna(metric_values['volume']):
                            metrics_by_keyword[keyword]['volumes'].append(metric_values['volume'])
                        if 'difficulty' in metric_values and not pd.isna(metric_values['difficulty']):
                            metrics_by_keyword[keyword]['difficulties'].append(metric_values['difficulty'])

                    # Mettre à jour les logs
                    if file_logger.get("status") == "succès":
                        logger.info(f"Fichier {idx + 1}: {len(keywords)} mots-clés extraits avec succès")
                    elif file_logger.get("status") == "succès partiel":
                        logger.warning(
                            f"Fichier {idx + 1}: {len(keywords)} mots-clés extraits, {file_logger.get('metrics_errors', 0)} erreurs de métriques")
                    else:
                        logger.error(f"Fichier {idx + 1}: {file_logger.get('error', 'Erreur inconnue')}")

                    # Mettre à jour la progression
                    files_processed += 1
                    progress_value = 0.1 + (0.2 * files_processed / total_files)
                    update_progress(f"Extraction des mots-clés ({files_processed}/{total_files})",
                                    min(0.3, progress_value))

                except Exception as e:
                    logger.error(f"Erreur lors du traitement d'un fichier: {str(e)}")
                    files_processed += 1

        # Résumé de l'extraction
        logger.info(
            f"Extraction terminée: {sum(len(kw) for kw in keywords_by_file.values())} mots-clés au total dans {len(keywords_by_file)} fichiers")

        # 3. Comptage des occurrences - 50%
        update_progress("Analyse des occurrences et dédoublonnage", 0.3, True)

        all_keywords = set()
        for keywords in keywords_by_file.values():
            all_keywords.update(keywords)

        # Compter les occurrences de chaque mot-clé avec suivi détaillé
        keyword_counts = {}
        keyword_count = len(all_keywords)
        keywords_processed = 0
        last_log_time = time.time()

        logger.info(f"Début de l'analyse des occurrences pour {keyword_count} mots-clés uniques")

        for keyword in all_keywords:
            count = sum(1 for keywords in keywords_by_file.values() if keyword in keywords)
            if count >= min_occurrences:
                keyword_counts[keyword] = count

            keywords_processed += 1

            # Mises à jour périodiques pour éviter de saturer l'interface
            current_time = time.time()
            if keywords_processed % 500 == 0 or (current_time - last_log_time) > 2:
                progress_value = 0.3 + (0.2 * keywords_processed / keyword_count)
                update_progress(f"Analyse des occurrences ({keywords_processed}/{keyword_count})",
                                min(0.5, progress_value))
                last_log_time = current_time

                if keywords_processed % 5000 == 0:
                    logger.info(f"Progression de l'analyse: {keywords_processed}/{keyword_count} mots-clés traités")

        filtered_count = len(keyword_counts)
        logger.info(
            f"Analyse des occurrences terminée: {filtered_count} mots-clés avec au moins {min_occurrences} occurrences")

        # 4. Vérification de la présence chez le client - 70%
        update_progress("Vérification de la présence chez le client", 0.5, True)

        client_keywords = set()
        if client_df is not None and not client_df.empty:
            logger.info("Détection des mots-clés présents chez le client")

            # Normaliser les noms de colonnes
            client_df.columns = [col.lower().strip() for col in client_df.columns]

            # Trouver la colonne contenant les mots-clés
            keyword_col = next((col for col in client_df.columns if col in ['keyword', 'mot clé', 'mot-clé']), None)

            if keyword_col:
                try:
                    client_keywords = set(client_df[keyword_col].dropna().astype(str).str.lower().str.strip())
                    logger.info(f"Détecté {len(client_keywords)} mots-clés uniques chez le client")
                except Exception as e:
                    logger.error(f"Erreur lors de l'extraction des mots-clés client: {str(e)}")
            else:
                logger.warning("Colonne de mots-clés non trouvée dans le fichier client")
        else:
            logger.info("Aucun fichier client fourni")

        # 5. Création du DataFrame résultat - 80%
        update_progress("Création du tableau de résultats", 0.7, True)
        logger.info(f"Construction du tableau final pour {len(keyword_counts)} mots-clés")

        results = []
        results_processed = 0
        result_count = len(keyword_counts)
        last_log_time = time.time()
        metrics_errors = 0

        for keyword, count in keyword_counts.items():
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
                    metrics_errors += 1
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
                    metrics_errors += 1
                    avg_difficulty = 0

            results.append({
                'Mot clé': keyword,
                'Nombre de fichiers': int(count),  # Assurer que c'est un entier
                'Volume de recherche': int(avg_volume),  # Assurer que c'est un entier
                'Difficulté': int(avg_difficulty),  # Assurer que c'est un entier
                'Présent chez le client': 'Oui' if keyword in client_keywords else 'Non'
            })

            results_processed += 1

            # Mises à jour périodiques pour éviter de saturer l'interface
            current_time = time.time()
            if results_processed % 500 == 0 or (current_time - last_log_time) > 2:
                progress_value = 0.7 + (0.25 * results_processed / result_count)
                update_progress(f"Construction du tableau ({results_processed}/{result_count})",
                                min(0.95, progress_value))
                last_log_time = current_time

                if results_processed % 2000 == 0:
                    logger.info(f"Progression de la construction: {results_processed}/{result_count} entrées traitées")

        if metrics_errors > 0:
            logger.warning(f"{metrics_errors} erreurs lors du calcul des métriques (volume/difficulté)")

        # 6. Tri et finalisation - 95%
        update_progress("Finalisation des résultats", 0.95, True)

        # Trier par nombre d'occurrences et volume
        results_df = pd.DataFrame(results).sort_values(['Nombre de fichiers', 'Volume de recherche'],
                                                       ascending=[False, False])

        logger.info(f"Analyse terminée avec succès: {len(results_df)} mots-clés dans le tableau final")

        # 7. Terminé - 100%
        update_progress("Analyse terminée", 1.0, True)

        # Supprimer la barre de progression et le message après un court délai
        time.sleep(0.5)
        progress_placeholder.empty()
        status_placeholder.success(f"Analyse terminée ! {len(results_df)} mots-clés trouvés.")

        return results_df

    except Exception as e:
        # En cas d'erreur, enregistrer et afficher
        logger.error(f"Erreur lors de l'analyse: {str(e)}")
        progress_placeholder.empty()
        status_placeholder.error(f"Erreur lors de l'analyse: {str(e)}")

        # Retourner un DataFrame vide
        return pd.DataFrame(columns=['Mot clé', 'Nombre de fichiers', 'Volume de recherche',
                                     'Difficulté', 'Présent chez le client'])


# Pour compatibilité avec l'interface originale
def analyze_common_keywords(all_dataframes, min_occurrences, client_df=None, debug_mode=False):
    """
    Version de compatibilité qui redirige vers la version avec barre de progression.
    """
    return analyze_common_keywords_with_progress(all_dataframes, min_occurrences, client_df, debug_mode)