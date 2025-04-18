import streamlit as st
import uuid
import pandas as pd
from utils.file_utils import read_csv_safely
from utils.styling import apply_custom_css, format_number
from analysis.semrush_analyzer import analyze_semrush_data, style_dataframe
# Importer la fonction correcte
from analysis.keyword_analyzer_with_progress import analyze_common_keywords
from ui.components import render_metrics, render_instructions, render_keyword_stats, render_opportunity_stats
from ui.layout import setup_page_config
from filtering_system import render_filter_ui, apply_filters

# Configuration initiale
setup_page_config()
apply_custom_css()


def main():
    st.title("📊 Analyseur de données SEMrush")

    # Instructions
    with st.expander("ℹ️ Instructions", expanded=False):
        render_instructions()

    # Mode débogage caché dans la sidebar
    debug_mode = st.sidebar.checkbox("Mode débogage", value=False, key="debug_mode")

    # Initialisation de la session
    if 'competitors' not in st.session_state:
        st.session_state.competitors = []

    # Initialiser les logs d'analyse s'ils n'existent pas
    if 'analysis_logs' not in st.session_state:
        st.session_state.analysis_logs = []

    # Bouton d'ajout de concurrent
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("➕ Ajouter un fichier concurrent", key="add_competitor"):
            st.session_state.competitors.append({
                'id': str(uuid.uuid4()),
                'file': None,
                'regex': "",
                'is_client': False,
                'dataframe': None
            })

    # Traitement des concurrents
    results = []
    competitors_to_remove = []
    all_dataframes = []
    client_df = None

    # Interface pour chaque concurrent
    for idx, competitor in enumerate(st.session_state.competitors):
        with st.container():
            col1, col2, col3, col4 = st.columns([6, 3, 1, 0.5])

            with col1:
                file = st.file_uploader(
                    f"Concurrent {idx + 1}",
                    type='csv',
                    key=f"file_{competitor['id']}"
                )
                competitor['file'] = file

            with col2:
                regex = st.text_input(
                    "Regex",
                    value=competitor.get('regex', ''),
                    key=f"regex_{competitor['id']}",
                    placeholder="Ex: brand|nom"
                )
                competitor['regex'] = regex

            with col3:
                is_client = st.checkbox(
                    "Client",
                    value=competitor.get('is_client', False),
                    key=f"client_{competitor['id']}"
                )
                competitor['is_client'] = is_client

            with col4:
                if st.button("🗑️", key=f"remove_{competitor['id']}", help="Supprimer ce concurrent"):
                    competitors_to_remove.append(idx)

            # Traitement du fichier
            if file:
                try:
                    # Obtenir ou charger le DataFrame
                    if competitor.get('dataframe') is None:
                        df = read_csv_safely(file)
                        competitor['dataframe'] = df
                    else:
                        df = competitor['dataframe']

                    if df is not None:
                        # Stocker le DataFrame pour l'analyse des mots-clés communs
                        if is_client:
                            client_df = df.copy()
                        else:
                            all_dataframes.append(df.copy())

                        # Analyser les données SEMrush
                        analysis = analyze_semrush_data(df, file.name, regex, is_client)
                        if analysis:
                            results.append(analysis)
                except Exception as e:
                    st.error(f"Erreur lors du traitement de {file.name}: {str(e)}")

    # Supprimer les concurrents à retirer
    for idx in reversed(competitors_to_remove):
        st.session_state.competitors.pop(idx)

    # Affichage des résultats si disponibles
    if results:
        st.markdown("---")

        # Métriques globales
        st.markdown("### 📈 Métriques globales")

        # Préparation des données pour l'affichage
        column_names = {
            'domain': 'Domaine',
            'is_client': 'is_client',
            'total_traffic': 'Traffic Total',
            'total_keywords': 'Nombre de mots clés',
            'top3_keywords': 'Mots clés Top 3 (Organic)',
            'unique_urls': 'URLs uniques',
            'brand_traffic': 'Traffic de marque'
        }

        df_results = pd.DataFrame(results)
        main_columns = list(column_names.keys())
        df_results = df_results[main_columns].rename(columns=column_names)

        # Marquer le client
        df_results['Domaine'] = df_results.apply(
            lambda x: f"🏠 {x['Domaine']}" if x['is_client'] else x['Domaine'],
            axis=1
        )

        # Supprimer la colonne is_client
        df_results = df_results.drop('is_client', axis=1)
        df_results = df_results.sort_values('Traffic Total', ascending=False)

        # Formatage des valeurs numériques
        df_results['Traffic Total'] = pd.to_numeric(df_results['Traffic Total']).astype(int)
        df_results['Traffic de marque'] = pd.to_numeric(df_results['Traffic de marque']).astype(int)

        # Calcul des moyennes et affichage des métriques
        client_data = df_results[df_results['Domaine'].str.startswith('🏠 ')]
        competitors_data = df_results[~df_results['Domaine'].str.startswith('🏠 ')]

        # Calcul des moyennes
        avg_competitors_urls = competitors_data['URLs uniques'].mean()
        avg_competitors_keywords = competitors_data['Nombre de mots clés'].mean()
        avg_competitors_traffic = competitors_data['Traffic Total'].mean()

        # Valeurs du client
        client_urls = client_data['URLs uniques'].iloc[0] if not client_data.empty else 0
        client_keywords = client_data['Nombre de mots clés'].iloc[0] if not client_data.empty else 0
        client_traffic = client_data['Traffic Total'].iloc[0] if not client_data.empty else 0

        # Calcul des différences
        def calc_difference(client_val, avg_val):
            if avg_val == 0:
                return 0
            return ((client_val - avg_val) / avg_val) * 100

        urls_diff = calc_difference(client_urls, avg_competitors_urls)
        keywords_diff = calc_difference(client_keywords, avg_competitors_keywords)
        traffic_diff = calc_difference(client_traffic, avg_competitors_traffic)

        # Affichage des métriques
        render_metrics(
            avg_competitors_urls, client_urls, urls_diff,
            avg_competitors_keywords, client_keywords, keywords_diff,
            avg_competitors_traffic, client_traffic, traffic_diff
        )

        # Affichage du tableau comparatif avec style amélioré
        st.markdown("### 📊 Tableau comparatif des performances")

        # Appliquer le style pandas compatible avec Streamlit
        styled_df = style_dataframe(df_results)

        # Afficher le tableau avec le style amélioré
        st.dataframe(
            styled_df,
            hide_index=True,
            use_container_width=True,
            height=400
        )

        # Analyse des mots-clés communs
        st.markdown("### 🔍 Analyse des mots clés communs")

        # Créer des onglets pour l'analyse et les logs
        analysis_tab, logs_tab = st.tabs(["Résultats d'analyse", "Logs de traitement"])

        with analysis_tab:
            # Configuration pour l'analyse des mots-clés communs
            min_occurrences = st.number_input(
                "Nombre minimum d'occurrences pour les mots clés communs",
                min_value=2,
                value=3,
                help="Un mot clé doit apparaître dans au moins ce nombre de fichiers pour être affiché dans le tableau des mots clés communs"
            )

            # Génération de l'analyse des mots-clés communs
            if all_dataframes:
                try:
                    # Calculer les mots-clés communs
                    if 'common_keywords_df' not in st.session_state or st.session_state.get('recalculate_keywords',
                                                                                            True):
                        common_keywords_df = analyze_common_keywords(all_dataframes, min_occurrences,
                                                                     client_df, debug_mode)
                        st.session_state.common_keywords_df = common_keywords_df
                        st.session_state.recalculate_keywords = False
                    else:
                        common_keywords_df = st.session_state.common_keywords_df

                    if not common_keywords_df.empty:
                        # Système de filtrage avancé avec UI améliorée
                        filter_config = render_filter_ui()

                        # Vérifier si les filtres ont été mis à jour
                        if filter_config.get("changed", True) and filter_config.get("rules"):
                            # Si oui, appliquer les filtres et mettre à jour le DataFrame
                            filtered_df = apply_filters(common_keywords_df, filter_config)

                            # Mettre le DataFrame filtré en cache
                            st.session_state.filtered_df = filtered_df
                        else:
                            # Sinon, utiliser les résultats filtrés précédemment ou le DataFrame complet
                            if 'filtered_df' in st.session_state:
                                filtered_df = st.session_state.filtered_df
                            else:
                                filtered_df = common_keywords_df

                        # Calculer les statistiques
                        total_keywords = len(filtered_df)
                        present_in_client = (filtered_df['Présent chez le client'] == 'Oui').sum()
                        percentage = round(present_in_client / total_keywords * 100) if total_keywords > 0 else 0

                        # Calculer le volume de recherche potentiel (mots-clés non présents chez le client)
                        potential_volume = filtered_df[filtered_df['Présent chez le client'] == 'Non'][
                            'Volume de recherche'].sum()

                        # Afficher les statistiques avec le composant amélioré
                        render_keyword_stats(total_keywords, present_in_client, percentage, potential_volume)

                        # Appliquer un style simple mais efficace au tableau des mots-clés
                        keywords_styler = filtered_df.style

                        # Formater les nombres
                        keywords_styler = keywords_styler.format({
                            'Volume de recherche': lambda x: f"{int(x):,}".replace(',', ' '),
                            'Difficulté': lambda x: f"{int(x):,}".replace(',', ' ')
                        })

                        # Mise en évidence simple des "Oui"
                        def highlight_presence(val):
                            if val == 'Oui':
                                return 'background-color: #e8f5e9; color: #2e7d32; font-weight: bold'
                            return ''

                        keywords_styler = keywords_styler.applymap(highlight_presence,
                                                                   subset=['Présent chez le client'])

                        # Style d'en-tête simple
                        keywords_styler = keywords_styler.set_table_styles([
                            {'selector': 'thead th',
                             'props': [('background-color', '#f5f5f5'),
                                       ('color', '#333333'),
                                       ('font-weight', 'bold'),
                                       ('text-align', 'center')]}
                        ])

                        # Afficher le DataFrame des mots-clés communs sans HTML complexe
                        st.dataframe(
                            keywords_styler,
                            hide_index=True,
                            use_container_width=True,
                            height=400
                        )

                        # Section pour les opportunités à faible KD
                        st.markdown("### 💡 Opportunités à faible difficulté")

                        max_kd = st.slider(
                            "Difficulté maximum (KD)",
                            min_value=0,
                            max_value=100,
                            value=30,
                            help="Affiche uniquement les mots-clés ayant une difficulté inférieure ou égale à cette valeur"
                        )

                        # Filtrer les mots-clés non présents chez le client avec KD inférieur au maximum
                        opportunities_df = filtered_df[
                            (filtered_df['Présent chez le client'] == 'Non') &
                            (filtered_df['Difficulté'] <= max_kd)
                            ].sort_values(['Difficulté', 'Volume de recherche'], ascending=[True, False])

                        if not opportunities_df.empty:
                            # Statistiques sur les opportunités avec le composant amélioré
                            opp_volume = opportunities_df['Volume de recherche'].sum()
                            render_opportunity_stats(len(opportunities_df), opp_volume)

                            # Styliser le tableau des opportunités simplement
                            opportunities_styler = opportunities_df.style.format({
                                'Volume de recherche': lambda x: f"{int(x):,}".replace(',', ' '),
                                'Difficulté': lambda x: f"{int(x):,}".replace(',', ' ')
                            })

                            # Coloration de la difficulté (vert quand faible)
                            def color_difficulty(val):
                                if val <= 10:
                                    return 'background-color: #e8f5e9; color: #2e7d32'
                                elif val <= 20:
                                    return 'background-color: #f1f8e9; color: #558b2f'
                                elif val <= 30:
                                    return 'background-color: #fffde7; color: #f57f17'
                                return ''

                            opportunities_styler = opportunities_styler.applymap(color_difficulty,
                                                                                 subset=['Difficulté'])

                            # Style d'en-tête simple
                            opportunities_styler = opportunities_styler.set_table_styles([
                                {'selector': 'thead th',
                                 'props': [('background-color', '#fff8e1'),
                                           ('color', '#333333'),
                                           ('font-weight', 'bold'),
                                           ('text-align', 'center')]}
                            ])

                            # Afficher le tableau des opportunités
                            st.dataframe(
                                opportunities_styler,
                                hide_index=True,
                                use_container_width=True,
                                height=400
                            )
                        else:
                            st.info("Aucune opportunité trouvée avec les critères spécifiés.")
                    else:
                        st.warning("Aucun mot-clé commun trouvé avec les critères spécifiés.")
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse des mots clés communs: {str(e)}")
                    if debug_mode:
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.warning("Ajoutez au moins un fichier concurrent pour analyser les mots-clés communs.")

        # Onglet des logs
        with logs_tab:
            st.markdown("### 📋 Logs détaillés de l'analyse")

            # Bouton pour effacer les logs
            if st.button("🗑️ Effacer les logs"):
                st.session_state.analysis_logs = []
                st.success("Logs effacés avec succès")

            # Vérifier si les logs existent
            if 'analysis_logs' not in st.session_state:
                st.session_state.analysis_logs = []

            # Afficher les logs existants s'il y en a
            if st.session_state.analysis_logs:
                # Limiter à 500 entrées pour éviter les problèmes de performance
                logs_to_display = st.session_state.analysis_logs[-500:] if len(
                    st.session_state.analysis_logs) > 500 else st.session_state.analysis_logs

                # Afficher les logs
                st.text_area("Logs d'analyse",
                             "\n".join(logs_to_display),
                             height=500,
                             key="logs_display")
            else:
                st.info("Aucun log d'analyse disponible. Lancez une analyse pour générer des logs.")


if __name__ == '__main__':
    main()