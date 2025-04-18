import streamlit as st
import uuid
import pandas as pd
from utils.file_utils import read_csv_safely
from utils.styling import apply_custom_css
from analysis.semrush_analyzer import analyze_semrush_data, style_dataframe
from analysis.keyword_analyzer import analyze_common_keywords
from ui.components import render_metrics, render_instructions
from ui.layout import setup_page_config

# Configuration initiale
setup_page_config()
apply_custom_css()


def main():
    st.title("📊 Analyseur de données SEMrush")

    # Instructions
    with st.expander("ℹ️ Instructions", expanded=False):
        render_instructions()

    # Configuration des mots-clés communs
    min_occurrences = st.number_input(
        "Nombre minimum d'occurrences pour les mots clés communs",
        min_value=2,
        value=3,
        help="Un mot clé doit apparaître dans au moins ce nombre de fichiers pour être affiché dans le tableau des mots clés communs"
    )

    # Mode débogage
    debug_mode = st.sidebar.checkbox("Mode débogage", value=False)

    # Initialisation de la session
    if 'competitors' not in st.session_state:
        st.session_state.competitors = []

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
                        if debug_mode:
                            st.write(f"Fichier chargé: {file.name}, {df.shape[0]} lignes, {df.shape[1]} colonnes")
                    else:
                        df = competitor['dataframe']
                        if debug_mode:
                            st.write(f"DataFrame récupéré du cache: {df.shape[0]} lignes, {df.shape[1]} colonnes")

                    if df is not None:
                        # Stocker le DataFrame pour l'analyse des mots-clés communs
                        if is_client:
                            client_df = df.copy()
                            if debug_mode:
                                st.write(f"Client DataFrame défini: {df.shape[0]} lignes")
                        else:
                            all_dataframes.append(df.copy())
                            if debug_mode:
                                st.write(f"DataFrame concurrent ajouté, total: {len(all_dataframes)}")

                        # Analyser les données SEMrush
                        analysis = analyze_semrush_data(df, file.name, regex, is_client)
                        if analysis:
                            results.append(analysis)
                            if debug_mode:
                                st.write(f"Analyse réussie pour {file.name}")
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

        # Affichage du tableau comparatif
        st.markdown("### 📊 Tableau comparatif des performances")
        st.dataframe(
            style_dataframe(df_results),
            hide_index=True,
            use_container_width=True,
            height=400
        )

        # Analyse des mots-clés communs
        st.markdown("### 🔍 Analyse des mots clés communs")

        # Débogage des dataframes pour l'analyse des mots-clés
        if debug_mode:
            st.write(f"Nombre de concurrents: {len(st.session_state.competitors)}")
            st.write(f"Nombre de DataFrames concurrents: {len(all_dataframes)}")
            st.write(f"Client DataFrame présent: {client_df is not None}")

            # Afficher les premières lignes de chaque DataFrame pour débogage
            if len(all_dataframes) > 0:
                with st.expander("Aperçu des DataFrames concurrents"):
                    for i, df in enumerate(all_dataframes):
                        st.write(f"DataFrame concurrent #{i + 1} - Premières lignes:")
                        st.dataframe(df.head(3))

            if client_df is not None:
                with st.expander("Aperçu du DataFrame client"):
                    st.dataframe(client_df.head(3))

        # Génération de l'analyse des mots-clés communs
        if all_dataframes:
            try:
                with st.spinner("Analyse des mots-clés communs en cours..."):
                    common_keywords_df = analyze_common_keywords(all_dataframes, min_occurrences, client_df)

                if not common_keywords_df.empty:
                    # Mise en forme du tableau des mots-clés communs
                    def style_presence(val):
                        if val == 'Oui':
                            return 'background-color: rgba(46, 204, 113, 0.1)'
                        return ''

                    styled_df = common_keywords_df.style.applymap(
                        style_presence,
                        subset=['Présent chez le client']
                    )

                    st.dataframe(
                        styled_df,
                        hide_index=True,
                        use_container_width=True,
                        height=400
                    )

                    # Statistiques sur les mots-clés communs
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Total mots clés communs",
                            len(common_keywords_df)
                        )
                    with col2:
                        present_in_client = (common_keywords_df['Présent chez le client'] == 'Oui').sum()
                        percentage = round(present_in_client / len(common_keywords_df) * 100) if len(
                            common_keywords_df) > 0 else 0
                        st.metric(
                            "Mots clés présents chez le client",
                            f"{present_in_client} ({percentage}%)"
                        )
                else:
                    st.warning("Aucun mot-clé commun trouvé avec les critères spécifiés.")
                    st.write(
                        "Essayez de réduire le nombre minimum d'occurrences ou vérifiez les formats de vos fichiers.")
            except Exception as e:
                st.error(f"Erreur lors de l'analyse des mots clés communs: {str(e)}")
                st.write("Détails:", e)
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("Ajoutez au moins un fichier concurrent pour analyser les mots-clés communs.")


if __name__ == '__main__':
    main()