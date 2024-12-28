import streamlit as st
import pandas as pd
import chardet
from urllib.parse import urlparse
import re
import uuid

# Configuration de la page en full width
st.set_page_config(
    page_title="Analyseur SEMrush",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé
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


def detect_encoding(file):
    """Détecte l'encodage du fichier."""
    raw_data = file.read(10000)
    file.seek(0)
    result = chardet.detect(raw_data)
    return result['encoding']


def read_csv_safely(file):
    """Lit un fichier CSV avec gestion du séparateur point-virgule."""
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
            raise


def extract_domain_from_url(url):
    """Extrait le nom de domaine d'une URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return None


def analyze_semrush_data(df, filename, brand_regex, is_client=False):
    """Analyse un DataFrame de données SEMrush."""
    try:
        first_url = df['URL'].iloc[0] if not df['URL'].empty else None
        if first_url:
            domain = extract_domain_from_url(first_url)
            if not domain:
                st.error(f"Impossible d'extraire le domaine de l'URL: {first_url}")
                return None
        else:
            st.error("Aucune URL trouvée dans le fichier")
            return None

        df.columns = [col.strip() for col in df.columns]

        required_columns = {
            'Keyword': str,
            'Position': float,
            'Traffic': float,
            'URL': str,
            'Position Type': str
        }

        for col, dtype in required_columns.items():
            if col not in df.columns:
                similar_cols = [c for c in df.columns if col.lower() in c.lower()]
                if similar_cols:
                    df[col] = df[similar_cols[0]].astype(dtype)
                else:
                    st.error(f"Colonne manquante : {col}")
                    return None

        df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(999)
        df['Traffic'] = pd.to_numeric(df['Traffic'], errors='coerce').fillna(0)

        organic_df = df[df['Position Type'].str.contains('Organic', case=False, na=False)]

        try:
            df['is_brand_keyword'] = df['Keyword'].astype(str).str.lower().str.contains(
                brand_regex, case=False, regex=True, na=False
            )
        except Exception as regex_error:
            st.error(f"Erreur avec l'expression régulière '{brand_regex}': {str(regex_error)}")
            df['is_brand_keyword'] = False

        return {
            'domain': domain,
            'is_client': is_client,
            'total_traffic': round(df['Traffic'].sum()),
            'total_keywords': len(df),
            'top3_keywords': len(organic_df[organic_df['Position'].between(1, 3)]),
            'unique_urls': df['URL'].nunique(),
            'brand_traffic': round(df[df['is_brand_keyword']]['Traffic'].sum())
        }

    except Exception as e:
        st.error(f"Erreur lors de l'analyse de {filename}: {str(e)}")
        return None


def style_dataframe(df):
    """Applique un style personnalisé au DataFrame avec mise en forme conditionnelle."""
    df_style = df.copy()

    for col in ['Traffic Total', 'Traffic de marque']:
        df_style[col] = pd.to_numeric(df_style[col], errors='coerce')

    def color_scale(val, col):
        if pd.isna(val):
            return ''

        max_val = df_style[col].max()
        min_val = df_style[col].min()
        norm_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0

        return f'background-color: rgba(0, 99, 255, {norm_val * 0.2}); color: {"#1e1e1e" if norm_val < 0.7 else "white"}'

    def style_domain(val):
        if val.startswith('🏠 '):  # Changé de 👑 à 🏠
            return 'font-weight: bold; background-color: rgba(255, 215, 0, 0.1)'
        return ''

    styles = []
    for col in ['Traffic Total', 'Traffic de marque']:
        styles.append({
            col: lambda x, c=col: color_scale(x, c)
        })

    styled_df = df_style.style
    styled_df = styled_df.applymap(style_domain, subset=['Domaine'])

    for style in styles:
        for col, func in style.items():
            styled_df = styled_df.applymap(func, subset=[col])

    return styled_df


def analyze_common_keywords(all_dataframes, min_occurrences, client_df=None):
    """Analyse les mots clés communs entre les fichiers."""
    # Mapping des noms de colonnes possibles
    column_mapping = {
        'search_volume': ['Search Volume', 'Volume de recherche', 'Volume'],
        'difficulty': ['Keyword Difficulty', 'Difficulté', 'KD', 'Difficulty']
    }

    # Fonction pour trouver la bonne colonne
    def find_column(df, possible_names):
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    # Dictionnaire pour stocker les informations par mot clé
    keywords_data = {}

    # Parcourir chaque DataFrame
    for df in all_dataframes:
        # Identifier les colonnes nécessaires
        volume_col = find_column(df, column_mapping['search_volume'])
        difficulty_col = find_column(df, column_mapping['difficulty'])

        # Filtrer uniquement les mots clés organiques
        organic_df = df[df['Position Type'].str.contains('Organic', case=False, na=False)]

        # Grouper par mot clé pour n'avoir qu'une occurrence par fichier
        # et prendre les moyennes des métriques pour chaque mot clé
        grouped_df = organic_df.groupby('Keyword').agg({
            volume_col: 'mean' if volume_col else 'size',
            difficulty_col: 'mean' if difficulty_col else 'size'
        }).reset_index()

        for _, row in grouped_df.iterrows():
            keyword = row['Keyword'].lower()  # Normalisation en minuscules

            if keyword not in keywords_data:
                keywords_data[keyword] = {
                    'occurrences': 0,  # Maintenant c'est le nombre de fichiers où le mot clé apparaît
                    'search_volumes': [],
                    'difficulties': [],
                    'in_client': False
                }

            # Incrémenter le compteur d'occurrences (maintenant 1 par fichier)
            keywords_data[keyword]['occurrences'] += 1

            # Ajouter les métriques moyennes si les colonnes existent
            if volume_col and pd.notna(row[volume_col]):
                keywords_data[keyword]['search_volumes'].append(row[volume_col])
            if difficulty_col and pd.notna(row[difficulty_col]):
                keywords_data[keyword]['difficulties'].append(row[difficulty_col])

    # Vérifier la présence dans le fichier client si fourni
    if client_df is not None:
        # Filtrer aussi les mots clés organiques du client
        client_organic_df = client_df[client_df['Position Type'].str.contains('Organic', case=False, na=False)]
        client_keywords = set(client_organic_df['Keyword'].str.lower())
        for keyword in keywords_data:
            keywords_data[keyword]['in_client'] = keyword in client_keywords

    # Filtrer les mots clés selon le seuil minimum d'occurrences
    filtered_keywords = {
        k: v for k, v in keywords_data.items()
        if v['occurrences'] >= min_occurrences
    }

    # Créer le DataFrame de résultats
    results = []
    for keyword, data in filtered_keywords.items():
        avg_search_volume = round(sum(data['search_volumes']) / len(data['search_volumes'])) if data[
            'search_volumes'] else 0
        avg_difficulty = round(sum(data['difficulties']) / len(data['difficulties'])) if data['difficulties'] else 0

        results.append({
            'Mot clé': keyword,
            'Nombre de fichiers': data['occurrences'],  # Renommé pour plus de clarté
            'Volume de recherche': avg_search_volume,
            'Difficulté': avg_difficulty,
            'Présent chez le client': 'Oui' if data['in_client'] else 'Non'
        })

    return pd.DataFrame(results).sort_values('Nombre de fichiers', ascending=False)

def main():
    st.title("📊 Analyseur de données SEMrush")

    with st.expander("ℹ️ Instructions", expanded=False):
        st.markdown("""
        1. Cliquez sur le bouton "Ajouter un fichier concurrent" pour chaque site à analyser
        2. Pour chaque concurrent :
           - Uploadez le fichier d'export SEMrush (CSV)
           - Définissez l'expression régulière pour identifier les mots-clés de marque
           - Cochez la case "Client" s'il s'agit de votre site
        3. Le tableau comparatif se mettra à jour automatiquement

        **Note**: Les expressions régulières peuvent inclure plusieurs variantes (ex: `marque|brand|nom`)
        """)

    min_occurrences = st.number_input(
        "Nombre minimum d'occurrences pour les mots clés communs",
        min_value=2,
        value=3,
        help="Un mot clé doit apparaître dans au moins ce nombre de fichiers pour être affiché dans le tableau des mots clés communs"
    )

    if 'competitors' not in st.session_state:
        st.session_state.competitors = []

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("➕ Ajouter un fichier concurrent", key="add_competitor"):
            st.session_state.competitors.append({
                'id': str(uuid.uuid4()),
                'file': None,
                'regex': "",
                'is_client': False,
                'dataframe': None  # Ajout du stockage du DataFrame
            })

    results = []
    competitors_to_remove = []

    for idx, competitor in enumerate(st.session_state.competitors):
        with st.container():
            col1, col2, col3, col4 = st.columns([6, 3, 1, 0.5])

            with col1:
                file = st.file_uploader(
                    f"Concurrent {idx + 1}",
                    type='csv',
                    key=f"file_{competitor['id']}"
                )

            with col2:
                regex = st.text_input(
                    "Regex",
                    value=competitor.get('regex', ''),
                    key=f"regex_{competitor['id']}",
                    placeholder="Ex: brand|nom"
                )

            with col3:
                is_client = st.checkbox(
                    "Client",
                    value=competitor.get('is_client', False),
                    key=f"client_{competitor['id']}"
                )

            with col4:
                if st.button("🗑️", key=f"remove_{competitor['id']}", help="Supprimer ce concurrent"):
                    competitors_to_remove.append(idx)

            # Afficher le fichier uploadé en dessous si présent
            if file:
                try:
                    df = read_csv_safely(file)
                    if df is not None:
                        competitor['dataframe'] = df  # Stocker le DataFrame
                        analysis = analyze_semrush_data(df, file.name, regex, is_client)
                        if analysis:
                            results.append(analysis)
                except Exception as e:
                    st.error(f"Erreur lors du traitement de {file.name}: {str(e)}")

            st.markdown("</div>", unsafe_allow_html=True)

    for idx in reversed(competitors_to_remove):
        st.session_state.competitors.pop(idx)

    if results:
        st.markdown("---")

        st.markdown("### 📈 Métriques globales")
        df_results = pd.DataFrame(results)

        column_names = {
            'domain': 'Domaine',
            'is_client': 'is_client',
            'total_traffic': 'Traffic Total',
            'total_keywords': 'Nombre de mots clés',
            'top3_keywords': 'Mots clés Top 3 (Organic)',
            'unique_urls': 'URLs uniques',
            'brand_traffic': 'Traffic de marque'
        }

        main_columns = list(column_names.keys())
        df_results = df_results[main_columns].rename(columns=column_names)

        # Ajouter l'emoji pour le client
        df_results['Domaine'] = df_results.apply(
            lambda x: f"🏠 {x['Domaine']}" if x['is_client'] else x['Domaine'],
            axis=1
        )

        # Supprimer la colonne is_client avant l'affichage
        df_results = df_results.drop('is_client', axis=1)

        df_results = df_results.sort_values('Traffic Total', ascending=False)

        # Convertir en numérique et formater sans décimales
        df_results['Traffic Total'] = pd.to_numeric(df_results['Traffic Total']).astype(int)
        df_results['Traffic de marque'] = pd.to_numeric(df_results['Traffic de marque']).astype(int)

        # Calcul des moyennes séparées pour client et concurrents
        client_data = df_results[df_results['Domaine'].str.startswith('🏠 ')]
        competitors_data = df_results[~df_results['Domaine'].str.startswith('🏠 ')]

        # Fonction pour formater les nombres avec séparateur de milliers
        def format_number(num):
            return f"{num:,.0f}".replace(',', ' ')

        # Calcul des moyennes des concurrents
        avg_competitors_urls = competitors_data['URLs uniques'].mean()
        avg_competitors_keywords = competitors_data['Nombre de mots clés'].mean()
        avg_competitors_traffic = competitors_data['Traffic Total'].mean()

        # Valeurs du client
        client_urls = client_data['URLs uniques'].iloc[0] if not client_data.empty else 0
        client_keywords = client_data['Nombre de mots clés'].iloc[0] if not client_data.empty else 0
        client_traffic = client_data['Traffic Total'].iloc[0] if not client_data.empty else 0

        # Calcul des différences en pourcentage
        def calc_difference(client_val, avg_val):
            if avg_val == 0:
                return 0
            return ((client_val - avg_val) / avg_val) * 100

        urls_diff = calc_difference(client_urls, avg_competitors_urls)
        keywords_diff = calc_difference(client_keywords, avg_competitors_keywords)
        traffic_diff = calc_difference(client_traffic, avg_competitors_traffic)

        # Affichage des métriques
        col1, col2, col3 = st.columns(3)
        metrics = [
            {
                "title": "🔗 Moyenne URLs uniques",
                "competitor_value": format_number(avg_competitors_urls),
                "client_value": format_number(client_urls),
                "difference": urls_diff
            },
            {
                "title": "🎯 Moyenne mots clés",
                "competitor_value": format_number(avg_competitors_keywords),
                "client_value": format_number(client_keywords),
                "difference": keywords_diff
            },
            {
                "title": "📈 Moyenne traffic",
                "competitor_value": format_number(avg_competitors_traffic),
                "client_value": format_number(client_traffic),
                "difference": traffic_diff
            }
        ]

        for col, metric in zip([col1, col2, col3], metrics):
            with col:
                st.markdown(f"""
                        <div class='custom-metric'>
                            <div style='color: #666; font-size: 0.8em;'>{metric['title']}</div>
                            <div style='margin: 0.5em 0;'>
                                <span style='font-size: 0.8em; color: #666;'>Concurrents:</span>
                                <div style='font-size: 1.2em; font-weight: bold;'>{metric['competitor_value']}</div>
                            </div>
                            <div style='margin: 0.5em 0;'>
                                <span style='font-size: 0.8em; color: #666;'>Client:</span>
                                <div style='font-size: 1.2em; font-weight: bold;'>{metric['client_value']}</div>
                            </div>
                            <div style='font-size: 0.9em; color: {"#2ecc71" if metric["difference"] >= 0 else "#e74c3c"}'>
                                {format_number(abs(metric["difference"]))}% {" au-dessus" if metric["difference"] >= 0 else " en dessous"} de la moyenne
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("### 📊 Tableau comparatif des performances")
        st.dataframe(
            style_dataframe(df_results),
            hide_index=True,
            use_container_width=True,
            height=400
        )

        st.markdown("### 🔍 Analyse des mots clés communs")

        # Récupérer tous les DataFrames
        all_dfs = []
        client_df = None

        # Debug: Afficher le nombre de concurrents
        st.write(f"Nombre de concurrents: {len(st.session_state.competitors)}")

        for competitor in st.session_state.competitors:
            # On vérifie maintenant si on a soit un fichier soit un DataFrame
            if competitor.get('file') or competitor.get('dataframe') is not None:
                try:
                    # Si on a un DataFrame, on l'utilise directement
                    if competitor.get('dataframe') is not None:
                        df = competitor['dataframe']
                        st.write("DataFrame récupéré du cache")
                    # Sinon on lit le fichier
                    elif competitor.get('file'):
                        df = read_csv_safely(competitor['file'])
                        competitor['dataframe'] = df
                        st.write("Nouveau DataFrame lu")

                    if df is not None:
                        if competitor.get('is_client'):
                            client_df = df
                            st.write("DataFrame client ajouté")
                        else:
                            all_dfs.append(df)
                            st.write("DataFrame concurrent ajouté")

                except Exception as e:
                    st.error(f"Erreur lors de la lecture du fichier pour l'analyse des mots clés: {str(e)}")
                    competitor['dataframe'] = None

        # Debug: Afficher le nombre de DataFrames collectés
        st.write(f"Nombre de DataFrames concurrents: {len(all_dfs)}")
        st.write(f"Client DataFrame présent: {client_df is not None}")

        if all_dfs:
            try:
                common_keywords_df = analyze_common_keywords(all_dfs, min_occurrences, client_df)

                if not common_keywords_df.empty:
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

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Total mots clés communs",
                            len(common_keywords_df)
                        )
                    with col2:
                        present_in_client = (common_keywords_df['Présent chez le client'] == 'Oui').sum()
                        st.metric(
                            "Mots clés présents chez le client",
                            f"{present_in_client} ({round(present_in_client / len(common_keywords_df) * 100)}%)"
                        )
                else:
                    st.warning("Le DataFrame des mots clés communs est vide")
            except Exception as e:
                st.error(f"Erreur lors de l'analyse des mots clés communs: {str(e)}")
                st.write("Détails de l'erreur:", e)
        else:
            st.warning("Aucun DataFrame concurrent n'a été collecté")

if __name__ == '__main__':
    main()