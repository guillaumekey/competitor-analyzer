import streamlit as st
import pandas as pd
from urllib.parse import urlparse
from utils.file_utils import normalize_column_names


def extract_domain_from_url(url):
    """Extrait le nom de domaine d'une URL.

    Args:
        url: URL à analyser

    Returns:
        Nom de domaine ou None en cas d'erreur
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return None


def analyze_semrush_data(df, filename, brand_regex, is_client=False):
    """Analyse un DataFrame de données SEMrush.

    Args:
        df: DataFrame pandas contenant les données SEMrush
        filename: Nom du fichier source
        brand_regex: Expression régulière pour détecter les mots-clés de marque
        is_client: Booléen indiquant s'il s'agit du site client

    Returns:
        Dictionnaire contenant les métriques analysées ou None en cas d'erreur
    """
    try:
        # Normaliser les noms de colonnes
        df = normalize_column_names(df)

        # Vérifier la présence de la colonne URL
        if 'url' not in df.columns:
            st.error(f"Colonne URL introuvable dans le fichier {filename}")
            return None

        # Extraire le domaine de la première URL
        first_url = df['url'].iloc[0] if not df['url'].empty else None
        if first_url:
            domain = extract_domain_from_url(first_url)
            if not domain:
                st.error(f"Impossible d'extraire le domaine de l'URL: {first_url}")
                return None
        else:
            st.error("Aucune URL trouvée dans le fichier")
            return None

        # Valider les colonnes nécessaires et convertir les types
        required_columns = {
            'keyword': str,
            'position': float,
            'traffic': float,
            'url': str,
            'position_type': str
        }

        for col, dtype in required_columns.items():
            if col not in df.columns:
                st.error(f"Colonne manquante après normalisation : {col}")
                return None

        # Conversion des types numériques
        df['position'] = pd.to_numeric(df['position'], errors='coerce').fillna(999)
        df['traffic'] = pd.to_numeric(df['traffic'], errors='coerce').fillna(0)

        # Filtrer les résultats organiques
        organic_df = df[df['position_type'].str.contains('Organic', case=False, na=False)]

        # Détecter les mots-clés de marque
        try:
            df['is_brand_keyword'] = df['keyword'].astype(str).str.lower().str.contains(
                brand_regex, case=False, regex=True, na=False
            )
        except Exception as regex_error:
            st.error(f"Erreur avec l'expression régulière '{brand_regex}': {str(regex_error)}")
            df['is_brand_keyword'] = False

        # Retourner les métriques calculées
        return {
            'domain': domain,
            'is_client': is_client,
            'total_traffic': round(df['traffic'].sum()),
            'total_keywords': len(df),
            'top3_keywords': len(organic_df[organic_df['position'].between(1, 3)]),
            'unique_urls': df['url'].nunique(),
            'brand_traffic': round(df[df['is_brand_keyword']]['traffic'].sum())
        }

    except Exception as e:
        st.error(f"Erreur lors de l'analyse de {filename}: {str(e)}")
        return None


def style_dataframe(df):
    """Applique un style simple et fiable au DataFrame, en évitant le HTML complexe.
    Utilise uniquement les méthodes pandas/Streamlit standard.

    Args:
        df: DataFrame pandas à formater

    Returns:
        DataFrame stylé
    """
    try:
        # Créer une copie pour éviter de modifier l'original
        styled_df = df.copy()

        # Formater les nombres avec séparateurs de milliers
        for col in ['Traffic Total', 'Nombre de mots clés', 'Traffic de marque', 'Mots clés Top 3 (Organic)']:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(lambda x: f"{int(x):,}".replace(',', ' '))

        # Créer un styler pandas
        styler = styled_df.style

        # Mettre en évidence le client (fond jaune très pâle)
        def highlight_client(x):
            if isinstance(x, str) and x.startswith('🏠'):
                return 'background-color: #fff8e1'
            else:
                return ''

        styler = styler.applymap(highlight_client, subset=['Domaine'])

        # Traffic Total - bleu très pâle
        def style_traffic(x):
            try:
                # Enlever les espaces pour la comparaison
                value = int(str(x).replace(' ', ''))
                if value > 30000:
                    return 'background-color: #e3f2fd'
                elif value > 15000:
                    return 'background-color: #e8f5e9'
                else:
                    return ''
            except:
                return ''

        styler = styler.applymap(style_traffic, subset=['Traffic Total'])

        # Nombre de mots clés - violet très pâle
        def style_keywords(x):
            try:
                value = int(str(x).replace(' ', ''))
                if value > 5000:
                    return 'background-color: #f3e5f5'
                elif value > 2000:
                    return 'background-color: #ede7f6'
                else:
                    return ''
            except:
                return ''

        styler = styler.applymap(style_keywords, subset=['Nombre de mots clés'])

        # Top 3 - orange très pâle
        def style_top3(x):
            try:
                value = int(str(x).replace(' ', ''))
                if value > 500:
                    return 'background-color: #fff3e0'
                elif value > 100:
                    return 'background-color: #ffebee'
                else:
                    return ''
            except:
                return ''

        styler = styler.applymap(style_top3, subset=['Mots clés Top 3 (Organic)'])

        # Traffic de marque - rouge très pâle
        def style_brand(x):
            try:
                value = int(str(x).replace(' ', ''))
                if value > 1000:
                    return 'background-color: #ffebee'
                elif value > 500:
                    return 'background-color: #fce4ec'
                else:
                    return ''
            except:
                return ''

        styler = styler.applymap(style_brand, subset=['Traffic de marque'])

        # Ajouter un style d'en-tête simple
        styler = styler.set_table_styles([
            {'selector': 'thead th',
             'props': [('background-color', '#f5f5f5'),
                       ('color', '#333333'),
                       ('font-weight', 'bold'),
                       ('text-align', 'center')]},
            {'selector': 'tbody tr:hover',
             'props': [('background-color', '#f9f9f9')]},
        ])

        # Aligner la colonne domaine à gauche
        styler = styler.set_properties(subset=['Domaine'], **{'text-align': 'left'})

        return styler

    except Exception as e:
        # En cas d'erreur, retourner le DataFrame original sans style
        if st.session_state.get('debug_mode', False):
            st.warning(f"Erreur lors de l'application du style: {str(e)}")
        return df