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
    """Applique un style personnalisé au DataFrame avec mise en forme conditionnelle.

    Args:
        df: DataFrame pandas à formater

    Returns:
        DataFrame stylé
    """
    df_style = df.copy()

    # Assurer que les colonnes numériques le sont bien
    for col in ['Traffic Total', 'Traffic de marque']:
        df_style[col] = pd.to_numeric(df_style[col], errors='coerce')

    def color_scale(val, col):
        """Fonction pour appliquer une échelle de couleur en fonction de la valeur."""
        if pd.isna(val):
            return ''

        max_val = df_style[col].max()
        min_val = df_style[col].min()
        norm_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0

        return f'background-color: rgba(0, 99, 255, {norm_val * 0.2}); color: {"#1e1e1e" if norm_val < 0.7 else "white"}'

    def style_domain(val):
        """Fonction pour mettre en évidence le domaine client."""
        if val.startswith('🏠 '):
            return 'font-weight: bold; background-color: rgba(255, 215, 0, 0.1)'
        return ''

    # Appliquer les styles
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