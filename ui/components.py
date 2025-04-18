import streamlit as st
from utils.styling import format_number


def render_instructions():
    """Affiche les instructions d'utilisation de l'application."""
    st.markdown("""
    1. Cliquez sur le bouton "Ajouter un fichier concurrent" pour chaque site à analyser
    2. Pour chaque concurrent :
       - Uploadez le fichier d'export SEMrush (CSV)
       - Définissez l'expression régulière pour identifier les mots-clés de marque
       - Cochez la case "Client" s'il s'agit de votre site
    3. Le tableau comparatif se mettra à jour automatiquement

    **Note**: Les expressions régulières peuvent inclure plusieurs variantes (ex: `marque|brand|nom`)
    """)


def render_metrics(avg_competitors_urls, client_urls, urls_diff,
                   avg_competitors_keywords, client_keywords, keywords_diff,
                   avg_competitors_traffic, client_traffic, traffic_diff):
    """Affiche les métriques comparatives entre le client et les concurrents.

    Args:
        avg_competitors_urls: Moyenne des URLs uniques des concurrents
        client_urls: Nombre d'URLs uniques du client
        urls_diff: Différence en pourcentage pour les URLs
        avg_competitors_keywords: Moyenne des mots-clés des concurrents
        client_keywords: Nombre de mots-clés du client
        keywords_diff: Différence en pourcentage pour les mots-clés
        avg_competitors_traffic: Moyenne du trafic des concurrents
        client_traffic: Trafic du client
        traffic_diff: Différence en pourcentage pour le trafic
    """
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