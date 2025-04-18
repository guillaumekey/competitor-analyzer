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
    """Affiche les métriques comparatives en utilisant uniquement des composants natifs Streamlit."""
    # Utiliser un conteneur avec background gris clair
    with st.container():
        # Ajouter un padding et un background gris clair simple
        st.markdown("""
        <style>
        div[data-testid="stVerticalBlock"]:has(div.metric-header) {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .metric-header {
            font-weight: bold;
            margin-bottom: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Créer trois colonnes pour les trois types de métriques
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-header">🔗 Moyenne URLs uniques</div>', unsafe_allow_html=True)
            st.markdown(f"**Concurrents:** {format_number(avg_competitors_urls)}")
            st.markdown(f"**Client:** {format_number(client_urls)}")

            diff_color = "green" if urls_diff >= 0 else "red"
            diff_text = "au-dessus" if urls_diff >= 0 else "en dessous"
            st.markdown(
                f"<span style='color:{diff_color}'>{format_number(abs(urls_diff))}% {diff_text} de la moyenne</span>",
                unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-header">🎯 Moyenne mots clés</div>', unsafe_allow_html=True)
            st.markdown(f"**Concurrents:** {format_number(avg_competitors_keywords)}")
            st.markdown(f"**Client:** {format_number(client_keywords)}")

            diff_color = "green" if keywords_diff >= 0 else "red"
            diff_text = "au-dessus" if keywords_diff >= 0 else "en dessous"
            st.markdown(
                f"<span style='color:{diff_color}'>{format_number(abs(keywords_diff))}% {diff_text} de la moyenne</span>",
                unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-header">📈 Moyenne traffic</div>', unsafe_allow_html=True)
            st.markdown(f"**Concurrents:** {format_number(avg_competitors_traffic)}")
            st.markdown(f"**Client:** {format_number(client_traffic)}")

            diff_color = "green" if traffic_diff >= 0 else "red"
            diff_text = "au-dessus" if traffic_diff >= 0 else "en dessous"
            st.markdown(
                f"<span style='color:{diff_color}'>{format_number(abs(traffic_diff))}% {diff_text} de la moyenne</span>",
                unsafe_allow_html=True)


def render_keyword_stats(total_keywords, present_in_client, percentage, potential_volume):
    """Affiche les statistiques des mots-clés avec les composants natifs Streamlit."""
    # Utiliser les composants métriques natifs de Streamlit qui fonctionnent très bien
    cols = st.columns(3)

    with cols[0]:
        st.metric(
            label="📊 Total mots clés communs",
            value=format_number(total_keywords)
        )

    with cols[1]:
        st.metric(
            label="🎯 Présents chez le client",
            value=f"{format_number(present_in_client)} ({percentage}%)"
        )

    with cols[2]:
        st.metric(
            label="💎 Volume recherche non exploité",
            value=format_number(potential_volume),
            help="Potentiel de trafic mensuel à conquérir"
        )


def render_opportunity_stats(opportunities_count, opportunity_volume):
    """Affiche les statistiques d'opportunités avec les composants natifs Streamlit."""
    # Utiliser les composants métriques natifs de Streamlit
    cols = st.columns(2)

    with cols[0]:
        st.metric(
            label="✨ Mots-clés à faible difficulté",
            value=format_number(opportunities_count)
        )

    with cols[1]:
        st.metric(
            label="🚀 Volume de recherche potentiel",
            value=format_number(opportunity_volume),
            help="Trafic mensuel facilement atteignable"
        )