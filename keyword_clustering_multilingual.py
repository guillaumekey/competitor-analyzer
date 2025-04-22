import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import plotly.express as px
from collections import Counter
import spacy
import os
import warnings

# Configuration pour les modèles de langue
LANGUAGE_MODELS = {
    'fr': 'fr_core_news_sm',
    'en': 'en_core_web_sm',
    'it': 'it_core_news_sm',
    'de': 'de_core_news_sm',
    'es': 'es_core_news_sm',
    'pt': 'pt_core_news_sm'
}

# Dictionnaire des stopwords pour différentes langues
STOPWORDS = {}


# Fonction pour charger ou télécharger des modèles de langue
@st.cache_resource
def load_language_model(lang_code):
    """
    Charge un modèle de langue spaCy. Le télécharge s'il n'est pas disponible.

    Args:
        lang_code: Code de langue (fr, en, it, de, es, pt)

    Returns:
        Modèle spaCy chargé
    """
    model_name = LANGUAGE_MODELS.get(lang_code)
    if not model_name:
        st.warning(f"Modèle pour la langue '{lang_code}' non configuré. Utilisation du modèle anglais par défaut.")
        model_name = 'en_core_web_sm'

    try:
        # Essayer de charger le modèle
        return spacy.load(model_name)
    except OSError:
        # Si le modèle n'est pas disponible, le télécharger
        with st.spinner(f"Téléchargement du modèle de langue {model_name}..."):
            try:
                spacy.cli.download(model_name)
                return spacy.load(model_name)
            except Exception as e:
                st.error(f"Impossible de télécharger le modèle {model_name}: {str(e)}")
                # Fallback sur le modèle anglais qui est généralement disponible
                try:
                    spacy.cli.download("en_core_web_sm")
                    return spacy.load("en_core_web_sm")
                except:
                    st.error(
                        "Impossible de charger un modèle de langue. Veuillez installer manuellement spaCy et ses modèles.")
                    return None


# Fonction pour détecter automatiquement la langue
def detect_language(text):
    """
    Détecte la langue d'un texte.

    Args:
        text: Texte à analyser

    Returns:
        Code de langue (fr, en, it, de, es, pt) ou 'en' par défaut
    """
    try:
        from langdetect import detect
        lang = detect(text)
        if lang in LANGUAGE_MODELS:
            return lang
        return 'en'  # Fallback sur l'anglais
    except:
        # Si langdetect n'est pas disponible ou échoue
        return 'en'  # Fallback sur l'anglais


# Fonction pour charger les stopwords pour toutes les langues supportées
@st.cache_data
def load_stopwords():
    """Charge les stopwords NLTK pour toutes les langues supportées."""
    global STOPWORDS

    # Mapping des langues NLTK
    nltk_langs = {
        'fr': 'french',
        'en': 'english',
        'it': 'italian',
        'de': 'german',
        'es': 'spanish',
        'pt': 'portuguese'
    }

    # Télécharger les stopwords si nécessaire
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Charger tous les stopwords disponibles
    for lang_code, nltk_lang in nltk_langs.items():
        try:
            STOPWORDS[lang_code] = set(stopwords.words(nltk_lang))
        except:
            # Si la langue n'est pas disponible, utiliser un ensemble vide
            STOPWORDS[lang_code] = set()

    return STOPWORDS


# Charger les stopwords au démarrage
load_stopwords()


# Modèle multilingue pour les embeddings
@st.cache_resource
def load_multilingual_model():
    """Charge un modèle multilingue pour les embeddings."""
    try:
        # Essayer de charger le modèle multilingue si disponible
        return spacy.load("xx_ent_wiki_sm")
    except:
        try:
            # Sinon, télécharger et retourner
            spacy.cli.download("xx_ent_wiki_sm")
            return spacy.load("xx_ent_wiki_sm")
        except:
            # Si ça échoue, retourner None
            return None


# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def preprocess_keywords(df, text_column='Mot clé'):
    """
    Prétraitement des mots-clés pour le clustering.

    Args:
        df: DataFrame avec les mots-clés
        text_column: Nom de la colonne contenant les mots-clés

    Returns:
        DataFrame avec les mots-clés prétraités
    """
    # Créer une copie pour éviter de modifier l'original
    processed_df = df.copy()

    # Normaliser les mots-clés (minuscules, sans accents)
    processed_df['processed_keyword'] = processed_df[text_column].str.lower()

    # Détecter la langue de chaque mot-clé
    if 'detected_language' not in processed_df.columns:
        # Appliquer la détection de langue sur un échantillon pour réduire le temps de traitement
        # Pour les petits datasets, analyser tous les mots-clés
        if len(processed_df) <= 1000:
            processed_df['detected_language'] = processed_df[text_column].apply(
                lambda x: detect_language(x) if isinstance(x, str) and x.strip() else 'en'
            )
        else:
            # Pour les grands datasets, faire un échantillon et attribuer la langue majoritaire
            # Cela fonctionne bien pour les jeux de données unilingues
            sample_size = min(1000, len(processed_df) // 10)
            sample = processed_df[text_column].dropna().sample(sample_size)
            langs = [detect_language(text) for text in sample if isinstance(text, str) and text.strip()]

            if langs:
                # Langue majoritaire
                dominant_lang = max(set(langs), key=langs.count)
                processed_df['detected_language'] = dominant_lang
            else:
                # Fallback sur l'anglais
                processed_df['detected_language'] = 'en'

    # Supprimer les caractères spéciaux, mais conserver les espaces
    processed_df['processed_keyword'] = processed_df['processed_keyword'].apply(
        lambda x: re.sub(r'[^\w\s]', '', x) if isinstance(x, str) else x
    )

    return processed_df


def create_keyword_vectors(processed_df, method='tfidf', max_features=5000):
    """
    Crée des vecteurs à partir des mots-clés selon différentes méthodes.

    Args:
        processed_df: DataFrame avec les mots-clés prétraités
        method: Méthode de vectorisation ('tfidf', 'spacy', 'multilingual')
        max_features: Nombre maximum de caractéristiques pour TF-IDF

    Returns:
        Matrice de caractéristiques et vectorizer (si applicable)
    """
    # S'assurer que la colonne existe
    if 'processed_keyword' not in processed_df.columns:
        st.error("La colonne 'processed_keyword' n'existe pas. Exécutez d'abord le prétraitement.")
        return None, None

    # Supprimer les valeurs NaN
    keywords = processed_df['processed_keyword'].dropna().tolist()

    if method == 'tfidf':
        # Vectorisation TF-IDF adaptée multilingue
        # Collecter tous les stopwords de toutes les langues
        all_stopwords = set()
        for lang_stopwords in STOPWORDS.values():
            all_stopwords.update(lang_stopwords)

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=list(all_stopwords),
            ngram_range=(1, 2)  # Unigrammes et bigrammes
        )
        feature_matrix = vectorizer.fit_transform(keywords)
        return feature_matrix, vectorizer

    elif method == 'spacy':
        # Vectorisation avec spaCy par langue
        vectors = []

        # Obtenir la langue de chaque mot-clé
        languages = processed_df['detected_language'].dropna().tolist()

        # S'assurer que nous avons une langue pour chaque mot-clé
        if len(languages) != len(keywords):
            # Utiliser l'anglais comme fallback
            languages = ['en'] * len(keywords)

        # Charger tous les modèles nécessaires
        needed_models = set(languages)
        loaded_models = {}

        for lang in needed_models:
            model = load_language_model(lang)
            if model:
                loaded_models[lang] = model

        # Traiter chaque mot-clé avec le modèle de sa langue
        for i, keyword in enumerate(keywords):
            lang = languages[i] if i < len(languages) else 'en'
            model = loaded_models.get(lang) or loaded_models.get('en')

            if model and keyword:
                doc = model(keyword)
                if doc.vector.any():  # Vérifier que le vecteur n'est pas nul
                    vectors.append(doc.vector)
                else:
                    # Fallback si le mot est inconnu: vecteur de zéros
                    vectors.append(np.zeros(model.vocab.vectors.shape[1]))
            else:
                # Si pas de modèle disponible, créer un vecteur vide
                # Utiliser une dimension standard pour spaCy
                vectors.append(np.zeros(300))

        feature_matrix = np.array(vectors)
        return feature_matrix, None

    elif method == 'multilingual':
        # Utiliser un modèle multilingue unique
        multilingual_model = load_multilingual_model()

        if multilingual_model:
            vectors = []
            for keyword in keywords:
                if isinstance(keyword, str) and keyword.strip():
                    doc = multilingual_model(keyword)
                    if doc.vector.any():
                        vectors.append(doc.vector)
                    else:
                        vectors.append(np.zeros(multilingual_model.vocab.vectors.shape[1]))
                else:
                    vectors.append(np.zeros(multilingual_model.vocab.vectors.shape[1]))

            feature_matrix = np.array(vectors)
            return feature_matrix, None
        else:
            st.warning("Modèle multilingue non disponible. Utilisation de TF-IDF à la place.")
            return create_keyword_vectors(processed_df, method='tfidf', max_features=max_features)

    else:
        st.error(f"Méthode de vectorisation '{method}' non reconnue")
        return None, None


def cluster_keywords(feature_matrix, method='kmeans', n_clusters=None):
    """
    Clusterise les mots-clés selon différentes méthodes.

    Args:
        feature_matrix: Matrice de caractéristiques
        method: Méthode de clustering ('kmeans', 'dbscan')
        n_clusters: Nombre de clusters (pour kmeans)

    Returns:
        Liste des labels de cluster pour chaque mot-clé
    """
    if feature_matrix is None or feature_matrix.shape[0] < 2:
        st.error("Données insuffisantes pour le clustering")
        return []

    # Déterminer le nombre optimal de clusters si non spécifié
    if method == 'kmeans' and n_clusters is None:
        # Utiliser la méthode du coude
        inertia = []
        silhouette_scores = []
        k_range = range(2, min(20, feature_matrix.shape[0] - 1))

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(feature_matrix)
                inertia.append(kmeans.inertia_)

                # Calculer le score de silhouette si possible
                if len(set(kmeans.labels_)) > 1:  # Au moins 2 clusters
                    silhouette_avg = silhouette_score(feature_matrix, kmeans.labels_)
                    silhouette_scores.append(silhouette_avg)
                else:
                    silhouette_scores.append(0)
            except Exception as e:
                warnings.warn(f"Erreur lors du calcul pour k={k}: {str(e)}")
                continue

        # Trouver le "coude" dans le graphique d'inertie ou le meilleur score de silhouette
        if silhouette_scores:
            best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
        else:
            best_k = 5  # Valeur par défaut si la méthode échoue

        n_clusters = best_k

    # Appliquer l'algorithme de clustering
    if method == 'kmeans':
        # K-means est simple et efficace pour la plupart des cas
        n_clusters = min(n_clusters, feature_matrix.shape[0] - 1)
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return clusterer.fit_predict(feature_matrix)

    elif method == 'dbscan':
        # DBSCAN est utile si vous ne savez pas combien de clusters vous voulez
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        return clusterer.fit_predict(feature_matrix)

    else:
        st.error(f"Méthode de clustering '{method}' non reconnue")
        return []


def label_clusters(df, clusters, vectorizer=None, text_column='processed_keyword'):
    """
    Étiquette automatiquement les clusters en fonction des mots les plus importants.

    Args:
        df: DataFrame avec les mots-clés
        clusters: Liste des labels de cluster
        vectorizer: Vectorizer TF-IDF (si disponible)
        text_column: Nom de la colonne contenant les mots-clés

    Returns:
        DataFrame avec les clusters étiquetés
    """
    if len(clusters) != len(df):
        st.error("Nombre de clusters incompatible avec le nombre de mots-clés")
        return df

    # Ajouter les labels de cluster au DataFrame
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters

    # Générer des étiquettes pour chaque cluster
    cluster_labels = {}

    for cluster_id in set(clusters):
        if cluster_id == -1:  # -1 est utilisé par DBSCAN pour les outliers
            cluster_labels[cluster_id] = "Outliers"
            continue

        # Extraire les mots-clés de ce cluster
        cluster_keywords = df_with_clusters[df_with_clusters['cluster'] == cluster_id][text_column].tolist()

        if not cluster_keywords:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
            continue

        # Méthode 1: Utiliser les mots les plus fréquents
        all_words = ' '.join([str(kw) for kw in cluster_keywords if isinstance(kw, str)]).split()
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.most_common(3)]

        if common_words:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}: {', '.join(common_words)}"
        else:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"

    # Ajouter les étiquettes au DataFrame
    df_with_clusters['cluster_label'] = df_with_clusters['cluster'].map(cluster_labels)

    return df_with_clusters


def visualize_clusters(df, feature_matrix):
    """
    Visualise les clusters de mots-clés en 2D ou 3D.

    Args:
        df: DataFrame avec les clusters
        feature_matrix: Matrice de caractéristiques

    Returns:
        Figure Plotly
    """
    # Réduire la dimensionnalité pour la visualisation
    if feature_matrix.shape[1] > 3:
        # Utiliser PCA pour réduire à 3 dimensions
        pca = PCA(n_components=3)
        reduced_features = pca.fit_transform(
            feature_matrix.toarray() if hasattr(feature_matrix, 'toarray') else feature_matrix)
    else:
        reduced_features = feature_matrix

    # Créer un DataFrame pour Plotly
    vis_df = pd.DataFrame()
    vis_df['x'] = reduced_features[:, 0]
    vis_df['y'] = reduced_features[:, 1]
    if reduced_features.shape[1] > 2:
        vis_df['z'] = reduced_features[:, 2]

    vis_df['mot_cle'] = df['Mot clé'].values[:len(reduced_features)]
    vis_df['cluster'] = df['cluster_label'].values[:len(reduced_features)]
    vis_df['volume'] = df['Volume de recherche'].values[:len(reduced_features)]

    # Créer la visualisation
    if reduced_features.shape[1] > 2:
        fig = px.scatter_3d(
            vis_df, x='x', y='y', z='z',
            color='cluster', hover_name='mot_cle',
            size='volume', size_max=20,
            title='Clusters de mots-clés en 3D'
        )
    else:
        fig = px.scatter(
            vis_df, x='x', y='y',
            color='cluster', hover_name='mot_cle',
            size='volume', size_max=20,
            title='Clusters de mots-clés en 2D'
        )

    fig.update_layout(height=700)
    return fig


def cluster_analysis_summary(df):
    """
    Fournit une analyse des clusters identifiés.

    Args:
        df: DataFrame avec les clusters

    Returns:
        DataFrame avec l'analyse des clusters
    """
    if 'cluster' not in df.columns:
        return pd.DataFrame()

    # Calculer des statistiques par cluster
    cluster_stats = df.groupby('cluster_label').agg({
        'Mot clé': 'count',
        'Volume de recherche': ['mean', 'sum'],
        'Difficulté': 'mean',
        'Présent chez le client': lambda x: (x == 'Oui').mean() * 100
    }).reset_index()

    # Renommer les colonnes
    cluster_stats.columns = [
        'Cluster', 'Nombre de mots-clés', 'Volume moyen',
        'Volume total', 'Difficulté moyenne', '% Présent chez le client'
    ]

    # Trier par volume total décroissant
    cluster_stats = cluster_stats.sort_values('Volume total', ascending=False)

    # Arrondir les valeurs numériques
    cluster_stats['Volume moyen'] = cluster_stats['Volume moyen'].round(0).astype(int)
    cluster_stats['Volume total'] = cluster_stats['Volume total'].round(0).astype(int)
    cluster_stats['Difficulté moyenne'] = cluster_stats['Difficulté moyenne'].round(1)
    cluster_stats['% Présent chez le client'] = cluster_stats['% Présent chez le client'].round(1)

    return cluster_stats


def get_cluster_top_keywords(df, n=5):
    """
    Récupère les mots-clés les plus importants de chaque cluster.

    Args:
        df: DataFrame avec les clusters
        n: Nombre de mots-clés à afficher par cluster

    Returns:
        Dict avec les clusters et leurs mots-clés les plus importants
    """
    top_keywords = {}

    for cluster in df['cluster_label'].unique():
        # Filtrer les mots-clés de ce cluster
        cluster_df = df[df['cluster_label'] == cluster]

        # Trier par volume de recherche
        top_kw = cluster_df.sort_values('Volume de recherche', ascending=False).head(n)

        # Stocker les résultats
        top_keywords[cluster] = top_kw

    return top_keywords


def render_clustering_ui(df):
    """
    Interface utilisateur pour le clustering des mots-clés.

    Args:
        df: DataFrame avec les mots-clés communs
    """
    st.markdown("### 🧩 Clustering de mots-clés")

    if df is None or df.empty:
        st.warning("Aucune donnée disponible pour le clustering. Veuillez d'abord analyser des mots-clés.")
        return

    # Options de clustering
    with st.expander("Options de clustering", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            method = st.radio(
                "Méthode de vectorisation",
                options=["TF-IDF (multilingual)", "SpaCy (par langue)", "SpaCy (modèle multilingue)"],
                index=0,
                key="vectorization_method"
            )
            if method == "TF-IDF (multilingual)":
                vectorization_method = "tfidf"
            elif method == "SpaCy (par langue)":
                vectorization_method = "spacy"
            else:
                vectorization_method = "multilingual"

        with col2:
            cluster_method = st.radio(
                "Méthode de clustering",
                options=["K-means", "DBSCAN"],
                index=0,
                key="cluster_method"
            )
            clustering_method = "kmeans" if cluster_method == "K-means" else "dbscan"

        if clustering_method == "kmeans":
            n_clusters = st.slider(
                "Nombre de clusters",
                min_value=2,
                max_value=min(50, len(df) - 1) if len(df) > 2 else 2,
                value=min(8, len(df) - 1) if len(df) > 2 else 2,
                help="Laissez vide pour déterminer automatiquement le nombre optimal"
            )
        else:
            n_clusters = None

    # Bouton pour lancer le clustering
    if st.button("Lancer le clustering", type="primary"):
        with st.spinner("Clustering en cours..."):
            # 1. Prétraitement
            st.info("Étape 1/4: Prétraitement des mots-clés...")
            processed_df = preprocess_keywords(df)

            # 2. Vectorisation
            st.info("Étape 2/4: Vectorisation des mots-clés...")
            feature_matrix, vectorizer = create_keyword_vectors(
                processed_df,
                method=vectorization_method
            )

            if feature_matrix is None or feature_matrix.shape[0] < 2:
                st.error("Impossible de créer des vecteurs pour les mots-clés.")
                return

            # 3. Clustering
            st.info("Étape 3/4: Clustering des mots-clés...")
            cluster_labels = cluster_keywords(
                feature_matrix,
                method=clustering_method,
                n_clusters=n_clusters
            )

            # 4. Étiquetage des clusters
            st.info("Étape 4/4: Étiquetage des clusters...")
            clustered_df = label_clusters(
                processed_df,
                cluster_labels,
                vectorizer
            )

            # Fusionner avec le DataFrame original
            result_df = df.copy()
            result_df['cluster'] = clustered_df['cluster'].values
            result_df['cluster_label'] = clustered_df['cluster_label'].values

            # Stocker le résultat en session state
            st.session_state.clustered_keywords = result_df

            # Nombre de clusters trouvés
            unique_clusters = len(set(cluster_labels))
            if -1 in cluster_labels:  # DBSCAN utilise -1 pour les outliers
                unique_clusters -= 1

            st.success(f"Clustering terminé ! {unique_clusters} clusters identifiés.")

    # Afficher les résultats du clustering s'ils existent
    if 'clustered_keywords' in st.session_state and not st.session_state.clustered_keywords.empty:
        clustered_df = st.session_state.clustered_keywords

        # Créer des onglets pour les différentes vues
        clustering_tabs = st.tabs(
            ["Aperçu des clusters", "Visualisation", "Tableau de données", "Mots-clés par cluster"])

        with clustering_tabs[0]:
            # Résumé des clusters
            st.markdown("#### 📊 Résumé des clusters")
            cluster_summary = cluster_analysis_summary(clustered_df)

            # Styliser le tableau
            def style_cluster_summary(df):
                return df.style.format({
                    'Volume moyen': lambda x: f"{x:,}".replace(',', ' '),
                    'Volume total': lambda x: f"{x:,}".replace(',', ' '),
                    'Difficulté moyenne': '{:.1f}',
                    '% Présent chez le client': '{:.1f}%'
                })

            st.dataframe(
                style_cluster_summary(cluster_summary),
                use_container_width=True,
                height=400
            )

        with clustering_tabs[1]:
            # Visualisation des clusters
            st.markdown("#### 🔍 Visualisation des clusters")

            try:
                # Récupérer les données vectorisées
                processed_df = preprocess_keywords(df)
                feature_matrix, _ = create_keyword_vectors(
                    processed_df,
                    method=vectorization_method
                )

                if feature_matrix is not None:
                    fig = visualize_clusters(clustered_df, feature_matrix)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Impossible de créer la visualisation des clusters.")
            except Exception as e:
                st.error(f"Erreur lors de la visualisation: {str(e)}")

        with clustering_tabs[2]:
            # Tableau complet des mots-clés avec leurs clusters
            st.markdown("#### 📋 Tableau complet des mots-clés")

            # Colonnes à afficher
            display_cols = ['Mot clé', 'cluster_label', 'Volume de recherche',
                            'Difficulté', 'Présent chez le client', 'Nombre de fichiers']
            display_df = clustered_df[display_cols].sort_values('cluster_label')

            # Renommer les colonnes
            display_df = display_df.rename(columns={'cluster_label': 'Cluster'})

            # Styliser le tableau
            def style_keyword_table(df):
                return df.style.format({
                    'Volume de recherche': lambda x: f"{int(x):,}".replace(',', ' '),
                    'Difficulté': lambda x: f"{int(x):,}".replace(',', ' ')
                })

            st.dataframe(
                style_keyword_table(display_df),
                use_container_width=True,
                height=500
            )

        with clustering_tabs[3]:
            # Afficher les mots-clés les plus importants de chaque cluster
            st.markdown("#### 🔑 Mots-clés principaux par cluster")

            top_keywords = get_cluster_top_keywords(clustered_df, n=10)

            for cluster, keywords_df in top_keywords.items():
                with st.expander(f"{cluster} ({len(keywords_df)} mots-clés)", expanded=False):
                    # Afficher les 10 premiers mots-clés par volume
                    display_cols = ['Mot clé', 'Volume de recherche', 'Difficulté', 'Présent chez le client']
                    st.dataframe(
                        keywords_df[display_cols],
                        use_container_width=True,
                        hide_index=True
                    )