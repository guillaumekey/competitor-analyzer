"""
Script pour installer les modèles spaCy nécessaires pour l'analyse multilingue.
Vous pouvez exécuter ce script avant de démarrer l'application.
"""
import subprocess
import sys


def install_spacy_models():
    """Installe les modèles de langue spaCy nécessaires."""
    # Liste des modèles à installer
    models = [
        "fr_core_news_sm",  # Français
        "en_core_web_sm",  # Anglais
        "it_core_news_sm",  # Italien
        "de_core_news_sm",  # Allemand
        "es_core_news_sm",  # Espagnol
        "pt_core_news_sm",  # Portugais
        "xx_ent_wiki_sm"  # Modèle multilingue
    ]

    print("Installation des modèles spaCy...")

    for model in models:
        print(f"Installation du modèle {model}...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            print(f"✅ {model} installé avec succès")
        except subprocess.CalledProcessError:
            print(f"❌ Erreur lors de l'installation de {model}")

    print("\nInstallation des ressources NLTK...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        print("✅ Ressources NLTK installées avec succès")
    except Exception as e:
        print(f"❌ Erreur lors de l'installation des ressources NLTK: {str(e)}")

    print("\nInstallation terminée!")


if __name__ == "__main__":
    install_spacy_models()