import streamlit as st
import pandas as pd
import numpy as np
import re


def render_filter_ui():
    """
    Interface de filtrage simplifiée.

    Returns:
        dict: Configuration de filtrage avec règles et opérateurs logiques
    """
    # Initialiser les états de session
    for key in ['filter_rules', 'filters_applied', 'filter_changed']:
        if key not in st.session_state:
            if key == 'filter_rules':
                st.session_state[key] = []
            else:
                st.session_state[key] = False

    # Interface pour les filtres
    with st.container():
        st.markdown("### Filtres avancés")
        st.text(f"{len(st.session_state.filter_rules)} règle(s) active(s)")

        # Sélection de l'opérateur logique global
        st.radio("Logique d'application des filtres :",
                 options=["ET (toutes les règles doivent être satisfaites)",
                          "OU (au moins une règle doit être satisfaite)"],
                 horizontal=True,
                 key="global_logical_op")
        global_op = "AND" if "ET" in st.session_state.global_logical_op else "OR"

        # Utiliser un formulaire pour le contrôle précis des interactions
        with st.form(key="filter_form"):
            # Afficher les règles existantes
            if st.session_state.filter_rules:
                st.markdown("#### Règles actives")

                # Utiliser des colonnes pour afficher les règles plus proprement
                for i, rule in enumerate(st.session_state.filter_rules):
                    rule_id = f"rule_{i}"

                    # Afficher la règle
                    st.text(f"{rule['field']} {rule['operator']} \"{rule['value']}\"")

                    # Ajouter des contrôles pour chaque règle
                    cols = st.columns([4, 1])
                    with cols[0]:
                        if i < len(st.session_state.filter_rules) - 1:
                            op_options = ["ET", "OU"]
                            op_default = 0 if rule.get('next_op', 'AND') == 'AND' else 1
                            rule['next_op'] = "AND" if st.selectbox(
                                "Opérateur avec la règle suivante :",
                                options=op_options,
                                index=op_default,
                                key=f"op_{rule_id}"
                            ) == "ET" else "OR"

                    with cols[1]:
                        if st.checkbox("Supprimer", key=f"del_{rule_id}"):
                            rule['delete'] = True

            # Interface pour ajouter une nouvelle règle
            st.markdown("#### Ajouter une nouvelle règle")

            # Utiliser des colonnes pour un formulaire plus compact
            cols = st.columns([2, 2, 3, 1])

            with cols[0]:
                field = st.selectbox(
                    "Champ",
                    options=["Mot clé", "Volume de recherche", "Difficulté", "Présent chez le client"],
                    key="new_rule_field"
                )

            with cols[1]:
                # Adapter les opérateurs selon le type de champ
                if field in ["Volume de recherche", "Difficulté"]:
                    operator_options = [">=", "<=", "=", ">", "<"]
                elif field == "Présent chez le client":
                    operator_options = ["est", "n'est pas"]
                else:  # Pour les mots-clés
                    operator_options = ["contient", "ne contient pas", "égal à", "commence par", "finit par", "regex"]

                operator = st.selectbox("Opérateur", options=operator_options, key="new_rule_operator")

            with cols[2]:
                # Adapter l'input selon le type de champ
                if field in ["Volume de recherche", "Difficulté"]:
                    value = st.number_input("Valeur", min_value=0, value=0, key="new_rule_value_numeric")
                    value = str(value)
                elif field == "Présent chez le client":
                    value = st.selectbox("Valeur", options=["Oui", "Non"], key="new_rule_value_select")
                else:  # Pour les mots-clés
                    value = st.text_input("Valeur", key="new_rule_value_text", placeholder="Terme à filtrer")

            with cols[3]:
                add_new_rule = st.checkbox("Ajouter", key="add_new_rule", value=False)

            # Bouton pour appliquer les filtres
            col1, col2 = st.columns([4, 1])
            with col2:
                submitted = st.form_submit_button(
                    "Appliquer les filtres",
                    type="primary",
                    use_container_width=True
                )

        # Traitement des actions de formulaire
        if submitted:
            # Supprimer les règles marquées
            st.session_state.filter_rules = [rule for rule in st.session_state.filter_rules if
                                             not rule.get('delete', False)]

            # Ajouter la nouvelle règle si demandé
            if add_new_rule and value:
                new_rule = {
                    "field": field,
                    "operator": operator,
                    "value": value,
                    "next_op": "AND"  # Par défaut
                }
                st.session_state.filter_rules.append(new_rule)

            # Marquer que les filtres ont été appliqués et modifiés
            st.session_state.filters_applied = True
            st.session_state.filter_changed = True

            # Forcer un rechargement pour appliquer les changements
            # Remplacer st.experimental_rerun() par st.rerun() qui est la fonction moderne
            st.rerun()

    # Retourner la configuration de filtrage si applicable
    if st.session_state.filters_applied:
        # Extraire les opérateurs entre règles
        rule_ops = []
        for i in range(len(st.session_state.filter_rules) - 1):
            rule_ops.append(st.session_state.filter_rules[i].get('next_op', 'AND'))

        # Créer la configuration complète
        filter_config = {
            "rules": [
                {key: rule[key] for key in ['field', 'operator', 'value']}
                for rule in st.session_state.filter_rules
            ],
            "rule_ops": rule_ops,
            "global_op": global_op
        }

        # Réinitialiser le drapeau de changement
        if st.session_state.filter_changed:
            st.session_state.filter_changed = False
            return filter_config

        # Si les filtres n'ont pas changé, retourner une version sans le drapeau de changement
        return {**filter_config, "changed": False}
    else:
        return {
            "rules": [],
            "rule_ops": [],
            "global_op": "AND",
            "changed": False
        }


def safe_numeric_conversion(value, default=0):
    """
    Convertit une valeur en nombre de manière sécurisée, en gérant les NaN et les erreurs.

    Args:
        value: Valeur à convertir
        default: Valeur par défaut si la conversion échoue

    Returns:
        Valeur numérique ou valeur par défaut
    """
    if pd.isna(value) or (isinstance(value, (float, np.float64)) and np.isnan(value)):
        return default

    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def apply_filters(df, filter_config):
    """
    Applique les règles de filtrage au DataFrame avec logique ET/OU.

    Args:
        df: DataFrame à filtrer
        filter_config: Configuration du filtrage avec règles et opérateurs

    Returns:
        DataFrame filtré
    """
    # Si pas de changement dans les filtres et pas de règles, retourner le DataFrame d'origine
    if not filter_config.get("changed", True) and not filter_config.get("rules", []):
        return df

    rules = filter_config.get("rules", [])
    rule_ops = filter_config.get("rule_ops", [])
    global_op = filter_config.get("global_op", "AND")

    if not rules:
        return df

    filtered_df = df.copy()

    # Appliquer les filtres avec logique ET/OU
    mask = None

    for i, rule in enumerate(rules):
        field = rule['field']
        operator = rule['operator']
        value = rule['value']

        # Créer un masque pour cette règle
        current_mask = None

        # Filtres pour les champs numériques
        if field in ["Volume de recherche", "Difficulté"]:
            try:
                # Conversion sécurisée
                numeric_value = safe_numeric_conversion(value)
                # Assurer que la colonne contient des valeurs numériques
                filtered_df[field] = filtered_df[field].apply(safe_numeric_conversion)

                if operator == ">=":
                    current_mask = filtered_df[field] >= numeric_value
                elif operator == "<=":
                    current_mask = filtered_df[field] <= numeric_value
                elif operator == "=":
                    current_mask = filtered_df[field] == numeric_value
                elif operator == ">":
                    current_mask = filtered_df[field] > numeric_value
                elif operator == "<":
                    current_mask = filtered_df[field] < numeric_value
            except Exception as e:
                st.warning(f"Erreur lors du filtrage sur {field}: {str(e)}")
                continue

        # Filtres pour les champs de présence
        elif field == "Présent chez le client":
            if operator == "est":
                current_mask = filtered_df[field] == value
            else:  # n'est pas
                current_mask = filtered_df[field] != value

        # Filtres pour les champs textuels (mots-clés)
        else:  # field == "Mot clé"
            if operator == "contient":
                current_mask = filtered_df[field].str.contains(value, case=False, na=False)
            elif operator == "ne contient pas":
                current_mask = ~filtered_df[field].str.contains(value, case=False, na=False)
            elif operator == "égal à":
                current_mask = filtered_df[field].str.lower() == value.lower()
            elif operator == "commence par":
                current_mask = filtered_df[field].str.lower().str.startswith(value.lower())
            elif operator == "finit par":
                current_mask = filtered_df[field].str.lower().str.endswith(value.lower())
            elif operator == "regex":
                try:
                    current_mask = filtered_df[field].str.contains(value, regex=True, case=False, na=False)
                except re.error:
                    st.warning(f"Expression régulière invalide: {value}")
                    continue

        # Combiner avec le masque existant selon l'opérateur logique
        if current_mask is not None:
            if mask is None:
                mask = current_mask
            else:
                # Obtenir l'opérateur logique pour cette règle
                op = "AND"
                if i > 0 and (i - 1) < len(rule_ops):
                    op = rule_ops[i - 1]

                # Appliquer l'opérateur
                if op == "AND":
                    mask = mask & current_mask
                else:  # "OR"
                    mask = mask | current_mask

    # Appliquer le masque final
    if mask is not None:
        return filtered_df[mask]

    return filtered_df


def reset_filters():
    """Réinitialise tous les filtres appliqués."""
    for key in ['filter_rules', 'filters_applied', 'filter_changed']:
        if key in st.session_state:
            if key == 'filter_rules':
                st.session_state[key] = []
            else:
                st.session_state[key] = False