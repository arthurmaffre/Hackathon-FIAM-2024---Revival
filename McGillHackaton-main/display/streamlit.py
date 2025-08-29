import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Titre de l'application
st.title("Hackathon McGill - Portfolio Optimization")

# Sous-titre
st.subheader("Analyse des portefeuilles financiers")

# Uploader un fichier CSV
uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Lire le fichier CSV
    data = pd.read_csv(uploaded_file)

    # Afficher les premières lignes du fichier
    st.write("Aperçu des données:")
    st.dataframe(data.head())

    # Sélection des colonnes
    columns = data.columns.tolist()
    selected_columns = st.multiselect("Sélectionnez les colonnes à analyser", columns)

    # Affichage des statistiques descriptives
    if selected_columns:
        st.write(f"Statistiques descriptives des colonnes sélectionnées : {selected_columns}")
        st.write(data[selected_columns].describe())

    # Graphique interactif avec Matplotlib
    st.write("Graphique d'analyse")
    column_to_plot = st.selectbox("Sélectionnez la colonne pour le graphique", columns)
    
    if column_to_plot:
        fig, ax = plt.subplots()
        ax.plot(data[column_to_plot], label=column_to_plot)
        ax.set_xlabel("Index")
        ax.set_ylabel(column_to_plot)
        ax.legend()
        st.pyplot(fig)

# Section de sélection de l'algorithme (par exemple, pour un modèle ML)
st.sidebar.title("Options de Modèle")
model_choice = st.sidebar.selectbox(
    "Choisissez un modèle d'IA pour l'analyse de portefeuille",
    ("Aucun", "Transformer", "Réseau Neuronal", "Modèle RL")
)

# Afficher une sélection en fonction du modèle choisi
if model_choice == "Transformer":
    st.sidebar.write("Vous avez choisi : Transformer. Ce modèle est idéal pour générer des allocations de portefeuille optimales.")
elif model_choice == "Réseau Neuronal":
    st.sidebar.write("Vous avez choisi : Réseau Neuronal. Ce modèle peut prévoir les tendances du marché.")
elif model_choice == "Modèle RL":
    st.sidebar.write("Vous avez choisi : Modèle RL (Reinforcement Learning).")

# Bouton pour lancer une analyse
if st.button("Lancer l'analyse"):
    st.write("L'analyse est en cours...")
    # Tu peux ajouter ici ton code d'analyse ou d'entraînement de modèle
    st.success("Analyse terminée !")
