#!/bin/bash
set -e

# Répertoire cible pour le projet GitHub et les données
PROJECT_DIR="/teamspace/studios/this_studio/McGillHackaton"
TARGET_DIR="$PROJECT_DIR/data/raw_data"
ZIP_FILE="$TARGET_DIR/dataset.zip"

# ID Google Drive du fichier ZIP
GDRIVE_ID="1bPptsraz6lZ3lIYJQVbKrzCe8y-dE80Y"

# URL du dépôt git
REPO_URL="https://github.com/Thomas4390/McGillHackaton"

# Installer gdown si nécessaire pour télécharger depuis Google Drive
if ! command -v gdown &> /dev/null; then
    echo "gdown non installé, installation en cours"
    pip install gdown
fi

# Installer unzip si nécessaire
if ! command -v unzip &> /dev/null; then
    echo "unzip non installé, installation en cours"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y unzip
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install unzip
    else
        echo "Système d'exploitation non supporté pour l'installation automatique de unzip"
        exit 1
    fi
fi

# Cloner le projet depuis GitHub si le dossier du projet n'existe pas
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Clonage du projet GitHub depuis $REPO_URL..."
    git clone $REPO_URL $PROJECT_DIR
else
    echo "Le projet GitHub existe déjà dans $PROJECT_DIR"
    cd $PROJECT_DIR
    git pull  # Met à jour le dépôt si déjà cloné
fi

# Créer le répertoire cible pour les données s'il n'existe pas
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p $TARGET_DIR
fi

# Télécharger le fichier ZIP depuis Google Drive si non présent
if [ ! -f "$ZIP_FILE" ]; then
    echo "Téléchargement du fichier dataset.zip depuis Google Drive..."
    gdown --id $GDRIVE_ID --output $ZIP_FILE
else
    echo "Le fichier ZIP existe déjà à l'emplacement $ZIP_FILE"
fi

# Extraire les fichiers directement dans le répertoire 'raw_data'
echo "Extracting dataset.zip into $TARGET_DIR..."
unzip -o $ZIP_FILE -d $TARGET_DIR

# Check if a 'dataset' directory was created
if [ -d "$TARGET_DIR/dataset" ]; then
    echo "Moving files from 'dataset' to $TARGET_DIR"
    mv $TARGET_DIR/dataset/* $TARGET_DIR/
    rm -r $TARGET_DIR/dataset
fi

# Supprimer l'archive ZIP après extraction
echo "Suppression du fichier ZIP..."
rm -f $ZIP_FILE

# Installation des dépendances si nécessaire
REQ_FILE="$PROJECT_DIR/requirements.txt"
if [ -f $REQ_FILE ]; then
    echo "Installation des dépendances depuis $REQ_FILE"
    pip install -r $REQ_FILE
else
    echo "Fichier requirements.txt non trouvé, aucune dépendance à installer"
fi
