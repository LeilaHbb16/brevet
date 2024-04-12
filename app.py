import os
import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


def parcourir_repertoire(repertoire):
    dfs = [] 
    for dossier, sous_repertoires, fichiers in os.walk(repertoire):
        for fichier in fichiers:
            if fichier.endswith('.json'):
                chemin_fichier = os.path.join(dossier, fichier)
    
                with open(chemin_fichier, 'r') as f:
                    contenu = json.load(f)
                    # Création d'un DataFrame à partir des données JSON
                    df = pd.json_normalize(contenu)
                    # Tokenisation du texte dans toutes les colonnes
                    df_tokenise = df.applymap(tokeniser_texte)
                    dfs.append(df_tokenise)    
    return dfs

def tokeniser_texte(texte):
    # Tokenisation des mots
    tokens = word_tokenize(str(texte))
    # Suppression de la ponctuation
    tokens = [mot for mot in tokens if mot not in string.punctuation]
    # Suppression des stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [mot for mot in tokens if mot.lower() not in stop_words]
    return tokens

# Les données nécessaires pour NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Chemin du répertoire à parcourir
repertoire_a_explorer = '../brevets_alternants'

# Charger les données JSON dans une liste de DataFrames
donnees = parcourir_repertoire(repertoire_a_explorer)
