import os
import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def parcourir_repertoire(repertoire):
    # Liste pour stocker les DataFrames de chaque fichier JSON
    dfs = [] 
    for dossier, sous_repertoires, fichiers in os.walk(repertoire):
        for fichier in fichiers:
            if fichier.endswith('.json'):
                chemin_fichier = os.path.join(dossier, fichier)
                with open(chemin_fichier, 'r') as f:
                    contenu = json.load(f)
                    # Création d'un DataFrame à partir des données JSON
                    df = pd.json_normalize(contenu)
                    # Tokenisation du texte 
                    df_tokenise = df.apply(lambda x: x.map(tokeniser_texte))
                    # Nettoyage du texte
                    df_nettoye = df_tokenise.apply(lambda x: x.map(nettoyer_texte))
                    dfs.append(df_nettoye)
                print()

    return dfs

def tokeniser_texte(texte):
    # Vérification si le texte est une chaîne de caractères
    if isinstance(texte, str):
        # Tokenisation des mots
        return word_tokenize(texte)
    else:
        return texte

def nettoyer_texte(texte):
    # Suppression de la ponctuation
    if isinstance(texte, list):
        texte = [mot for mot in texte if mot not in string.punctuation]
        # Concaténation des mots dans une chaîne de caractères
        texte = ' '.join(texte)
    # Suppression des caractères spéciaux
    if isinstance(texte, str):
        texte = ''.join(caractere for caractere in texte if caractere.isalnum() or caractere.isspace())
    # Suppression des mots vides
    mots_vides = set(stopwords.words('english'))
    
    if isinstance(texte, str):
        # Convertir la chaîne de caractères en une liste de mots
        mots = texte.split()
        texte = ' '.join(mot for mot in mots if mot.lower() not in mots_vides)
    
    return texte

# Télécharger les données nécessaires pour NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Chemin du répertoire à parcourir
repertoire_a_explorer = '../brevets_alternants'

# Charger les données JSON dans une liste de DataFrames
donnees = parcourir_repertoire(repertoire_a_explorer)
