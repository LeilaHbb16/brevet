import os
import json
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Les données nécessaires pour NLTK
nltk.download('stopwords')
nltk.download('punkt')

def parcourir_repertoire(repertoire, cles_a_supprimer):
    dfs = [] 
    for dossier, sous_repertoires, fichiers in os.walk(repertoire):
        for fichier in fichiers:
            if fichier.endswith('.json'):
                chemin_fichier = os.path.join(dossier, fichier)
    
                with open(chemin_fichier, 'r') as f:
                    contenu = json.load(f)
                    
                    # Suppression  les clés non pertinentes
                    for cle in cles_a_supprimer:
                        contenu.pop(cle, None)
                    
                    # Création d'un DataFrame à partir des données JSON
                    df = pd.json_normalize(contenu)
                    # Tokenisation du texte dans toutes les colonnes
                    df_tokenise = df.applymap(tokeniser_texte)
                    dfs.append(df_tokenise)    
    return dfs
# Liste des clés à supprimer
cles_a_supprimer = ['doc_number', 'country_code', 'kind_code','lang','date']

def tokeniser_texte(texte):
    # Tokenisation des mots
    tokens = word_tokenize(str(texte))
    # Suppression de la ponctuation
    tokens = [mot for mot in tokens if mot not in string.punctuation]
    # Suppression des stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [mot for mot in tokens if mot.lower() not in stop_words]
     # Lemmatisation des tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
   
    return tokens

# Chemin du répertoire à parcourir
repertoire_a_explorer = '../brevets_alternants'

# Charger les données JSON dans une liste de DataFrames
donnees = parcourir_repertoire(repertoire_a_explorer, cles_a_supprimer)

def load_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Chemin vers le fichier d'embeddings GloVe
embedding_file_path = './wordEmb/glove.6B.100d.txt'

# Charger les embeddings
embeddings_index = load_embeddings(embedding_file_path)
