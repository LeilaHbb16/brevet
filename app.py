import os
import json
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


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


# Chemin vers le fichier GloVe
glove_file = './wordEmb/glove.6B.100d.txt'

# Chemin de sortie pour le fichier converti
word2vec_output_file = './wordEmb/glove.6B.100d.word2vec'

# Convertir le fichier GloVe au format word2vec
glove2word2vec(glove_file, word2vec_output_file)

# Charger les embeddings Word2Vec convertis
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)



def load_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index
def vectoriser_texte(texte, embeddings_model):
    # Initialiser une liste pour stocker les vecteurs de mots
    vecteurs = []
    # Parcourir chaque mot dans le texte tokenisé
    for mot in texte:
        # Vérifier si le mot se trouve dans le modèle d'embeddings
        if mot in embeddings_model:
            # Récupérer le vecteur correspondant au mot
            vecteur_mot = embeddings_model[mot]
            # Ajouter le vecteur du mot à la liste des vecteurs
            vecteurs.append(vecteur_mot)
    # Calculer la moyenne des vecteurs de mots
    if vecteurs:
        # Moyenne des vecteurs de mots
        representation_texte = np.mean(vecteurs, axis=0) 
    else:
        # Vecteur zéro si aucun mot n'est présent
        representation_texte = np.zeros(embeddings_model.vector_size)  
    return representation_texte


representations_vectorielles = []
for document in donnees:
    # Vectoriser le texte du document en utilisant les embeddings Word2Vec
    representation_document = vectoriser_texte(document, word2vec_model)
    # Ajouter la représentation vectorielle du document à la liste des représentations vectorielles
    representations_vectorielles.append(representation_document)
