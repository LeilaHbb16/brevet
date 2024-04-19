import os
import json
import nltk
import string
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import KFold

# Les données nécessaires pour NLTK
nltk.download('stopwords')
nltk.download('punkt')


# def parcourir_repertoire(repertoire, cles_a_supprimer):
#     dfs = [] 
#     for dossier, sous_repertoires, fichiers in os.walk(repertoire):
#         for fichier in fichiers:
#             if fichier.endswith('.json'):
#                 chemin_fichier = os.path.join(dossier, fichier)
                
#                 # print("Traitement du fichier :", chemin_fichier)
                
#                 # Obtenir le nom de la catégorie à partir du chemin du fichier
#                 categorie = os.path.basename(os.path.dirname(chemin_fichier))
    
#                 with open(chemin_fichier, 'r') as f:
#                     contenu = json.load(f)
#                     # Suppression des clés non pertinentes
#                     for cle in cles_a_supprimer:
#                         contenu.pop(cle, None)
                

#                     # Création d'un DataFrame à partir des données JSON
#                     df = pd.json_normalize(contenu)
#                     print('contenu',contenu)
#                     # Ajout de la colonne pour la catégorie
#                     df['Categorie'] = categorie
                    
#                     # Afficher le DataFrame après ajout de la colonne
#                     print("DataFrame après ajout de la catégorie :", df)
                    
#                     # Tokenisation du texte dans toutes les colonnes
#                     # df_tokenise = df.applymap(tokeniser_texte)
                  
#                     # dfs.append(df_tokenise)
                    
#     return dfs

# def parcourir_repertoire(repertoire, cles_a_supprimer):
#     dfs = [] 
#     for dossier, sous_repertoires, fichiers in os.walk(repertoire):
#         for fichier in fichiers:
#             if fichier.endswith('.json'):
#                 chemin_fichier = os.path.join(dossier, fichier)
                
#                 # Obtenir le nom de la catégorie à partir du chemin du fichier
#                 categorie = os.path.basename(os.path.dirname(chemin_fichier))
    
#                 with open(chemin_fichier, 'r') as f:
#                     contenu = json.load(f)
                    
#                     # Suppression des clés non pertinentes
#                     for cle in cles_a_supprimer:
#                         contenu.pop(cle, None)
                    
#                     # Nettoyage du texte
#                     for cle, valeur in contenu.items():
#                         if isinstance(valeur, str):

#                             contenu[cle] = re.sub(r'[\n\"]| \d+\.\s*', '', valeur).lower()
                    
#                     print(contenu)
#                     # Création d'un DataFrame à partir des données JSON
#                     df = pd.json_normalize(contenu)
                    
#                     # Ajout de la colonne pour la catégorie
#                     df['Categorie'] = categorie
                    
#                     # Afficher le DataFrame après ajout de la colonne
#                     # print("DataFrame après ajout de la catégorie :", df)
                    
#                     # Tokenisation du texte dans toutes les colonnes
#                     # df_tokenise = df.applymap(tokeniser_texte)
                  
#                     # dfs.append(df_tokenise)
                    
#     return dfs

def parcourir_repertoire(repertoire):
    dfs = []
    for dossier, sous_repertoires, fichiers in os.walk(repertoire):
        for fichier in fichiers:
            if fichier.endswith('.json'):
                chemin_fichier = os.path.join(dossier, fichier)
                
                with open(chemin_fichier, 'r') as f:
                    contenu = json.load(f)
                    # Récupération de la description et de la catégorie
                    description = contenu.get('description', None)
                    categorie = os.path.basename(os.path.dirname(chemin_fichier))
                    
                    if description is not None:
                        # Création d'un DataFrame avec description et categorie comme colonnes
                        df = pd.DataFrame({'Description': [description], 'Categorie': [categorie]})
                        
                        # Ajout du DataFrame à la liste des DataFrames
                        dfs.append(df)
                        
                    
    return dfs


# Liste des clés à supprimer
cles_a_supprimer = ['doc_number', 'country_code', 'kind_code','lang','date','classification-ipcr','claims']

# def tokeniser_texte(texte):
#     # Tokenisation des mots
#     tokens = word_tokenize(str(texte))
#     # Suppression de la ponctuation
#     tokens = [mot for mot in tokens if mot not in string.punctuation]
#     # Suppression des stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [mot for mot in tokens if mot.lower() not in stop_words]
#     # Lemmatisation des tokens
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
   
#     return tokens

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
repertoire_a_explorer = './brevets_alternants'

# Charger les données JSON dans une liste de DataFrames
donnees = parcourir_repertoire(repertoire_a_explorer)

# Affichage des valeurs des colonnes Description et Categorie pour chaque DataFrame
for df in donnees:
    print("Description :")
    for description in df['Description']:
        print(description)
    print("Categorie :", df['Categorie'].iloc[0])  # Supposant que toutes les valeurs de catégorie sont identiques pour chaque DataFrame
    print("\n")

# def load_embeddings(file_path):
#     embeddings_index = {}
#     with open(file_path, encoding='utf-8') as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             coefs = np.asarray(values[1:], dtype='float32')
#             embeddings_index[word] = coefs
#     return embeddings_index

# Chemin vers le fichier d'embeddings GloVe
embedding_file_path = './wordEmb/glove.6B.100d.txt'

# Charger les embeddings
# embeddings_index = load_embeddings(embedding_file_path)


# def vectoriser_texte(texte, embeddings_index, embedding_dim):
#     # Initialiser une liste pour stocker les vecteurs de mots
#     print("type text",type(texte))
#     vecteurs = []
#     # Parcourir chaque mot dans le texte tokenisé
#     for mot in texte:
#         # Vérifier si le mot se trouve dans les embeddings GloVe
#         print("mot",mot)
        
#         # if mot in embeddings_index:
#         #     # print("iff")
#         #     # Récupérer le vecteur GloVe correspondant
#         #     vecteur_mot = embeddings_index[mot]
#         #     # print("vecteur mot ",vecteur_mot)
#         # else:
            
#         #      # Vecteur zéro pour les mots hors vocabulaire
#         #     vecteur_mot = np.zeros(embedding_dim) 
#         #     print("else")
#         #     print("vecteur mot ",vecteur_mot)
#         # # Ajouter le vecteur du mot à la liste des vecteurs
#         # vecteurs.append(vecteur_mot)
#     # Agréger les vecteurs de mots pour obtenir une représentation vectorielle du texte
#     if vecteurs:
#         # Moyenne des vecteurs de mots
#         representation_texte = np.mean(vecteurs, axis=0) 
#     else:
#         # Vecteur zéro si aucun mot n'est présent
#         representation_texte = np.zeros(embedding_dim)  
#     return representation_texte


#Taille de dimension des embeddings GloVe
# embedding_dim = 100 

# representations_vectorielles = []
# for document in donnees:
#     # print("typeeeeeee",type(donnees[document]))
#     # print("type documt",type(document))
#     # print("typeee donnee",type(donnees))

#     # Vectorisation du texte du document en utilisant les embeddings GloVe
#     representation_document = vectoriser_texte(donnees[document], embeddings_index, embedding_dim)
#     # Ajout de la représentation vectorielle du document à la liste des représentations vectorielles
#     representations_vectorielles.append(representation_document)

    # print(representations_vectorielles)


# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# for fold, (train_index, val_index) in enumerate(kf.split(representations_vectorielles), 1):
#     # Diviser les données en ensembles d'entraînement et de validation
#     X_train, X_val = [representations_vectorielles[i] for i in train_index], [representations_vectorielles[i] for i in val_index]