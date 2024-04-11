import os
import json
import pandas as pd

def parcourir_repertoire(repertoire):
    # Liste pour stocker les DataFrames de chaque fichier JSON
    dfs = [] 
    for dossier, sous_repertoires, fichiers in os.walk(repertoire):
        for fichier in fichiers:
            if fichier.endswith('.json'):
                chemin_fichier = os.path.join(dossier, fichier)
                print(f"Lecture du fichier {chemin_fichier}:")
                with open(chemin_fichier, 'r') as f:
                    contenu = json.load(f)
                    # Création d'un DataFrame à partir des données JSON
                    df = pd.json_normalize(contenu)
                    dfs.append(df)
                print()

    return dfs

# Chemin du répertoire à parcourir
repertoire_a_explorer = '../brevets_alternants'

# Charger les données JSON dans une liste de DataFrames
donnees = parcourir_repertoire(repertoire_a_explorer)
