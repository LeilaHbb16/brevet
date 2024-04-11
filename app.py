import os
import json
from pprint import pprint

def parcourir_repertoire(repertoire):
    for dossier, sous_repertoires, fichiers in os.walk(repertoire):
        for fichier in fichiers:
            if fichier.endswith('.json'):
                chemin_fichier = os.path.join(dossier, fichier)
                print(f"Contenu du fichier {chemin_fichier}:")
                with open(chemin_fichier, 'r') as f:
                    contenu = json.load(f)
                    # Imprimer de manière plus lisible
                    pprint(contenu)
                print()

# Chemin du répertoire à parcourir
repertoire_a_explorer = '../brevets_alternants'

# Appel de la fonction pour parcourir le répertoire
parcourir_repertoire(repertoire_a_explorer)
