#!/usr/bin/env python3
import os
import math
import time
import logging
import pandas as pd
from multiprocessing import Pool, cpu_count

# Importez vos modules (assurez-vous que les notebooks exportés ou les modules .py sont dans le PYTHONPATH)
from graph_utilities import (
    lire_graphe,
    #generate_cyclic_graph,
    #generate_chain_graph,
    #generate_spider_graph
)
#from mtd_math import solver, solver_beta, solver_beta_dichotomic
from algo_recherche import recherche_profondeur, recherche_largeur, couverture_balle

# Configuration du logging
logging.basicConfig(
    filename='calculation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_graph(item):
    """
    Traite un graphe en appliquant tous les solveurs et renvoie un dictionnaire de résultats.
    """
    name, graphe = item
    result = {"Graphe": name}
    
    try:
        nb_sommets = len(graphe)
    except Exception as e:
        nb_sommets = None
        logging.error(f"{name} - Erreur lors de la détermination du nombre de sommets: {e}")
    result["V"] = nb_sommets
    
    try:
        k = math.sqrt(nb_sommets) if nb_sommets is not None else None
    except Exception as e:
        k = None
    result["sqrt(V)"] = k
    result["B_max"] = math.ceil(k) if k is not None else None

    # Recherche en profondeur
    try:
        start = time.perf_counter()
        res_rech_p, cout_rech_p = recherche_profondeur(graphe)
        t_rech_p = time.perf_counter() - start
        result["Recherche Profoncdeur"] = res_rech_p
        result["Cout RP"] = cout_rech_p
        result["temps RP"] = t_rech_p
        logging.info(f"{name} - recherche_profondeur terminé en {t_rech_p:.2f} sec")
    except Exception as e:
        result["Recherche Profoncdeur"] = str(e)
        result["Cout RP"] = None
        result["temps RP"] = None
        logging.error(f"{name} - Erreur dans recherche_profondeur: {e}")

    # Recherche en largeur
    try:
        start = time.perf_counter()
        etat, res_rech_l, cout_rech_l = recherche_largeur(graphe)
        t_rech_l = time.perf_counter() - start
        result["Recherche Largeur"] = res_rech_l
        result["Cout RL"] = cout_rech_l
        result["temps RL"] = t_rech_l
        logging.info(f"{name} - recherche_largeur terminé en {t_rech_l:.2f} sec")
    except Exception as e:
        result["Recherche Largeur"] = str(e)
        result["Cout RL"] = None
        result["temps RL"] = None
        logging.error(f"{name} - Erreur dans recherche_largeur: {e}")

    # Couverture des balles
    try:
        start = time.perf_counter()
        res_couv, cout_couv = couverture_balle(graphe)
        t_couv = time.perf_counter() - start
        result["Couverture des balles"] = res_couv
        result["Cout CB"] = cout_couv
        result["temps CB"] = t_couv
        logging.info(f"{name} - couverture_balle terminé en {t_couv:.2f} sec")
    except Exception as e:
        result["Couverture des balles"] = str(e)
        result["Cout CB"] = None
        result["temps CB"] = None
        logging.error(f"{name} - Erreur dans couverture_balle: {e}")

    return result




#def main():
#    start_time = time.perf_counter()
#    logging.info("Démarrage du traitement parallèle des graphes...")
#    
#    # Charger les graphes
#    graphes = {
#        #"Stranke94": lire_graphe(r'instances/Stranke94/Stranke94.mtx'), # 10 sommets (3.16)
#        #"mouse_visual-cortex_1": lire_graphe(r'instances/bn-mouse_visual-cortex_1/bn-mouse_visual-cortex_1.mtx'), # 29 sommets (5.38)
#        "karate": lire_graphe(r'instances/karate/karate.mtx'), # 34 sommets (5,83)
#        #"dolphins": lire_graphe(r'instances/dolphins/dolphins.mtx'), # 62 sommets (7.87)
#        #"polbooks": lire_graphe(r'instances/polbooks/polbooks.mtx'), # 105 sommets (12.88)
#        #"sphere3": lire_graphe(r'instances/sphere3/sphere3.mtx'), # 258 sommets (16.06)
#        #"ca-netscience": lire_graphe(r'instances/ca-netscience/ca-netscience.mtx'), # 379 sommets (19.47)
#        #"government": lire_graphe(r'instances/fb-pages-government/fb-pages-government.mtx'), # 7 057 sommets (84.005)
#        #"crocodile": lire_graphe(r'instances/web-wiki-crocodile/web-wiki-crocodile.mtx'), # 11 631 sommets (107.84)
#        #"blogcatalog": lire_graphe(r'instances/BlogCatalog/soc-BlogCatalog.mtx'), # 88 784 sommets (297.96)
#        #"gowalla_edges": lire_graphe(r'instances/loc-gowalla_edges/loc-gowalla_edges.mtx') # 196 591 sommets (443.38)
#    }
#    items = list(graphes.items())
#    
#    # Détermine le nombre de processus à utiliser
#    total_cpu = cpu_count()
#    nb_processes = max(1, math.floor(total_cpu * 0.7))
#    logging.info(f"Utilisation de {nb_processes} processus pour traiter {len(items)} graphes")
#    
#    # Traitement parallèle
#    with Pool(nb_processes) as pool:
#        results = pool.map(process_graph, items)
#    
#    # Sauvegarde des résultats dans un fichier CSV
#    df = pd.DataFrame(results)
#    output_csv = "resultats" + "nom du graphe" ".csv"
#    df.to_csv(output_csv, index=False)
#    logging.info(f"Les résultats ont été sauvegardés dans {output_csv}")
#    
#    total_time = time.perf_counter() - start_time
#    logging.info(f"Traitement terminé en {total_time:.2f} secondes")
#    print(f"Traitement terminé en {total_time:.2f} secondes")


def main():
    start_time = time.perf_counter()
    logging.info("Démarrage du traitement séquentiel des graphes...")
    
    # Charger les graphes
    graphes = {
        #"Stranke94": lire_graphe(r'instances/Stranke94/Stranke94.mtx'), # 10 sommets (3.16)
        #"mouse_visual-cortex_1": lire_graphe(r'instances/bn-mouse_visual-cortex_1/bn-mouse_visual-cortex_1.mtx'), # 29 sommets (5.38)
        #"karate": lire_graphe(r'instances/karate/karate.mtx'), # 34 sommets (5,83)
        #"dolphins": lire_graphe(r'instances/dolphins/dolphins.mtx'), # 62 sommets (7.87)
        #"polbooks": lire_graphe(r'instances/polbooks/polbooks.mtx'), # 105 sommets (12.88)
        "sphere3": lire_graphe(r'instances/sphere3/sphere3.mtx'), # 258 sommets (16.06)
        "ca-netscience": lire_graphe(r'instances/ca-netscience/ca-netscience.mtx'), # 379 sommets (19.47)
        #"government": lire_graphe(r'instances/fb-pages-government/fb-pages-government.mtx'), # 7 057 sommets (84.005)
        #"crocodile": lire_graphe(r'instances/web-wiki-crocodile/web-wiki-crocodile.mtx'), # 11 631 sommets (107.84)
        #"blogcatalog": lire_graphe(r'instances/BlogCatalog/soc-BlogCatalog.mtx'), # 88 784 sommets (297.96)
        #"gowalla_edges": lire_graphe(r'instances/loc-gowalla_edges/loc-gowalla_edges.mtx') # 196 591 sommets (443.38)
    }
    
    results = []  # Pour stocker les résultats de chaque graphe
    
    for item in list(graphes.items()):
        graph_name = item[0]
        logging.info(f"Lancement du traitement pour le graphe {graph_name}")
        result = process_graph(item)
        logging.info(f"Traitement terminé pour le graphe {graph_name}")

        df = pd.DataFrame([result])
        output_csv = os.path.join("resultats", f"{result['Graphe']}_resultats.csv")
        df.to_csv(output_csv, index=False)
        logging.info(f"Les résultats pour {result['Graphe']} ont été sauvegardés dans {output_csv}\n")

    
    total_time = time.perf_counter() - start_time
    logging.info(f"Traitement terminé en {total_time:.2f} secondes")
    print(f"Traitement terminé en {total_time:.2f} secondes")


if __name__ == "__main__":
    main()

#nohup python3 main.py > output.log 2>&1 &

# OU BIEN

#tmux new -s calcul
#python3 main.py
# puis Ctrl+B, D pour détacher
