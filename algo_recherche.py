import random
import math
import heapq
import time
import matplotlib.pyplot as plt
from collections import deque, defaultdict


from graph_utilities import calculer_voisinage_etendu
from graph_utilities import lire_graphe, afficher_graphe, visualiser_graphe_par_etape
from graph_utilities import generate_cyclic_graph, generate_chain_graph, generate_spider_graph



def test_but(etat_actuel):
    """
    Vérifie si tous les sommets sont brûlés.
    """
    return all(val == 1 for val in etat_actuel.values())


# ## 1ère méthode



def choisir_sommet_aleatoire(etat_actuel):
    sommets_non_brules = [sommet for sommet, etat in etat_actuel.items() if etat == 0]
    return random.choice(sommets_non_brules) if sommets_non_brules else None




def successeurs_p(graphe, etat_actuel):
    """
    Génère l'état suivant en propageant la brûlure aux voisins des sommets déjà brûlés.
    :param graphe: Le graphe sous forme de dictionnaire.
    :param etat_actuel: Dictionnaire contenant l'état actuel des sommets (brûlés ou non brûlés).
    :return: Nouveau dictionnaire représentant l'état des sommets après propagation.
    """

    brulage = False  # Indique si un sommet a été brûlé dans cet appel de la fonction

    # Copier l'état actuel pour générer le nouvel état
    nouvel_etat = etat_actuel.copy()

    # Récupérer tous les sommets brûlés
    sommets_brules = [sommet for sommet, etat in nouvel_etat.items() if etat == 1]

    # Propager la brûlure aux voisins des sommets brûlés
    for sommet in sommets_brules:
        for voisin in graphe.get(sommet, []):  # Obtenir les voisins dans la liste d'adjacence
            if nouvel_etat[voisin] == 0:  # Brûler uniquement les sommets non brûlés
                nouvel_etat[voisin] = 1
                brulage = True

    return nouvel_etat, brulage



def choisir_sommet_a_bruler(graphe, etat_actuel):
    """
    Sélectionne un sommet non brûlé ayant le maximum de voisins.

    :param graphe: Le graphe sous forme de dictionnaire (liste d'adjacence).
    :param etat_actuel: Dictionnaire contenant l'état actuel des sommets.
    :return: Le sommet non brûlé avec le maximum de voisins, ou None si aucun sommet disponible.
    """
    # Trouver les sommets non brûlés
    sommets_non_brules = [sommet for sommet, etat in etat_actuel.items() if etat == 0]

    if not sommets_non_brules:
        # Aucun sommet non brûlé
        return None

    # Trouver le sommet avec le maximum de voisins
    sommet_max_voisins = max(sommets_non_brules, key=lambda sommet: len(graphe.get(sommet, [])))

    return sommet_max_voisins



def recherche_profondeur(graphe):
    """
    Algorithme générique de recherche pour brûler un graphe en choisissant un sommet par étape.
    :param graphe: Le graphe sous forme de dictionnaire (liste d'adjacence).
    :param etat_initial: Dictionnaire représentant l'état initial des sommets (brûlés ou non brûlés).
    :param successeurs: Fonction qui génère l'état suivant (propagation).
    :param test_but: Fonction qui vérifie si tous les sommets sont brûlés.
    :return: Chemin (liste des états successifs), sommets brûlés activement à chaque étape, et coût total.
    """
    # Initialisation : créer la liste des états à traiter
    etat_initial = {sommet: 0 for sommet in graphe}
    etats_a_traiter = deque([{"etat": etat_initial, "cout": 1, "brules_actifs": []}])

    while etats_a_traiter:
        # Extraire un état
        noeud = etats_a_traiter.pop()

        noeud["etat"], brulage = successeurs_p(graphe, noeud["etat"])

        if brulage:
            noeud["cout"] += 1  # Chaque étape coûte 1

        # Vérifier si tous les sommets sont brûlés
        if test_but(noeud["etat"]):
            return noeud["brules_actifs"], noeud["cout"]

        # Choisir un nouveau sommet à brûler activement
        nouveau_sommet = choisir_sommet_a_bruler(graphe, noeud["etat"])
        #nouveau_sommet = choisir_sommet_aleatoire(noeud["etat"])
        if nouveau_sommet is None:
            # Si aucun sommet à brûler n'est disponible, retourner l'état actuel
            continue

        # Copier l'état courant
        etat_suivant = noeud["etat"].copy()

        # Marquer le sommet comme "brûlé"
        etat_suivant[nouveau_sommet] = 1

        # Ajouter le nouvel état à la liste des états à traiter
        etats_a_traiter.append({
            "etat": etat_suivant,
            "cout": noeud["cout"],  # Chaque étape coûte 1
            "brules_actifs": noeud["brules_actifs"] + [nouveau_sommet]
        })

    # Si aucun état final n'est trouvé
    return [], float("inf")



# # 2éme méthode



def propager(graphe, etat_actuel, cout_actuel):
    """
    Propage la brûlure aux voisins des sommets déjà brûlés (un "pas" de propagation),
    en incrémentant le coût à chaque fois qu'un nouveau sommet est enflammé.

    Retourne (nouvel_etat, cout_apres_propagation).
    """
    nouvel_etat = etat_actuel.copy()

    # Récupérer tous les sommets brûlés
    sommets_brules = [s for s, etat in nouvel_etat.items() if etat == 1]

    # Propager la brûlure aux voisins
    for sommet in sommets_brules:
        for voisin in graphe.get(sommet, []):
            if nouvel_etat[voisin] == 0:  # s’il n’était pas brûlé
                nouvel_etat[voisin] = 1
                #cout_actuel += 1  # Incrémenter le coût de propagation

    return nouvel_etat, cout_actuel



def successeurs(graphe, etat_actuel, cout_actuel):
    """
    Génère TOUS les états possibles en allumant manuellement
    chaque sommet non brûlé, puis en propageant le feu.

    Pour chacun, on renvoie le triplet (action, nouvel_etat, cout_action),
    où:
      - action       = le sommet qu'on a allumé
      - nouvel_etat  = l'état après allumage + propagation
      - cout_action  = le coût "supplémentaire" induit par cet allumage (et sa propagation)
    """
    liste_succ = []

    # Pour chaque sommet non brûlé, on simule "allumer ce sommet"
    for sommet in graphe:
        if etat_actuel[sommet] == 0:
            # 1) On copie l'état
            nouvel_etat = etat_actuel.copy()

            # 3) On propage la brûlure (un "pas" de propagation)
            nouvel_etat_propage, nouveau_cout = propager(graphe, nouvel_etat, cout_actuel)

            # 2) On allume manuellement ce sommet
            nouvel_etat_propage[sommet] = 1

            # 4) Le "coût d'action" = la différence entre le coût après et avant
            #    (autrement dit, combien on a dû payer en plus pour cet allumage + propagation)
            #cout_action = nouveau_cout - cout_actuel
            cout_action = 1

            # 5) On ajoute ce successeur à la liste
            liste_succ.append( (sommet, nouvel_etat_propage, cout_action) )

    return liste_succ



def recherche_largeur(graphe):
    """
    Traduit directement le pseudo-code "Algorithme 1" en Python,
    en utilisant la fonction 'successeurs' (qui retourne des triplets)
    et 'test_but'.

    :param graphe:        dictionnaire {sommet: [voisins]}
    :param etat_initial:  dictionnaire {sommet: 0/1}
    :param successeurs:   fonction(etat, cout) -> liste de (action, etat_suivant, cout_action)
    :param test_but:      fonction(etat) -> bool
    :return: (etat_solution, chemin_actions, cout_total) 
             ou (None, [], float("inf")) si on ne trouve pas de solution
    """
    etat_initial = {sommet: 0 for sommet in graphe}

    # 1) Construire le nœud initial = (état=etat_initial, chemin=[], coût=0)
    noeud_initial = {
        "etat": etat_initial,
        "chemin": [],
        "cout": 0
    }

    # 2) Mettre le nœud initial dans une file FIFO
    file = deque([noeud_initial])

    # 3) Tant que la file n'est pas vide
    while file:
        noeud = file.popleft()
        etat_courant = noeud["etat"]
        chemin_courant = noeud["chemin"]
        cout_courant = noeud["cout"]

        # -- Test but --
        if test_but(etat_courant):
            return etat_courant, chemin_courant, cout_courant

        # -- Parcourir tous les successeurs --
        #    successeurs(...) doit retourner [(action, etat_suivant, cout_action), ...]
        for (action, etat_suivant, cout_action) in successeurs(graphe, etat_courant, cout_courant):
            # Nouveau coût
            nouveau_cout = cout_courant + cout_action
            # Nouveau chemin
            nouveau_chemin = chemin_courant + [action]

            # Créer le noeud successeur
            noeud_suivant = {
                "etat": etat_suivant,
                "chemin": nouveau_chemin,
                "cout": nouveau_cout
            }
            # L'insérer dans la file
            file.append(noeud_suivant)

    # 4) Si on sort de la boucle, pas de solution trouvée
    return None, [], float("inf")



# Couverture des balles


def bruler(graphe, u, r):
    """
    Effectue une BFS depuis le sommet u en limitant la profondeur à r.
    Retourne l'ensemble des sommets brûlés (atteints).
    """
    visited = {u}
    queue = deque([(u, 0)])  # (sommet, profondeur)
    while queue:
        current, depth = queue.popleft()
        if depth < r:
            for neighbor in graphe.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
    return visited




def iterations_brulage_totale(graphe, chemin):
    """
    Retourne le nombre d'itérations nécessaires pour que tous les sommets du graphe soient brûlés.
    Optimisé pour de grands graphes en utilisant des entiers pour représenter l'état :
      0 : non brûlé
      1 : en attente de brûlage (jaune)
      2 : brûlé (rouge)

    :param graphe: Dictionnaire représentant le graphe (liste d'adjacence).
    :param chemin: Liste des sommets choisis comme sources de feu.
    :return: Nombre d'itérations pour brûler entièrement le graphe.
    """
    total = len(graphe)
    # Initialisation: tous les sommets sont en état 0 (non brûlés)
    state = {sommet: 0 for sommet in graphe}
    burned = set()   # Ensemble des sommets brûlés (état 2)
    to_burn = set()  # Ensemble des sommets en attente de brûlage (état 1)
    iteration = 0

    while len(burned) < total:
        iteration += 1
        new_burned = set()

        # Ajout de la source externe pour cette itération, si disponible
        if iteration <= len(chemin):
            source = chemin[iteration - 1]
            if state[source] != 2:
                state[source] = 2
                burned.add(source)
                new_burned.add(source)

        # Les sommets en attente (état 1) deviennent brûlés (état 2)
        if to_burn:
            new_burned |= to_burn
            for node in to_burn:
                state[node] = 2
            burned |= to_burn
            to_burn.clear()

        # Propagation du feu : depuis chaque nouveau sommet brûlé, colorier les voisins non brûlés
        for node in new_burned:
            for voisin in graphe[node]:
                if state[voisin] == 0:
                    state[voisin] = 1
                    to_burn.add(voisin)

    return iteration




def couverture_balle(graphe):
    """
    Cherche une séquence de balles (centre, rayon) qui couvre entièrement le graphe.

    Contraintes :
      - Les rayons possibles sont 1, 2, ..., ⌈√n⌉ où n = nombre de sommets.
      - Un même rayon ne peut être utilisé qu'une seule fois dans une solution.
      - On explore d'abord les grandes valeurs de rayon (pour potentiellement réduire
        rapidement le nombre de balles et/ou le rayon maximal utilisé).

    Retourne la séquence des balles (centre, rayon) de la solution trouvée.
    """

    n = len(graphe)
    V = list(graphe.keys())
    best_sequence = None
    best_max_r = float('inf')
    best_num_balls = float('inf')
    max_rayon_possible = math.ceil(math.sqrt(n))
    #max_rayon_possible = round(math.sqrt(n))

    # File de priorité (heap) avec pour chaque état :
    # (heuristique, rayon_max_actuel, nombre_de_balles, sommets_brulés, séquence, rayons_utilisés)
    heap = []
    heapq.heappush(heap, (0, 0, 0, set(), [], set()))

    # Dictionnaire de mémorisation pour éviter de revisiter des états moins optimaux
    memo = defaultdict(lambda: (float('inf'), float('inf')))

    while heap:
        heur, current_max_r, current_num, brules, seq, used_r = heapq.heappop(heap)

        # Élagage : si l'état courant est moins bon que la meilleure solution trouvée, on passe
        # On passe cette branche si l'on sait déjà qu'elle est moins bonne que la meilleure solution trouvée
        if current_max_r > best_max_r or (current_max_r == best_max_r and current_num >= best_num_balls):
            continue

        # Si tous les sommets sont brûlés, on met à jour la meilleure solution
        if len(brules) == n:
            if current_max_r < best_max_r or (current_max_r == best_max_r and current_num < best_num_balls):
                best_max_r, best_num_balls, best_sequence = current_max_r, current_num, seq
            continue

        # Pour chaque sommet non brûlé, on essaie de l'étendre avec différents rayons
        for v in V:
            if v not in brules:
                # On ne considère pas un rayon supérieur à celui maximum autorisé
                # ni supérieur à best_max_r (pour rester dans une solution potentiellement optimale)
                max_possible_r = min(max_rayon_possible, best_max_r)
                # On explore les grands rayons en premier
                for r in range(max_possible_r - 1, -1, -1): # source de problèèèèèmes
                    # faire un vecteur de rayons utilisés
                    # used_r = [0 for _ in range(max_rayon_possible + 1)]
                    # used_r = [1,1,0,1]
                    if r in used_r:
                        continue  # Ce rayon a déjà été utilisé dans la séquence courante

                    # Calcul à la demande des sommets brûlés par la balle (v, r)
                    burned_with_ball = bruler(graphe, v, r)
                    nouveaux_brules = brules | burned_with_ball
                    new_max_r = max(current_max_r, r)
                    new_num = current_num + 1

                    # Clé pour mémorisation : combinaison des sommets brûlés et des rayons utilisés
                    key = (frozenset(nouveaux_brules), frozenset(used_r | {r}))
                    if (new_max_r, new_num) >= memo[key]:
                        continue
                    memo[key] = (new_max_r, new_num)

                    # Heuristique : rayon maximal utilisé + fraction des sommets restants
                    # l'algorithme donne plus de poids à la minimisation du rayon maximal, mais prend aussi en compte la progression.
                    remaining = n - len(nouveaux_brules)
                    heuristic = new_max_r + (remaining / n)
                    heapq.heappush(heap, (heuristic, new_max_r, new_num, nouveaux_brules, seq + [(v, r)], used_r | {r}))


    # Optionnel : trier la séquence par rayon décroissant (similaire à la version initiale)
    res = sorted(best_sequence, key=lambda x: x[1], reverse=True)
    #res = [(center, rayon + 1) for center, rayon in res]
    bn = iterations_brulage_totale(graphe, [center for center, rayon in res])
    return res, bn




# ## Comparaison des heuristiques

# ### Heuristique multiplicative



def heuristique_multiplicative(current_max_r, current_num, brules, n, graphe):
    """
    Heuristique par défaut :
      - current_max_r : le rayon maximal utilisé jusqu'à présent.
      - brules : l'ensemble des sommets déjà "brûlés".
      - n : le nombre total de sommets.

    Renvoie une valeur heuristique qui combine le rayon maximal et
    la fraction des sommets non encore couverts.
    """
    remaining = n - len(brules)
    return current_max_r + (1 + (remaining / n))


# ### Heuristique quadratique sur la couverture restante



def heuristique_quadratique(current_max_r, current_num, brules, n, graphe):
    """
    Calcule l'heuristique en utilisant un terme quadratique sur la fraction restante.
      h = current_max_r + (remaining/n)²
    Cette formulation peut favoriser les états ayant couvert une partie importante du graphe.
    """
    remaining = n - len(brules)
    return current_max_r + (remaining / n) ** 2


# ### Heuristique combinant rayon et nombre de balles utilisées



def heuristique_combine(current_max_r, current_num, brules, n, graphe):
    """
    Heuristique combinée qui prend en compte :
      - Le rayon maximal utilisé,
      - Le nombre de balles posées,
      - La fraction de sommets non couverts.
    Le choix des coefficients (ici 0.5) peut être ajusté selon l'importance que vous souhaitez donner à chaque critère.
      h = current_max_r + 0.5 * current_num + (remaining / n)
    """
    remaining = n - len(brules)
    return current_max_r - 0.5 * current_num + (remaining / n)


# ### Heuristique basée sur le degré moyen des sommets non couverts



def heuristique_degre(current_max_r, current_num, brules, n, graphe):
    """
    Calcule l'heuristique en tenant compte du degré moyen des sommets non couverts.

    L'idée est que si les sommets restants ont un degré élevé, ils
    devraient être plus faciles à couvrir. On pénalise donc moins ces états.

    Paramètres :
      - current_max_r : le rayon maximal utilisé jusqu'à présent.
      - brules : ensemble des sommets déjà couverts.
      - n : nombre total de sommets.
      - graphe : dictionnaire représentant le graphe (pour accéder aux voisins).

    Retourne une valeur heuristique.
    """
    remaining = set(graphe.keys()) - brules
    if not remaining:
        return current_max_r
    avg_deg = sum(len(graphe[v]) for v in remaining) / len(remaining)
    # On pénalise moins si le degré moyen est élevé
    penalty = (len(remaining) / n) / (avg_deg + 1)
    return current_max_r + penalty


# ### fonction couverture de balle adapté



def couverture_balle_beta(graphe, heuristique_func):
    """
    Cherche une séquence de balles (centre, rayon) qui couvre entièrement le graphe.

    Contraintes :
      - Les rayons possibles sont 1, 2, ..., ⌈√n⌉ où n = nombre de sommets.
      - Un même rayon ne peut être utilisé qu'une seule fois dans une solution.
      - On explore d'abord les grandes valeurs de rayon.

    Paramètres :
      - graphe : dictionnaire représentant le graphe.
      - heuristique_func : fonction qui calcule l'heuristique à partir de (current_max_r, brules, n).

    Retourne la séquence des balles (centre, rayon) de la solution trouvée.
    """
    n = len(graphe)
    V = list(graphe.keys())
    best_sequence = None
    best_max_r = float('inf')
    best_num_balls = float('inf')
    max_rayon_possible = math.ceil(math.sqrt(n))

    # File de priorité (heap) contenant pour chaque état :
    # (heuristique, rayon_max_actuel, nombre_de_balles, sommets_brulés, séquence, rayons_utilisés)
    heap = []
    heapq.heappush(heap, (0, 0, 0, set(), [], set()))

    # Dictionnaire de mémorisation pour éviter de revisiter des états moins optimaux
    memo = defaultdict(lambda: (float('inf'), float('inf')))

    while heap:
        heur, current_max_r, current_num, brules, seq, used_r = heapq.heappop(heap)

        # Élagage : on passe si l'état courant est moins bon que la meilleure solution trouvée
        if current_max_r > best_max_r or (current_max_r == best_max_r and current_num >= best_num_balls):
            continue

        # Si tous les sommets sont brûlés, on met à jour la meilleure solution
        if len(brules) == n:
            if current_max_r < best_max_r or (current_max_r == best_max_r and current_num < best_num_balls):
                best_max_r, best_num_balls, best_sequence = current_max_r, current_num, seq
            continue

        # Pour chaque sommet non brûlé, on essaie de l'étendre avec différents rayons
        for v in V:
            if v not in brules:
                max_possible_r = min(max_rayon_possible, best_max_r)
                # On explore les grands rayons en premier
                for r in range(max_possible_r - 1, -1, -1):
                    if r in used_r:
                        continue  # Ce rayon a déjà été utilisé dans la séquence courante

                    # Calcul à la demande des sommets brûlés par la balle (v, r)
                    burned_with_ball = bruler(graphe, v, r)  # fonction supposée définie ailleurs
                    nouveaux_brules = brules | burned_with_ball
                    new_max_r = max(current_max_r, r)
                    new_num = current_num + 1

                    # Clé de mémorisation : combinaison des sommets brûlés et des rayons utilisés
                    key = (frozenset(nouveaux_brules), frozenset(used_r | {r}))
                    if (new_max_r, new_num) >= memo[key]:
                        continue
                    memo[key] = (new_max_r, new_num)

                    # Utilisation de la fonction d'heuristique passée en argument
                    heuristic = heuristique_func(new_max_r, new_num, nouveaux_brules, n, graphe)
                    heapq.heappush(heap, (heuristic, new_max_r, new_num, nouveaux_brules, seq + [(v, r)], used_r | {r}))

    # Optionnel : trier la séquence par rayon décroissant et ajuster le rayon (si nécessaire)
    res = sorted(best_sequence, key=lambda x: x[1], reverse=True)
    return res



# ## couverture_balle_timing



def couverture_balle_beta_timing(graphe, heuristique_func):
    """
    Retourne la séquence finale (liste des couples (sommet, rayon+1)) et une liste 
    'burning_progress' contenant, pour chaque itération, un tuple 
    (temps, current_max_r, best_max_r, current_burning, best_burning)
    qui permet de suivre l'évolution, en parallèle, du burning number (current_burning)
    et du burning number basé sur les rayons (current_max_r).

    - temps : temps écoulé depuis le démarrage.
    - current_max_r : valeur utilisée dans les conditions (rayon maximum rencontré).
    - best_max_r : meilleure valeur complète (selon les conditions sur les rayons).
    - current_burning : burning number candidat de la séquence courante,
      calculé comme max(i + (r+1)) pour chaque boule placée.
    - best_burning : meilleur burning number candidat trouvé pour une solution complète.
    """

    start_time = time.time()
    burning_progress = []
    n = len(graphe)
    V = list(graphe.keys())

    best_sequence = None
    best_max_r = float('inf')
    best_num_balls = float('inf')
    best_burning = float('inf')
    max_rayon_possible = math.ceil(math.sqrt(n))

    # Chaque état est un tuple :
    # (heuristique, current_max_r, current_num, current_burning, brules, seq, used_r)
    heap = []
    heapq.heappush(heap, (0, 0, 0, 0, set(), [], set()))
    memo = defaultdict(lambda: (float('inf'), float('inf')))

    while heap:
        heur, current_max_r, current_num, current_burning, brules, seq, used_r = heapq.heappop(heap)
        current_time = time.time() - start_time

        # Enregistrement de la progression avec le burning number candidat en parallèle
        #burning_progress.append((current_time, current_max_r, best_max_r, current_burning, best_burning))
        burning_progress.append((current_time, current_burning, best_burning))

        # On conserve les conditions sur best_max_r (et best_num_balls)
        if current_max_r > best_max_r or (current_max_r == best_max_r and current_num >= best_num_balls):
            continue

        # Solution complète : mise à jour de best_max_r et best_burning
        if len(brules) == n:
            if current_max_r < best_max_r or (current_max_r == best_max_r and current_num < best_num_balls):
                best_max_r = current_max_r
                best_num_balls = current_num
                best_sequence = seq
                #best_burning = current_burning
                best_burning = new_burning + 1
            continue

        # Exploration des possibilités
        for v in V:
            if v not in brules:
                max_possible_r = min(max_rayon_possible, best_max_r)
                for r in range(max_possible_r - 1, -1, -1):
                    if r in used_r:
                        continue

                    # La fonction 'bruler' doit être définie et retourner l'ensemble des sommets brûlés
                    burned_with_ball = bruler(graphe, v, r)
                    nouveaux_brules = brules | burned_with_ball
                    new_max_r = max(current_max_r, r)
                    new_num = current_num + 1
                    # Mise à jour du burning number candidat : la nouvelle boule (placée à l'indice current_num)
                    # contribue par current_num + r
                    new_burning = new_max_r + 1
                    #new_burning = max(current_burning, current_num + r)

                    key = (frozenset(nouveaux_brules), frozenset(used_r | {r}))

                    if (new_max_r, new_num) >= memo[key]:
                        continue
                    memo[key] = (new_max_r, new_num)

                    candidate_seq = seq + [(v, r)]
                    heuristic = heuristique_func(new_max_r, new_num, nouveaux_brules, n, graphe)
                    heapq.heappush(heap, (heuristic, new_max_r, new_num, new_burning, nouveaux_brules, candidate_seq, used_r | {r}))

    final_time = time.time() - start_time
    #burning_progress.append((final_time, best_burning, best_burning))

    if best_sequence is None:
        return [], burning_progress

    res = sorted(best_sequence, key=lambda x: x[1], reverse=True)
    return res, burning_progress




def plot_multiple_burning_progress(burning_progress_list, labels):
    """
    Affiche la progression du burning number pour plusieurs heuristiques sur le même graphique.

    Paramètres :
      - burning_progress_list : liste de listes de tuples (temps, current_burning, best_burning)
                                Chaque élément correspond aux données d'une heuristique.
      - labels : liste de labels (chaîne de caractères) associée à chaque heuristique.
    """
    plt.figure(figsize=(20, 6))

    for bp, label in zip(burning_progress_list, labels):
        # Extraction des données
        times = [t for t, current, best in bp]
        current_values = [current for t, current, best in bp]
        best_values = [best for t, current, best in bp]


        # Trace la courbe et récupère l'objet Line2D pour obtenir la couleur utilisée
        line, = plt.plot(times, current_values, label=label, linestyle='--', marker='o', markersize=1)
        color = line.get_color()
        plt.plot(times, best_values, linestyle='-', marker='o', markersize=10, color=color)        

        # Marquer le dernier point avec un marqueur plus gros et de la même couleur
        #if times and best_values:
        #    plt.plot(times[-1], best_values[-1], marker='o', color=color, markersize=15)

    plt.xlabel('Temps (secondes)')
    plt.ylabel('Burning Number')
    plt.title('Progression du Burning Number pour différentes heuristiques')
    plt.legend()
    plt.grid(True)
    plt.show()




def comparaison_heuristiques(graphe):
    """
    Compare les différentes heuristiques sur un même graphe.
    """
    heuristiques = [heuristique_multiplicative, heuristique_quadratique, heuristique_combine, heuristique_degre]
    labels = ['Multiplicative', 'Quadratique', 'Combine', 'Degré']
    burning_progress_list = []

    for heuristique in heuristiques:
        _, burning_progress = couverture_balle_beta_timing(graphe, heuristique)
        burning_progress_list.append(burning_progress)

    plot_multiple_burning_progress(burning_progress_list, labels)




def plot_burning_progress_adatpted(burning_progress):
    """
    Affiche la progression du burning number à partir de la liste burning_progress.
    La fonction adapte automatiquement l'échelle du temps en fonction des données.

    burning_progress : liste de tuples (temps, current_max_r, best_max_r)
    """
    # Extraction des données
    times = [t for t, current, best in burning_progress]
    current_values = [current for t, current, best in burning_progress]
    best_values = [best for t, current, best in burning_progress]

    # Détermination de l'échelle adaptée
    max_time = max(times) if times else 0
    if max_time < 1:
        factor = 1000
        time_unit = "ms"
    elif max_time < 60:
        factor = 1
        time_unit = "s"
    elif max_time < 3600:
        factor = 1/60
        time_unit = "min"
    else:
        factor = 1/3600
        time_unit = "h"

    times_scaled = [t * factor for t in times]

    # Création du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(times_scaled, current_values, label='Burning Number courant', linestyle='--', marker='o')
    plt.plot(times_scaled, best_values, label='Meilleur Burning Number', linestyle='-', marker='o')

    plt.xlabel(f'Temps ({time_unit})')
    plt.ylabel('Burning Number')
    plt.title('Progression du Burning Number au cours du temps')
    plt.legend()
    plt.grid(True)
    plt.show()


# - imposer un limite de 10s -> faire la recherche pendant max 10s
# - tester pour diff periode de temps, arreter après avoir passer cette periode de temps, et voir au bout de quel moment arrive à un solution optimale (ou meme comparer la qté de a solution)
# 
# => figure : 
# - x : periode d'exec
# - y : burning number
# - clé : diff graphe
# 
# 
# (HPC serveur frontal de l'isima => faire assez de calcule en paralelle qu'on veux)

# - bruler le graphe à un certain rayon et calculer ça connectivité 
# - calculer le nbr de composant connexe ( à chercher ça def)
# - diametre = le nbr de sommet entre les deux sommet les plus éloigné du graphe

# objectif : relaxation linéaire pour les petits graphes
