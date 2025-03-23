from gurobipy import Model, GRB, quicksum
import math




from graph_utilities import calculer_voisinage_etendu
from graph_utilities import lire_graphe, afficher_graphe, visualiser_graphe_par_etape
from graph_utilities import generate_cyclic_graph, generate_chain_graph, generate_spider_graph





def solver(graphe):
    # Paramètres du modèle
    V = list(graphe.keys())  # Sommets du graphe
    B = math.ceil(len(V) ** 0.5)  # Longueur maximale de la séquence de brûlage

    # Création du modèle
    model = Model("GBP-IP")

    # Variables : x[v, i] pour chaque sommet v et étape i
    x = model.addVars(V, range(1, B+1), vtype=GRB.BINARY, name="x")

    # Variable pour minimiser b
    z = model.addVar(vtype=GRB.CONTINUOUS, name="z")

    # Contrainte (6) : Chaque sommet apparaît au plus une fois dans la séquence
    for v in V:
        model.addConstr(quicksum(x[v, i] for i in range(1, B+1)) <= 1, f"once_{v}")

    # Contrainte (7) : Chaque position i de la séquence a au plus un sommet assigné
    for i in range(1, B+1):
        model.addConstr(quicksum(x[v, i] for v in V) <= 1, f"position_{i}")

    # Contrainte (8) : Chaque sommet doit être brûlé au plus tard à l'étape B
    for v in V:
        model.addConstr(
            quicksum(
                quicksum(x[u, i] 
                        for u in V 
                        if v in calculer_voisinage_etendu(graphe, u, B-i))  # Voisins pouvant brûler v
                for i in range(1, B+1)
        ) >= 1, f"cover_{v}")

    # Contrainte (9) : i * \sum_{v \in V} x[v, i] <= b, \forall i \in {1, ..., B}
    for i in range(1, B+1):
        model.addConstr(i * quicksum(x[v, i] for v in V) <= z, f"minimize_z_{i}")

    # Fonction objectif : Minimiser b
    model.setObjective(z, GRB.MINIMIZE)

    model.setParam("OutputFlag", 0)  # Désactiver les logs

    model.write("solver.lp") # Écrire le modèle dans un fichier

    # Résolution du modèle
    model.optimize()

    # Affichage des résultats
    if model.status == GRB.OPTIMAL:
        #print("Solution optimale trouvée :")
        chemin = []  # Liste pour enregistrer les sommets brûlés à chaque étape
        for i in range(1, B+1):
            for v in V:
                if x[v, i].x > 0.5:  # Vérifier si x[v, i] est actif
                    chemin.append(v)  # Ajouter le sommet et l'étape au chemin
                    #print(f"Le sommet {v} est brûlé à l'étape {i}")

        #print("Chemin trouvé :", chemin)  # Afficher la séquence complète des sommets brûlés
        return chemin, len(chemin)
    else:
        #print("Pas de solution optimale trouvée.")
        return None, None





def solver_beta(graphe):
    V = list(graphe.keys())
    B_max = math.ceil(len(V) ** 0.5)  # Longueur maximale de la séquence de brûlage

    for B in range(1, B_max + 1):
        model = Model("GBP-BETA")
        x = model.addVars(V, range(1, B+1), vtype=GRB.BINARY, name="x")

        # Contrainte (6)
        for v in V:
            model.addConstr(quicksum(x[v, i] for i in range(1, B+1)) <= 1, f"once_{v}")

        # Contrainte (7)
        for i in range(1, B+1):
            model.addConstr(quicksum(x[v, i] for v in V) == 1, f"position_{i}")

        # Contrainte (8)
        for v in V:
            model.addConstr(
                quicksum(
                    x[u, i] 
                    for i in range(1, B+1) 
                    for u in V 
                    if v in calculer_voisinage_etendu(graphe, u, B - i)
                ) >= 1, 
                f"cover_{v}"
            )

        model.setParam("OutputFlag", 0)  # Désactiver les logs
        model.write("solver_gbp.lp") # Écrire le modèle dans un fichier
        model.optimize()

        # Affichage des résultats
        if model.status == GRB.OPTIMAL:
            #print(f"Solution trouvée pour B = {B}")
            chemin = []  # Liste pour enregistrer les sommets brûlés à chaque étape
            for i in range(1, B+1):
                for v in V:
                    if x[v, i].x > 0.5:  # Vérifier si x[v, i] est actif
                        chemin.append(v)  # Ajouter le sommet et l'étape au chemin

            #print("Chemin trouvé :", chemin)  # Afficher la séquence complète des sommets brûlés
            return chemin, B  # Retourner la solution optimale
    return None, None




def solver_beta_dichotomic(graphe):    
    V = list(graphe.keys())
    B_max = math.ceil(len(V) ** 0.5)

    low, high = 1, B_max
    solution = None
    B_solution = None

    while low <= high:
        mid = (low + high) // 2

        model = Model("GBP-BETA-DICO")
        # Création des variables de décision pour B = mid
        x = model.addVars(V, range(1, mid+1), vtype=GRB.BINARY, name="x")

        # Contrainte (6) : chaque sommet ne peut être brûlé qu'une seule fois
        for v in V:
            model.addConstr(quicksum(x[v, i] for i in range(1, mid+1)) <= 1, f"once_{v}")

        # Contrainte (7) : à chaque étape, exactement un sommet est brûlé
        for i in range(1, mid+1):
            model.addConstr(quicksum(x[v, i] for v in V) == 1, f"position_{i}")

        # Contrainte (8) : couverture de tous les sommets
        for v in V:
            model.addConstr(
                quicksum(
                    x[u, i]
                    for i in range(1, mid+1)
                    for u in V
                    if v in calculer_voisinage_etendu(graphe, u, mid - i)
                ) >= 1,
                f"cover_{v}"
            )

        model.setParam("OutputFlag", 0)  # Désactiver les logs
        model.write("solver_gbp.lp")     # Écriture optionnelle du modèle dans un fichier
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # Si une solution est trouvée, on enregistre le chemin et essaie de réduire B
            chemin = []
            for i in range(1, mid+1):
                for v in V:
                    if x[v, i].x > 0.5:
                        chemin.append(v)
            solution = chemin
            B_solution = mid
            high = mid - 1  # On recherche si une solution avec un B plus petit existe
        else:
            low = mid + 1  # Pas de solution pour mid, on augmente B

    return solution, B_solution





