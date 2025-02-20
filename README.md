mkdir burning_graph
cd burning_graph

git remote add origin https://github.com/OussamaLay/Graph_Burning_Problem.git

switch to aya branch : git checkout aya

push : git push -u origin aya


python -m venv .venv

.venv/Scripts/activate

pip install -r requirements.txt

Pour voir le fichier `diagUML.dot` est utilisé pour dessiner le diagramme de classe, en utilisant l'extention `DotUML`.

Il suffit d'installer l'extention `DotUML` sur Visual Studio Code, ouvrir le fichier `diagUML.dot`, clique droit sur le code, cliquer sur `Open DotUML Preview to the side`, et puis une fenètre affichera le diagramme complet.


```plaintext
fonction couverture_balle(graphe):
    n ← nombre de sommets de graphe
    max_rayon_possible ← ceil(sqrt(n))
    best_seq ← None, best_max ← +∞, best_count ← +∞
    heap ← [(0, 0, 0, ∅, [], ∅)]   // (heur, max_rayon, nb_balles, brûlés, séquence, rayons utilisés)
    memo ← dictionnaire vide

    tant que heap n'est pas vide:
        (heur, max_r, count, brûlés, seq, used) ← extraire_min(heap)
        si (max_r > best_max) ou (max_r == best_max et count ≥ best_count): continuer
        si taille(brûlés) == n:
            best_seq ← seq; best_max ← max_r; best_count ← count; continuer
        pour chaque sommet v ∉ brûlés:
            pour r allant de min(max_rayon_possible, best_max) à 1:
                si r ∈ used: continuer
                nouveaux ← brûlés ∪ bruler(graphe, v, r)
                new_max ← max(max_r, r); new_count ← count + 1; new_used ← used ∪ {r}
                key ← (nouveaux, new_used)
                si key déjà enregistré avec (new_max, new_count) meilleur: continuer
                memo[key] ← (new_max, new_count)
                heur_new ← new_max + (n - taille(nouveaux)) / n
                ajouter (heur_new, new_max, new_count, nouveaux, seq + [(v, r)], new_used) dans heap

    retourner best_seq s'il existe, sinon []
```