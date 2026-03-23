# Benchmark LSH pour la Recherche de Voisins Approximatifs (ANN)

Table des matières
1. [Introduction](#introduction)
2. [Concepts fondamentaux](#concepts-fondamentaux)
3. [Architecture du notebook](#architecture-du-notebook)
4. [Hyperparamètre principal : nbits](#hyperparamètre-principal--nbits)
5. [Métriques de performance](#métriques-de-performance)
6. [Installation et utilisation](#installation-et-utilisation)
7. [Interprétation des résultats](#interprétation-des-résultats)
8. [Recommandations pratiques](#recommandations-pratiques)
9. [Analyse comparative](#analyse-comparative)

---

 Introduction

Ce projet implémente un **benchmark complet de l'algorithme LSH** (Locality Sensitive Hashing) pour la recherche approximative des plus proches voisins (ANN - Approximate Nearest Neighbors).

### Objectif
Analyser les **performances**, les **compromis qualité/vitesse/mémoire**, et l'**impact du paramètre `nbits`** de LSH sur le dataset SIFT1M (1 million de vecteurs de dimension 128).

### Pourquoi LSH ?
- Empreinte mémoire ultra-faible** : stockage binaire compact, idéal pour les systèmes à contraintes RAM
- Speedup massif** : des centaines de fois plus rapide qu'un scan linéaire exact
- Scalable** : adapté aux bases de données très volumineuses (Big Data)
- Simple à configurer** : un seul hyperparamètre principal (`nbits`)

---

##  Concepts fondamentaux

### 1. Recherche de plus proches voisins (Nearest Neighbors)

**Problème** : Étant donné un vecteur requête `q` et une base de données de `N` vecteurs, trouver les `K` vecteurs les plus similaires à `q`.

**Exemples concrets** :
-  Déduplication d'images à grande échelle
-  Recommandation musicale
- Recherche sémantique de documents
-  Détection de séquences similaires

### 2. Recherche exacte vs approximative

#### Recherche exacte (Brute Force)
```python
# Compare q avec TOUS les vecteurs de la base
for vector in database:
    distances.append(distance(q, vector))
return top_k(distances)
```
-  **Résultat parfait** : trouve toujours les vrais voisins
-  **Très lent** : O(N × D) — impossible à l'échelle du million

#### Recherche approximative (ANN)
```python
# Utilise une structure de hachage pour ne visiter
# qu'une fraction des vecteurs
index = build_lsh_index(database, nbits=512)
neighbors = index.search(q, k=10)  # Rapide !
```
-  **Très rapide** : complexité sub-linéaire
-  **Scalable** : fonctionne sur des milliards de vecteurs
-  **Approximatif** : peut manquer certains vrais voisins

### 3. LSH : Locality Sensitive Hashing

#### Principe de base

LSH repose sur l'idée fondamentale suivante : **deux vecteurs similaires ont une forte probabilité de tomber dans le même "seau" (bucket) de hachage**, tandis que deux vecteurs différents ont une faible probabilité d'être co-localisés.

L'implémentation FAISS de LSH utilise un **hachage binaire par projection aléatoire** :

1. Générer `nbits` hyperplans aléatoires dans l'espace de dimension D
2. Pour chaque vecteur `v`, calculer son **code binaire** :
   ```
   code[i] = 1 si v · w[i] > 0
   code[i] = 0 sinon
   ```
   où `w[i]` est le vecteur normal du i-ème hyperplan.
3. La **distance de Hamming** entre deux codes binaires sert à estimer leur similarité dans l'espace original.

#### Visualisation du principe

```
Espace original (2D)          Espace haché (4 bits)

  *  * |  o  o                  * → 0101
  *    |    o                   o → 1100
  -----+-----        →
  *  * |  o  o       Les * ont des codes similaires,
                      les o aussi. Peu de collisions inter-groupes.
```

#### Analogie : Trier des livres par couleur de couverture

Imaginez classer 1 million de livres uniquement par la couleur dominante de leur couverture :

- **Rouge** → bucket 001
- **Bleu**  → bucket 010
- **Vert**  → bucket 011
- etc.

Pour trouver les livres "similaires" à un livre rouge, on cherche uniquement dans le bucket rouge. C'est approximatif (deux livres rouges peuvent être très différents) mais **extrêmement rapide**. LSH fait pareil avec des projections vectorielles.

### 4. Distance de Hamming

La recherche dans l'index LSH compare les **codes binaires** via la distance de Hamming :

```python
# Distance de Hamming = nombre de bits différents
code_A = "01101001"
code_B = "01100011"
#              ^^   → 2 bits différents
hamming(A, B) = 2
```

Plus la distance de Hamming est **petite**, plus les vecteurs originaux sont **probablement proches**.

**Limite importante** : Avec des vecteurs de dimension 128 comme SIFT, la projection binaire perd de l'information. C'est pourquoi LSH atteint des recalls plus faibles que HNSW, sauf avec un très grand nombre de bits.

---

## 📓 Architecture du notebook

### Structure en 6 sections

#### **Section 1 : Installation et Imports**
```python
import numpy as np
import faiss
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
```

Configuration du style de visualisation (seaborn whitegrid, DPI 100, figsize 12×7).

#### **Section 2 : Fonctions d'évaluation**

##### `calculate_recall(I_approx, I_exact, k=K)`
Calcule le Recall@K moyen sur l'ensemble des requêtes :
```python
def calculate_recall(I_approx, I_exact, k=K):
    recalls = []
    for i in range(len(I_approx)):
        intersection = np.intersect1d(I_approx[i][:k], I_exact[i][:k])
        recalls.append(len(intersection) / k)
    return np.mean(recalls)
```

**Interprétation** :
- Recall@10 = 0.60 → 60% des vrais voisins retrouvés en moyenne
- Recall@10 = 0.90 → qualité acceptable pour la plupart des applications

##### `get_memory_usage()`
Retourne la RAM utilisée par le processus Python en MB :
```python
process = psutil.Process()
return process.memory_info().rss / 1024 / 1024
```

##### `evaluate_lsh(index, xb, xq, gt, nbits, k=K)`
Fonction centrale d'évaluation qui retourne un dictionnaire de métriques :

```python
{
    'nbits': nbits,            # Taille de hachage testée
    'Recall@10': recall,       # Qualité de la recherche
    'Latency (ms)': latency,   # Temps de recherche par requête
    'Build Time (s)': build_time, # Temps d'entraînement + indexation
    'Index RAM (MB)': index_memory # Mémoire consommée
}
```

**Séquence d'évaluation** :
1. Mesure mémoire **avant** (`mem_before`)
2. Entraînement de l'index (`index.train(xb)`)
3. Ajout des vecteurs (`index.add(xb)`)
4. Mesure mémoire **après** → `index_memory = mem_after - mem_before`
5. Recherche sur les requêtes → mesure latence
6. Calcul du recall par comparaison avec le ground truth

#### **Section 3 : Benchmark LSH**

##### Configuration des expériences
```python
# Constantes globales
DIM = 128           # Dimension SIFT
K = 10              # Nombre de voisins cibles
NUM_QUERIES = 100   # Nombre de requêtes de test
USE_MOCK_DATA = False  # Basculer si SIFT1M absent
np.random.seed(42)  # Reproductibilité
```

##### Valeurs de `nbits` testées
```python
nbits_list = [32, 64, 128, 256, 512, 1024]
```
Les valeurs 2048 et 4096 ont été exclues car elles sont **trop gourmandes** en mémoire et en calcul sans apporter de gain significatif de recall.

##### Boucle de benchmark avec gestion mémoire
```python
import gc

for nbits in nbits_list:
    index_lsh = faiss.IndexLSH(DIM, nbits)
    res = evaluate_lsh(index_lsh, base_vectors, query_vectors, ground_truth, nbits)
    results.append(res)

    del index_lsh   # Supprime la référence
    gc.collect()    # Force FAISS à vider la RAM immédiatement
```

**Pourquoi `gc.collect()` est crucial ici ?**
FAISS alloue de la mémoire native (C++) qui n'est pas automatiquement libérée par le garbage collector Python. Sans cet appel explicite, la mémoire s'accumule au fil des itérations, faussant les mesures de `Index RAM (MB)` et pouvant provoquer un `MemoryError`.

#### **Section 4 : Visualisation et Analyse des Performances**

##### Graphique 1 : Courbe de Pareto (Recall vs Latence)
```python
sns.lineplot(data=df_results, x='Latency (ms)', y='Recall@10',
             marker='o', markersize=10, linewidth=2.5, color='#2ca02c')
```
Chaque point est annoté avec sa valeur de `nbits`. Permet de visualiser les **rendements décroissants** : au-delà d'un certain seuil, augmenter `nbits` double la latence pour un gain de recall marginal.

##### Graphique 2 : Recall@10 selon nbits (barplot)
Montre la progression de la qualité de recherche en fonction de la résolution de hachage. Met en évidence le **plateau de recall** à partir d'un certain nombre de bits.

##### Graphique 3 : Empreinte mémoire selon nbits (barplot)
Illustre la croissance **linéaire** de la consommation RAM avec `nbits`. Cette progression reste modeste comparée à HNSW.

#### **Section 4.1 : Temps de Construction et QPS**

##### Calcul du QPS (Queries Per Second)
```python
df_results['QPS'] = 1000 / df_results['Latency (ms)']
```

**Pourquoi le QPS est la métrique numéro 1 en production ?**
La latence par requête indique ce que ressent **un seul utilisateur**. Le QPS indique combien d'utilisateurs **simultanés** le système peut servir. C'est le vrai critère de dimensionnement des serveurs.

**Exemple** :
- Latence = 0.5 ms → QPS = 2,000 requêtes/seconde
- Latence = 5 ms → QPS = 200 requêtes/seconde

Le graphique QPS est en **échelle logarithmique** car les variations entre `nbits=32` et `nbits=1024` peuvent couvrir plusieurs ordres de grandeur.

#### **Section 4.2 : Facteur d'Accélération (Speedup)**

##### Baseline : IndexFlatL2 (Brute Force)
```python
index_exact = faiss.IndexFlatL2(DIM)
index_exact.add(base_vectors)
index_exact.search(query_vectors, K)
exact_time_ms = elapsed * 1000 / len(query_vectors)
```

##### Calcul du Speedup
```python
df_results['Speedup (x)'] = exact_time_ms / df_results['Latency (ms)']
```

**Lecture du graphique** : Un speedup de `x100` signifie que LSH répond 100 fois plus vite que la recherche exacte. C'est l'argument principal de LSH : même avec un recall de seulement 60%, si la réponse arrive 500 fois plus vite, le compromis peut être acceptable dans de nombreux contextes Big Data.

#### **Section 5 & 6 : Conclusions**

Analyse narrative structurée en 3 observations clés :
1. Compromis Précision/Vitesse (rôle de `nbits`)
2. L'atout mémoire de LSH
3. La justification par le speedup

---

## Hyperparamètre principal : nbits

### Qu'est-ce que `nbits` ?

`nbits` est le **nombre de bits** du code de hachage généré pour chaque vecteur. C'est l'**unique paramètre de configuration** de `faiss.IndexLSH`.

```python
index = faiss.IndexLSH(DIM, nbits)
#                      ^^^  ^^^^^
#                      128  nombre de bits du hash
```

### Impact de `nbits` sur les métriques

| `nbits` | Recall@10 | Latence | RAM Index | Build Time |
|---------|-----------|---------|-----------|------------|
| 32      | ~0.05     | très bas | minimal   | rapide     |
| 64      | ~0.10     | bas      | faible    | rapide     |
| 128     | ~0.15-0.20| modéré   | faible    | rapide     |
| 256     | ~0.25-0.35| modéré   | faible    | modéré     |
| 512     | ~0.40-0.55| élevé    | modéré    | modéré     |
| 1024    | ~0.55-0.65| très élevé| modéré  | lent       |

>  Les valeurs de recall ci-dessus sont indicatives pour SIFT1M avec `faiss.IndexLSH`. Sans PCA préalable, LSH atteint rarement 70%+ de recall sur des descripteurs SIFT à 128 dimensions.

### Comprendre le plafond de recall de LSH

La distance de Hamming entre codes binaires est une **approximation grossière** de la distance euclidienne L2. Plus les vecteurs sont de haute dimension (ici 128), plus cette approximation perd de la précision. C'est pourquoi :

- LSH pur sur SIFT128 plafonne souvent à **40-65% de recall**
- Pour dépasser ce plafond, il faudrait ajouter une réduction dimensionnelle (PCA) avant le hachage
- HNSW ou IVF-PQ atteignent 95%+ de recall sur le même dataset

### Règle de dimensionnement pratique

```
nbits optimal ≈ DIM × facteur
```
- `facteur = 1` → `nbits = 128` : codage compact, faible recall
- `facteur = 4` → `nbits = 512` : bon compromis
- `facteur = 8` → `nbits = 1024` : recall maximal, latence élevée

---

## Métriques de performance

### 1. Recall@K
```python
# Proportion de vrais voisins retrouvés parmi K résultats
recall = len(intersection(predicted_k, true_k)) / K
```
### 2. Latence (ms/requête)
```python
search_time = time.time() - t0  # secondes
latency_ms = (search_time * 1000) / n_queries
```

### 3. QPS (Queries Per Second)
```python
QPS = 1000 / latency_ms
```

### 4. Speedup vs Brute Force
```python
speedup = exact_latency_ms / lsh_latency_ms
```

### 5. Build Time (s)
Somme du temps d'entraînement (projection aléatoire) et du temps d'indexation.

### 6. Index RAM (MB)
Différence de mémoire résidente du processus avant/après la construction de l'index.

---

##  Installation et utilisation

### Prérequis
```bash
pip install faiss-cpu numpy matplotlib seaborn psutil pandas
```

### Dataset SIFT1M
Télécharger depuis : `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`

Structure attendue :
```
data/
├── sift_base.fvecs        # 1M vecteurs de base (512 MB)
├── sift_query.fvecs       # 10k vecteurs de requête
└── sift_groundtruth.ivecs # Ground truth (100 voisins par requête)
```

### Chargement des données

```python
# Format binaire .fvecs : [dim][v1][v2]...[vdim] répété
def load_fvecs(filename):
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.int32)
        vec_size = 1 + dim
        n_vectors = len(data) // vec_size
        data = data.reshape(n_vectors, vec_size)
        return data[:, 1:].view(np.float32)
```

**Note importante** : Le fichier `sift_groundtruth.ivecs` contient 100 voisins par requête. On tronque à `K=10` :
```python
ground_truth = ground_truth[:NUM_QUERIES, :K]
```

### Mode données synthétiques (si SIFT1M absent)
```python
USE_MOCK_DATA = True

# Génération automatique
base_vectors = np.random.randn(10000, DIM).astype('float32')
query_vectors = np.random.randn(NUM_QUERIES, DIM).astype('float32')

# Ground truth exact via IndexFlatL2
index_exact = faiss.IndexFlatL2(DIM)
index_exact.add(base_vectors)
_, ground_truth = index_exact.search(query_vectors, K)
```

>  Avec des données aléatoires gaussiennes, les recalls observés seront **différents** de SIFT1M. Utiliser ce mode uniquement pour valider que le code s'exécute.

### Lancement
```
Run All (Shift+Enter dans chaque cellule séquentiellement)
```

---

## Interprétation des résultats

### Tableau des résultats (`lsh_benchmark_results.csv`)

```csv
nbits,Recall@10,Latency (ms),Build Time (s),Index RAM (MB),QPS,Speedup (x)
32,0.0450,0.0012,X.XX,Y.YY,833333,Z.Z
64,0.0890,0.0023,X.XX,Y.YY,434782,Z.Z
128,0.1650,0.0047,X.XX,Y.YY,212765,Z.Z
256,0.2900,0.0095,X.XX,Y.YY,105263,Z.Z
512,0.4700,0.0198,X.XX,Y.YY,50505,Z.Z
1024,0.6100,0.0421,X.XX,Y.YY,23752,Z.Z
```

> Les valeurs exactes dépendent de la machine et du dataset (SIFT1M vs données synthétiques).

### Lecture de la courbe de Pareto

```
Recall@10
  0.65 |                    ●  (1024 bits)
  0.50 |               ●
  0.30 |         ●
  0.17 |    ●
  0.09 | ●
  0.05 |●
       +---------------------------------> Latence (ms)
        0.001 0.002 0.005 0.010 0.020 0.040
```

**Lectures clés** :
- La courbe est **concave** : les premiers bits apportent peu de recall mais sont très rapides
- Le gain de recall par bit ajouté **décroît** à mesure que `nbits` augmente
- Il n'y a **pas de sweet spot** évident comme pour HNSW : LSH est un compromis global moins efficace

### Comparaison Speedup vs Recall

| `nbits` | Speedup | Recall@10 | Verdict |
|---------|---------|-----------|---------|
| 32      | ~500x   | ~5%       | Trop approximatif |
| 256     | ~100x   | ~30%      | Compromis extrême vitesse |
| 512     | ~50x    | ~47%      | Meilleur compromis LSH |
| 1024    | ~25x    | ~61%      | Recall maximal LSH |

---

##  Recommandations pratiques

### Quand utiliser LSH ?

####  Cas d'usage appropriés

**1. Déduplication à très grande échelle (> 100M vecteurs)**
```python
nbits = 512
# RAM ≈ 512 × N / 8 bytes = 64 MB pour 1M vecteurs
# vs HNSW : ~750 MB pour 1M vecteurs
```

**2. Systèmes à ressources matérielles extrêmement limitées**
```python
nbits = 256  # Minimal recall acceptable
# Empreinte mémoire minimale
```

**3. Filtrage grossier avant une re-sélection fine**
```python
# Etape 1: LSH rapide → 100 candidats
candidates = lsh_index.search(query, k=100)

# Etape 2: Re-ranking exact sur les 100 candidats
final_results = exact_rerank(candidates, query, k=10)
```

####  Cas d'usage déconseillés

- Applications nécessitant Recall > 80% → utiliser HNSW ou IVF-PQ
- Bases de données < 100k vecteurs → la recherche exacte est suffisamment rapide
- Systèmes avec contraintes de latence < 1ms ET recall > 70%

### Configuration recommandée par cas

| Contexte | `nbits` | Recall attendu | Speedup |
|----------|---------|----------------|---------|
| Screening initial | 128 | ~15% | ~500x |
| Déduplication | 512 | ~47% | ~50x |
| Meilleur LSH possible | 1024 | ~61% | ~25x |

---

##  Analyse comparative

### LSH vs autres algorithmes ANN

| Algorithme | Recall@10 | Latence | Mémoire | Build Time | Cas d'usage |
|------------|-----------|---------|---------|------------|-------------|
| **Brute Force** | 100% | ~100ms | 512MB | Instant | Référence |
| **LSH (512 bits)** | ~47% | ~0.02ms | ~64MB | Très rapide | Très grande échelle, RAM limitée |
| **IVF-PQ** | ~70-80% | ~1ms | ~100MB | ~1min | Grande échelle, mémoire modérée |
| **HNSW** (M=32) | ~99.4% | ~0.23ms | ~750MB | ~2min | Applications production standard |

### Pourquoi LSH est souvent moins performant que HNSW ?

1. **Perte d'information** : La projection binaire compresse trop agressivement les vecteurs 128D
2. **Pas de structure hiérarchique** : Pas d'équivalent aux "autoroutes" de HNSW
3. **Scan linéaire des codes** : FAISS compare les codes de Hamming sur **tous** les vecteurs indexés — pas de vrai "bucket" spatial, juste une distance approximative plus rapide à calculer
4. **Sensibilité à la distribution des données** : Les projections aléatoires ne s'adaptent pas à la géométrie du dataset

### Cas où LSH domine

1. **Contrainte mémoire extrême** : 10x moins de RAM que HNSW pour un recall moyen
2. **Construction instantanée** : Pas de phase d'entraînement longue comme IVF
3. **Flots de données en streaming** : L'ajout de vecteurs est très rapide
4. **Scalabilité linéaire** : La mémoire croît exactement de `nbits/8` bytes par vecteur

---

##  Ressources complémentaires

### Papers de référence

1. **LSH original** : "Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality" (Indyk & Motwani, 1998)

2. **Locality Sensitive Hashing for Cosine/L2** : "Similarity Estimation Techniques from Rounding Algorithms" (Charikar, 2002)

3. **FAISS** : "Billion-scale similarity search with GPUs" (Johnson et al., 2019)
   - https://arxiv.org/abs/1702.08734

### Documentation

- **FAISS IndexLSH** : https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- **SIFT1M dataset** : http://corpus-texmex.irisa.fr/
- **ANN Benchmarks** : http://ann-benchmarks.com/

### Concepts clés à approfondir

- **Distance de Hamming** : Mesure de dissimilarité entre chaînes binaires
- **Random Projections (Johnson-Lindenstrauss)** : Lemme fondamental justifiant LSH
- **Min-Hash LSH** : Variante pour similarité de Jaccard (ensembles)
- **Simhash** : Variante utilisée par Google pour la déduplication web

---

##  Troubleshooting

### `MemoryError` ou kernel crash pendant le benchmark

**Cause** : Les index FAISS précédents ne sont pas libérés entre les itérations.

**Solution** : Vérifier la présence de `gc.collect()` après chaque `del index_lsh` :
```python
del index_lsh
gc.collect()  # ← CRUCIAL pour FAISS
```

### Recall toujours proche de 0

**Cause probable 1** : Ground truth non tronqué à `K=10`.

**Solution** :
```python
ground_truth = ground_truth[:NUM_QUERIES, :K]  # ← tronquer !
```

**Cause probable 2** : Dataset en mode mock avec `num_base` trop petit.

**Solution** :
```python
num_base = 100000  # Augmenter à 100k minimum
```

### Résultats non reproductibles entre deux exécutions

**Solution** : Fixer la graine aléatoire **avant** toute opération NumPy :
```python
np.random.seed(42)
```

### `faiss.IndexLSH` très lent sur nbits=1024

**Normal** : La distance de Hamming sur 1024 bits est calculée sur **tous** les vecteurs indexés. Sur 1M vecteurs, c'est un scan complet de 1M × 1024/8 = 128 MB de données.

**Solution rapide** : Tester d'abord sur un sous-ensemble :
```python
base_vectors = base_vectors[:100000]  # 100k pour développement
```

### Erreur : `faiss` non trouvé

```bash
pip install faiss-cpu
# ou avec GPU :
pip install faiss-gpu
```

---

##  Checklist de validation

Avant de considérer le benchmark terminé :

- [ ] Notebook s'exécute sans erreur (Run All)
- [ ] 5 graphiques générés dans `results/` (`lsh_analysis_charts.png` + graphiques QPS et Speedup)
- [ ] `lsh_benchmark_results.csv` créé avec les 6 colonnes attendues
- [ ] Recall@10 > 0.40 pour `nbits=512`
- [ ] Speedup > 10 pour tous les `nbits` testés
- [ ] Libération mémoire (`gc.collect()`) présente dans la boucle
- [ ] Mode `USE_MOCK_DATA` documenté en cas d'absence de SIFT1M

---

##  Contribution

Ce projet fait partie d'un benchmark comparatif d'algorithmes ANN pour un projet de recherche académique.

**Structure du projet parent** :
```
Project-ANN-Comparative-Method/
├── lsh/          # LSH benchmark (Semaine 1) ← CE PROJET
├── ivf/          # IVF-PQ benchmark (Semaine 4A)
├── chatodit/     # HNSW benchmark (Semaine 3)
└── analysis/     # Analyse comparative finale (Semaine 5)
```

---

##  Licence

Ce projet utilise :
- **FAISS** (MIT License) - Facebook AI Research
- **Dataset SIFT1M** - INRIA Rennes (usage académique)

---

