
# Benchmark IVF-PQ pour la Recherche de Voisins Approximatifs (ANN)

##  Table des matières
1. [Introduction](#introduction)
2. [Concepts fondamentaux](#concepts-fondamentaux)
3. [Architecture du notebook](#architecture-du-notebook)
4. [Hyperparamètres IVF-PQ](#hyperparamètres-ivf-pq)
5. [Métriques de performance](#métriques-de-performance)
6. [Installation et utilisation](#installation-et-utilisation)
7. [Interprétation des résultats](#interprétation-des-résultats)
8. [Recommandations pratiques](#recommandations-pratiques)
9. [Analyse comparative](#analyse-comparative)

---

##  Introduction

Ce projet implémente un **benchmark complet de l'algorithme IVF-PQ** (Inverted File with Product Quantization) pour la recherche approximative des plus proches voisins (ANN - Approximate Nearest Neighbors).

### Objectif
Analyser les **performances**, les **compromis qualité/vitesse/mémoire**, et l'**impact des hyperparamètres `nlist`, `nprobe` et `m`** de IVF-PQ sur le dataset SIFT1M (1 million de vecteurs de dimension 128).

### Pourquoi IVF-PQ ?
-  **Très faible empreinte mémoire** : la quantification PQ compresse les vecteurs d'un facteur ×16 (avec m=8 sur dim=128)
-  **Rapide** : la partition IVF évite de scanner tous les vecteurs
-  **Scalable** : conçu pour des bases > 10 millions de vecteurs
-  **Ajustable dynamiquement** : `nprobe` modifiable sans reconstruire l'index

---

##  Concepts fondamentaux

### 1. Le double problème de la recherche à grande échelle

Pour 1 million de vecteurs en dimension 128 :
- **Problème vitesse** : comparer une requête à 1M vecteurs = 128M opérations flottantes
- **Problème mémoire** : stocker 1M × 128 × 4 bytes = **512 MB** rien que pour les données brutes

IVF-PQ résout les deux à la fois en combinant deux techniques complémentaires.

### 2. IVF : Inverted File Index (Fichier Inversé)

#### Principe

IVF partitionne l'espace vectoriel en `nlist` régions (cellules) via un algorithme **K-means**. Chaque vecteur de la base est assigné à la cellule dont le centroïde est le plus proche.

```
            K-means clustering
Base de données → [cell_1: v1, v3, v8, ...]
                  [cell_2: v2, v5, ...]
                  [cell_3: v4, v6, v9, ...]
                  ...
                  [cell_1024: ...]
```

**Lors d'une requête** : au lieu de comparer avec **tous** les 1M vecteurs, on compare uniquement avec les vecteurs des `nprobe` cellules les plus proches du vecteur requête.

```
Requête q
    ↓
Trouver les nprobe cellules les plus proches de q
    ↓
Chercher uniquement dans ces nprobe × (N/nlist) vecteurs
    ↓
Résultats approximatifs
```

**Analogie : Chercher un appartement dans Paris**

Sans IVF : visiter tous les appartements de Paris (750,000+).
Avec IVF : décider d'abord dans quel(s) arrondissement(s) chercher (= `nprobe` cellules), puis visiter uniquement les appartements de ces arrondissements.

- Plus `nprobe` est grand → plus d'arrondissements visités → meilleure précision → plus lent
- Moins `nprobe` est grand → recherche rapide mais peut rater des bons résultats

#### Règle de dimensionnement de nlist

```
nlist recommandé = √N (racine carrée du nombre de vecteurs)
```
Pour N = 1,000,000 vecteurs : `nlist ≈ 1024` est une valeur standard.

### 3. PQ : Product Quantization (Quantification par Produit)

#### Le problème de mémoire

Même avec IVF, les vecteurs résiduels (différence entre chaque vecteur et son centroïde) doivent être stockés pour le calcul de distance. En float32 : 1M × 128 × 4 = 512 MB.

#### Solution PQ : compression par sous-vecteurs

PQ découpe chaque vecteur de dimension D en `m` **sous-vecteurs** de dimension D/m, puis quantifie chaque sous-vecteur séparément.

```
Vecteur original (128 dim, 512 bytes)
    ↓ Découpage en m=8 sous-vecteurs
[v1..v16] [v17..v32] [v33..v48] ... [v113..v128]
   SQ1        SQ2       SQ3             SQ8

    ↓ Quantification (256 codes possibles par sous-vecteur = 8 bits)
[  42   ] [  17   ] [  255  ] ... [   3   ]

    ↓ Code final PQ : 8 octets au lieu de 512 !
Facteur de compression : 512 / 8 = 64x
```

**Précisément** :
- Chaque sous-vecteur de dimension `D/m` est approximé par l'un de `2^nbits = 256` centroïdes appris
- Le vecteur complet est représenté par `m` indices d'octets
- Taille mémoire par vecteur : `m × nbits/8` bytes = `m` bytes (avec nbits=8)

#### Tables de distances asymétriques (ADC)

Lors d'une recherche, PQ utilise une astuce pour éviter de décompresser les vecteurs :

```python
# Pour chaque requête q :
# 1. Calculer les distances entre q[j] et tous les centroïdes du sous-espace j
# 2. Stocker dans une table de lookup : table[j][code] = distance partielle
# 3. Pour chaque vecteur compressé : somme des distances partielles ≈ distance totale
```

Cela permet de calculer des distances approchées **directement sur les codes compressés**, sans jamais reconstruire les vecteurs originaux.

### 4. IVF-PQ : Combinaison des deux techniques

```
Phase de construction :
┌─────────────────────────────────────────────────┐
│ 1. K-means sur les vecteurs → nlist centroïdes  │
│ 2. Assignation de chaque vecteur à une cellule  │
│ 3. Calcul des résiduels (v - centroïde)         │
│ 4. Quantification PQ des résiduels              │
└─────────────────────────────────────────────────┘

Phase de recherche :
┌─────────────────────────────────────────────────┐
│ 1. Trouver les nprobe cellules les plus proches │
│ 2. Calculer la table de distances PQ            │
│ 3. Scanner les codes PQ des nprobe cellules     │
│ 4. Retourner les K candidats avec dist. min.    │
└─────────────────────────────────────────────────┘
```

**Double approximation** :
1. **IVF** : On ne visite que `nprobe/nlist` de la base
2. **PQ** : Les distances sont calculées sur les résiduels compressés

C'est pourquoi IVF-PQ atteint un recall légèrement inférieur à HNSW : il y a deux sources d'erreur au lieu d'une.

### 5. Coût initial d'entraînement

Contrairement à HNSW ou LSH, IVF-PQ **nécessite une phase d'entraînement** avant de pouvoir indexer :

```python
# Entraînement sur un sous-ensemble représentatif
index.train(train_data)  # Apprend les centroïdes IVF et PQ

# Indexation des vecteurs
index.add(base_vectors)
```

Le train K-means est le goulot d'étranglement de la construction. Une fois entraîné, l'ajout de nouveaux vecteurs est rapide.

---

##  Architecture du notebook

### Structure en 10 sections

#### **Section 1 : Installation et Imports**
```python
import faiss
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from typing import Tuple, List, Dict
```

Pas de seaborn ici : les visualisations utilisent matplotlib pur avec un style personnalisé.

#### **Section 2 : Chargement du Dataset SIFT1M**

##### `read_fvecs(filename)` et `read_ivecs(filename)`
Deux fonctions de lecture du format binaire SIFT :

```python
def read_fvecs(filename):
    with open(filename, 'rb') as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0]  # Lire dimension
        f.seek(0)

        file_size = os.path.getsize(filename)
        vec_size = 4 + d * 4  # 4 bytes dim + d × 4 bytes float
        n_vectors = file_size // vec_size

        data = np.fromfile(f, dtype=np.int32, count=n_vectors * (d + 1))
        data = data.reshape(-1, d + 1)

        return data[:, 1:].view(dtype=np.float32)
```

**Différence avec le notebook LSH** : `read_fvecs` calcule `n_vectors` via la taille du fichier (`os.path.getsize`) plutôt que de lire en une seule passe. Plus robuste sur les grands fichiers.

##### `load_sift1m(data_dir)`
Charge les 3 fichiers du dataset et affiche les shapes de validation :
```
✓ Base vectors: (1000000, 128)
✓ Query vectors: (10000, 128)
✓ Ground truth: (10000, 100)
```

##### Vérification du dataset
Section dédiée qui liste les fichiers attendus, leurs tailles en Mo, et affiche un message d'aide si des fichiers manquent :
```python
expected_files = [
    'sift_base.fvecs',        # ~512 MB
    'sift_query.fvecs',       # ~5 MB
    'sift_groundtruth.ivecs', # ~4 MB
]
```

#### **Section 3 : Module de Métriques**

##### `calculate_recall_at_k(predictions, ground_truth, k=10)`
```python
def calculate_recall_at_k(predictions, ground_truth, k=10):
    n_queries = predictions.shape[0]
    recalls = []
    for i in range(n_queries):
        pred_set = set(predictions[i, :k])
        true_set = set(ground_truth[i, :k])
        intersection = len(pred_set.intersection(true_set))
        recall = intersection / k
        recalls.append(recall)
    return np.mean(recalls)
```

##### `measure_query_time(index, query_vectors, k=10, n_runs=3)`
Amélioration par rapport aux autres notebooks : mesure sur **3 runs** avec **warmup** préalable.

```python
# Warmup : évite les effets de cache à froid
_, _ = index.search(query_vectors[:100], k)

# Mesure sur n_runs exécutions
times = []
for _ in range(n_runs):
    start = time.time()
    D, I = index.search(query_vectors, k)
    times.append(time.time() - start)

avg_time = np.mean(times)
```

**Pourquoi 3 runs ?** La variabilité de la latence sur un OS moderne peut être de ±20%. Faire la moyenne de 3 mesures donne une estimation plus stable.

**Pourquoi le warmup ?** Le premier appel `search` initialise les caches CPU et les structures internes FAISS. Le mesurer fausserait la latence réelle en production.

##### `estimate_index_size(index)`
Estimation analytique de la mémoire :
```python
# Taille des centroïdes IVF
centroids_size = nlist * d * 4  # nlist × 128 floats

# Taille des codes PQ (m octets par vecteur)
codes_size = ntotal * m

# Total estimé
size_bytes = centroids_size + codes_size
size_mb = size_bytes / (1024 * 1024)
```

**Note** : C'est une estimation basse. La mémoire réelle inclut également les structures d'inverted lists, les métadonnées, etc.

#### **Section 4 : Création et Entraînement de l'Index IVF-PQ**

##### Configuration initiale
```python
d = 128       # Dimension des vecteurs SIFT
nlist = 1024  # Nombre de cellules Voronoi
m = 8         # Sous-quantificateurs (128 / 8 = 16 dim par sous-vecteur)
nbits = 8     # 256 codes possibles par sous-quantificateur
```

**Contrainte FAISS** : `m` doit être un **diviseur de `d`**. Avec d=128, les valeurs valides sont : 1, 2, 4, 8, 16, 32, 64, 128.

**Impact de `m` sur la compression** :
| m | Taille par vecteur | Facteur compression |
|---|-------------------|---------------------|
| 4 | 4 bytes (vs 512) | ×128 |
| 8 | 8 bytes (vs 512) | ×64 |
| 16 | 16 bytes (vs 512) | ×32 |
| 32 | 32 bytes (vs 512) | ×16 |

##### Construction de l'index en deux étapes

**Étape 1 : Coarse quantizer**
```python
quantizer = faiss.IndexFlatL2(d)
```
Le `quantizer` est un index exact qui sert à trouver le centroïde IVF le plus proche d'un vecteur. C'est un `IndexFlatL2` car on veut une assignation exacte aux cellules.

**Étape 2 : Index IVF-PQ**
```python
index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
```

##### Phase d'entraînement
```python
train_size = 100000  # 100k suffisent pour apprendre les centroïdes
train_data = base_vectors[:train_size].copy()

index_ivfpq.train(train_data)
```

**Pourquoi 100k et pas 1M ?** L'entraînement K-means converge généralement avec ~50× `nlist` exemples. Avec `nlist=1024`, 100k exemples (≈ 97×) sont amplement suffisants. Utiliser 1M triplerait le temps d'entraînement sans améliorer significativement les centroïdes.

##### Phase d'indexation
```python
index_ivfpq.add(base_vectors)  # Ajoute TOUS les 1M vecteurs
```

Cette étape quantifie et compresse chaque vecteur :
1. Calcul du centroïde IVF le plus proche
2. Calcul du résiduel = vecteur - centroïde
3. Encodage PQ du résiduel

#### **Section 5 : Benchmark du paramètre nprobe**

```python
nprobe_values = [1, 2, 5, 10, 20, 50, 100, 200]
k = 10
```

Pour chaque valeur de `nprobe` :
```python
index_ivfpq.nprobe = nprobe  # Modification à chaud, sans reconstruire !
qps, latency_ms, (D, I) = measure_query_time(index_ivfpq, query_vectors, k)
recall = calculate_recall_at_k(I, ground_truth, k)
```

**Avantage clé** : `nprobe` est modifiable **sans reconstruire l'index**. C'est le levier d'ajustement temps-réel du trade-off précision/vitesse en production.

Sortie du benchmark :
```
nprobe   Recall@10   Latence(ms)        QPS
==================================================
     1      0.XXXX         X.XX       XXXX.X
     2      0.XXXX         X.XX       XXXX.X
     5      0.XXXX         X.XX       XXXX.X
    10      0.XXXX         X.XX       XXXX.X
    ...
```

#### **Section 6 : Visualisations (Impact nprobe)**

##### Graphique 1 : Impact de nprobe sur Recall et Latence (double subplot)
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Axe X en log pour mieux voir les petites valeurs de nprobe
ax1.set_xscale('log')
ax2.set_xscale('log')
```

L'échelle logarithmique est justifiée : nprobe varie de 1 à 200 (facteur 200). En échelle linéaire, les points nprobe=1,2,5 seraient illisibles.

##### Graphique 2 : Scatter plot Trade-off Recall vs Latence
```python
scatter = plt.scatter(latency_list, recall_list, c=nprobe_list,
                      cmap='viridis', s=250, alpha=0.8, edgecolors='black')
plt.colorbar(scatter, label='nprobe')
```

Chaque point est annoté avec sa valeur de `nprobe`. La colormap `viridis` permet de voir la progression croissante de `nprobe`. Ce graphique est l'équivalent de la **courbe de Pareto** des autres notebooks.

**Fichiers générés** :
- `ivfpq_nprobe_impact.png`
- `ivfpq_recall_latency.png`

#### **Section 7 : Benchmark du paramètre nlist**

```python
nlist_values = [256, 512, 1024, 2048]
nprobe_fixed = 10  # Valeur fixe pour isoler l'effet de nlist
```

Pour chaque `nlist`, un **nouvel index est créé, entraîné et peuplé** depuis zéro :
```python
for nlist_val in nlist_values:
    quantizer_temp = faiss.IndexFlatL2(d)
    index_temp = faiss.IndexIVFPQ(quantizer_temp, d, nlist_val, m, nbits)

    start = time.time()
    index_temp.train(train_data)
    index_temp.add(base_vectors)
    build_time = time.time() - start

    index_temp.nprobe = nprobe_fixed
    # Mesure recall, latence, mémoire...
```

**Ce qu'on mesure** :
- `build_time` : Temps total train + add (nlist plus grand = K-means plus long)
- `recall` : À `nprobe` fixe, plus de cellules = meilleur recall (partitionnement plus fin)
- `latency_ms` : À `nprobe` fixe, plus de cellules = chaque cellule est plus petite = recherche plus rapide
- `memory_mb` : Taille estimée de l'index

Tableau de sortie :
```
   nlist     Build(s)    Recall@10   Latence(ms)   Memory(Mo)
=================================================================
     256         X.XX       0.XXXX         X.XX         XX.XX
     512         X.XX       0.XXXX         X.XX         XX.XX
    1024         X.XX       0.XXXX         X.XX         XX.XX
    2048         X.XX       0.XXXX         X.XX         XX.XX
```

**Fichier généré** : `ivfpq_nlist_impact.png`

#### **Section 8 : Résumé des Résultats**

Synthèse automatique des meilleures configurations :
```python
# Configuration pour meilleur recall (nprobe maximal)
best_recall = max(results_nprobe, key=lambda x: x['recall'])

# Configuration la plus rapide avec recall acceptable (> 50%)
fast_results = [r for r in results_nprobe if r['recall'] > 0.5]
fastest = min(fast_results, key=lambda x: x['latency_ms'])
```

#### **Section 9 : Sauvegarde CSV**

Deux fichiers générés :
```python
df_nprobe.to_csv('ivfpq_nprobe_results.csv', index=False)
df_nlist.to_csv('ivfpq_nlist_results.csv', index=False)
```

---

##  Hyperparamètres IVF-PQ

### Vue d'ensemble des 4 paramètres

| Paramètre | Rôle | Phase impactée | Modifiable sans reconstruire ? |
|-----------|------|----------------|-------------------------------|
| `nlist` | Nombre de cellules K-means | Construction + Recherche |  Non |
| `m` | Nombre de sous-quantificateurs PQ | Construction |  Non |
| `nbits` | Bits par code PQ | Construction |  Non |
| `nprobe` | Cellules visitées à la recherche | Recherche uniquement |  Oui |

### `nlist` : Granularité du partitionnement

```python
# Création
index = faiss.IndexIVFPQ(quantizer, d, nlist=1024, m, nbits)

# Règle empirique
nlist_optimal = int(np.sqrt(N))  # Pour N vecteurs
# N=1M → nlist=1024 ✓
```

**Impact** :
- `nlist` ↑ → Cellules plus petites → `nprobe` constant visite moins de vecteurs → Recherche plus rapide
- `nlist` ↑ → Plus de centroïdes à apprendre → Build time ↑
- `nlist` ↑ → Mémoire centroïdes ↑ (nlist × 128 × 4 bytes)

### `m` : Compression PQ

```python
# m doit diviser d=128
m_options = [4, 8, 16, 32]  # Valeurs communes
```

**Impact** :
- `m` ↑ → Moins de compression → Meilleur recall → Plus de mémoire
- `m` ↓ → Compression agressive → Recall ↓ → Mémoire minimale

**Ratio mémoire** pour N=1M vecteurs :
```
m=4  → 4 MB de codes  (compression ×128)
m=8  → 8 MB de codes  (compression ×64)
m=16 → 16 MB de codes (compression ×32)
m=32 → 32 MB de codes (compression ×16)
```
(vs 512 MB sans compression)

### `nbits` : Précision des codes PQ

```python
nbits = 8  # Standard (256 centroïdes par sous-vecteur)
# Alternatives: nbits=4 (16 centroïdes, ultra-compact), nbits=12 (4096, meilleur recall)
```

`nbits=8` est le standard car 8 bits = 1 octet, ce qui est optimal pour l'alignement mémoire. Les valeurs inférieures (4) réduisent drastiquement la précision.

### `nprobe` : Le levier de production

```python
# Ajustement dynamique sans reconstruction
index.nprobe = 10   # Rapide, recall modéré
results = index.search(query, k=10)

index.nprobe = 50   # Plus précis, plus lent
results = index.search(query, k=10)
```

**Relation nprobe/recall/latence** (approximative pour nlist=1024, N=1M) :

| nprobe | % base visitée | Recall@10 approx. | Latence approx. |
|--------|---------------|-------------------|-----------------|
| 1      | 0.1%          | ~25-35%           | très bas        |
| 5      | 0.5%          | ~50-60%           | bas             |
| 10     | 1%            | ~65-75%           | modéré          |
| 20     | 2%            | ~75-82%           | modéré          |
| 50     | 5%            | ~82-88%           | élevé           |
| 100    | 10%           | ~88-93%           | élevé           |
| 200    | 20%           | ~92-96%           | très élevé      |

---

##  Métriques de performance

### 1. Recall@K (principale métrique de qualité)
```python
recall = len(set(predicted_k) ∩ set(true_k)) / K
```

### 2. QPS (Queries Per Second)
```python
qps = n_queries / avg_search_time
```

### 3. Latence (ms/requête)
```python
latency_ms = (avg_search_time / n_queries) * 1000
```

### 4. Build Time (secondes)
```python
build_time = train_time + add_time
```

### 5. Empreinte mémoire (Mo)
Estimation analytique :
```python
memory_mb = (nlist * d * 4 + ntotal * m) / (1024 * 1024)
```

---

##  Installation et utilisation

### Prérequis
```bash
pip install faiss-cpu numpy matplotlib pandas
```

### Dataset SIFT1M
Télécharger depuis :
```
ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
```

Décompresser et placer les fichiers :
```
sift1m/
├── sift_base.fvecs        # ~512 MB (1M vecteurs × 128 dims × 4 bytes)
├── sift_query.fvecs       # ~5 MB (10k vecteurs)
└── sift_groundtruth.ivecs # ~4 MB (10k requêtes × 100 voisins)
```

### Vérification des fichiers
Le notebook inclut une cellule de diagnostic :
```python
for name in expected_files:
    path = os.path.join(data_dir, name)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f'OK  {name}  ({size_mb:.1f} Mo)')
    else:
        print(f'MISSING  {name}')
```

### Lancement
1. Vérifier que les fichiers SIFT1M sont présents dans `./sift1m/`
2. Exécuter les cellules séquentiellement (Run All)

**Durée estimée** :
- Chargement des données : ~30 secondes
- Entraînement (nlist=1024) : ~60-120 secondes
- Indexation (1M vecteurs) : ~30-60 secondes
- Benchmark nprobe (8 valeurs) : ~2-5 minutes
- Benchmark nlist (4 valeurs × train+add) : ~10-20 minutes

---

## Interprétation des résultats

### Fichier `ivfpq_nprobe_results.csv`
```csv
nprobe,recall,qps,latency_ms
1,0.XXXX,XXXX.X,X.XX
2,0.XXXX,XXXX.X,X.XX
5,0.XXXX,XXXX.X,X.XX
10,0.XXXX,XXXX.X,X.XX
20,0.XXXX,XXXX.X,X.XX
50,0.XXXX,XXXX.X,X.XX
100,0.XXXX,XXXX.X,X.XX
200,0.XXXX,XXXX.X,X.XX
```

### Fichier `ivfpq_nlist_results.csv`
```csv
nlist,build_time,recall,latency_ms,memory_mb
256,X.XX,0.XXXX,X.XX,XX.XX
512,X.XX,0.XXXX,X.XX,XX.XX
1024,X.XX,0.XXXX,X.XX,XX.XX
2048,X.XX,0.XXXX,X.XX,XX.XX
```

### Lecture de la courbe de Pareto nprobe

```
Recall@10
  0.93 |                           ● (nprobe=200)
  0.90 |                      ●
  0.82 |              ●
  0.73 |         ●
  0.60 |    ●
  0.42 | ●
       +-----------------------------------> Latence (ms)
         0.5  1.0   2.0   5.0  10.0  20.0
```

**Sweet spot typique** : `nprobe = 20-50`
- Recall@10 : ~75-88%
- Latence : 2-5 ms
- Bon compromis pour la plupart des usages production

### Impact de nlist (à nprobe=10 fixe)

```
nlist=256  → Grosses cellules, recherche lente dans chaque cellule
nlist=512  → Meilleur équilibre
nlist=1024 → Standard recommandé ✓
nlist=2048 → Build time x2, gain marginal en recall
```

---

## 🎓 Recommandations pratiques

### Configuration recommandée dans le notebook
```python
nlist = 1024  # Bon compromis build time / recall
m = 8         # Compression x64, qualité correcte
nbits = 8     # Standard
nprobe = 20-50  # Ajuster selon le besoin
```

### Cas d'usage par profil

#### 1. Très grande échelle (> 10M vecteurs), RAM limitée
```python
nlist = int(np.sqrt(N))  # Adapter à N
m = 8                    # Compression agressive
nbits = 8
nprobe = 10-20           # Priorité vitesse
```
- Recall@10 : ~65-75%
- RAM : ~N × 8 bytes ≈ 80 MB pour 10M vecteurs
- **Priorité** : Empreinte mémoire minimale

#### 2. Production standard (équilibre qualité/vitesse)
```python
nlist = 1024
m = 16          # Plus de précision (compression x32)
nbits = 8
nprobe = 50
```
- Recall@10 : ~85-90%
- Latence : ~3-5 ms
- **Priorité** : Bon recall sans exploser la mémoire

#### 3. Haute précision (recall > 90%)
```python
nlist = 2048
m = 32          # Moins de compression (compression x16)
nbits = 8
nprobe = 100
```
- Recall@10 : ~90-95%
- Latence : ~5-10 ms
- **Priorité** : Qualité, mémoire encore acceptable

#### 4. Ajustement dynamique en production (sans reconstruction)
```python
# Construction une seule fois
index = build_ivfpq(vectors, nlist=1024, m=8, nbits=8)

# Mode temps réel : rapide, recall moyen
index.nprobe = 10
results_fast = index.search(query, k=10)

# Mode précis : meilleur recall
index.nprobe = 100
results_accurate = index.search(query, k=10)
```

**Use case** : Endpoints API différenciés :
- `POST /search?mode=fast` → nprobe=10
- `POST /search?mode=accurate` → nprobe=100

---

##  Analyse comparative

### IVF-PQ vs autres algorithmes ANN

| Algorithme | Recall@10 | Latence | Mémoire (1M vecteurs) | Build Time |
|------------|-----------|---------|----------------------|------------|
| **Brute Force** | 100% | ~100ms | 512 MB | Instant |
| **LSH (512 bits)** | ~47% | ~0.02ms | ~64 MB | <1 min |
| **IVF-PQ** (nlist=1024, m=8, nprobe=50) | ~85% | ~3ms | ~12 MB | ~2 min |
| **HNSW** (M=32, ef=100) | ~99.4% | ~0.23ms | ~750 MB | ~2 min |

### Position de IVF-PQ

IVF-PQ occupe une **niche spécifique** dans le spectre ANN :

```
LSH ──────────────── IVF-PQ ──────────────── HNSW
↑                       ↑                      ↑
Mémoire minimale    Compromis équilibré    Recall maximal
Recall faible       (mémoire + recall)     Mémoire élevée
Vitesse extrême     Vitesse correcte       Vitesse extrême
```

### Quand IVF-PQ domine-t-il HNSW ?

1. **Contrainte mémoire stricte** : IVF-PQ stocke 1M vecteurs en ~10 MB vs ~750 MB pour HNSW
2. **Très grande échelle** : Sur 100M+ vecteurs, HNSW consomme des dizaines de GB — IVF-PQ reste gérable
3. **Reconstruction fréquente** : Si l'index doit être reconstruit régulièrement, IVF-PQ est plus rapide
4. **Batch processing** : Pour des recherches différées non-temps-réel, la latence IVF-PQ est acceptable

### La double approximation d'IVF-PQ

```
Erreur IVF : Le vrai voisin peut être dans une cellule non visitée
    → Contrôlée par nprobe (↑ nprobe = ↓ erreur IVF)

Erreur PQ  : La distance est calculée sur des codes compressés
    → Contrôlée par m (↑ m = ↓ erreur PQ)
```

HNSW n'a qu'une seule source d'erreur (la navigation greedy dans le graphe), d'où son recall supérieur.

---

##  Ressources complémentaires

### Papers de référence

1. **Product Quantization** : "Product Quantization for Nearest Neighbor Search" (Jégou et al., 2011)
   - https://inria.hal.science/inria-00514462

2. **IVF-PQ dans FAISS** : "Billion-scale similarity search with GPUs" (Johnson et al., 2019)
   - https://arxiv.org/abs/1702.08734

3. **Survey ANN** : "Approximate Nearest Neighbor Search in High Dimensional Data" (Wang et al., 2021)

### Documentation

- **FAISS IndexIVFPQ** : https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#cell-probe-methods
- **FAISS Guidelines** : https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
- **SIFT1M dataset** : http://corpus-texmex.irisa.fr/

### Concepts clés à approfondir

- **K-means clustering** : Algorithme de quantification vectorielle au cœur d'IVF
- **Voronoi diagrams** : Représentation géométrique du partitionnement IVF
- **Asymmetric Distance Computation (ADC)** : Technique de calcul rapide de distance PQ
- **Residual encoding** : Pourquoi quantifier le résiduel plutôt que le vecteur brut

---

##  Troubleshooting

### `FileNotFoundError` au chargement

```python
raise FileNotFoundError(
    "Dataset SIFT1M introuvable. Télécharge-le depuis "
    "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz "
    "et place les fichiers dans ./sift1m"
)
```

**Ce notebook ne dispose pas de mode mock** (contrairement au notebook LSH). Il faut impérativement SIFT1M ou ajouter manuellement un fallback :

```python
# Fallback à ajouter si SIFT1M absent :
if not os.path.exists('./sift1m/sift_base.fvecs'):
    print(" Génération données synthétiques...")
    N, d = 100000, 128
    base_vectors = np.random.randn(N, d).astype('float32')
    query_vectors = np.random.randn(1000, d).astype('float32')
    index_exact = faiss.IndexFlatL2(d)
    index_exact.add(base_vectors)
    _, ground_truth = index_exact.search(query_vectors, 100)
```

### `AssertionError: m must divide d`

**Cause** : La valeur de `m` choisie ne divise pas `d=128`.

**Valeurs valides** : m ∈ {1, 2, 4, 8, 16, 32, 64, 128}

```python
# Vérification avant création
assert d % m == 0, f"m={m} ne divise pas d={d}"
```

### `index_ivfpq.is_trained == False` à l'add

**Cause** : Appel à `index.add()` avant `index.train()`.

**Solution** : Toujours dans cet ordre :
```python
index.train(train_data)  # 1. Entraîner d'abord
index.add(base_vectors)  # 2. Puis indexer
```

### Recall très faible (~0%) pour toutes les valeurs de nprobe

**Cause probable** : Le ground truth contient 100 voisins et n'est pas tronqué à K=10.

**Solution** :
```python
ground_truth = ground_truth[:n_queries, :k]  # Tronquer !
```

### Build time très long (> 10 minutes pour nlist=1024)

**Cause** : K-means sur 100k vecteurs en dimension 128, normal pour nlist élevé.

**Solution rapide** : Réduire `train_size` :
```python
train_size = 50000  # 50k suffisent pour nlist=1024
```

### Benchmark nlist très long

**Cause** : 4 index différents sont construits entièrement (train + add × 1M vecteurs).

**Solution** : Réduire l'ensemble de base pour les tests :
```python
base_subset = base_vectors[:100000]  # 100k pour développement
```

### Mémoire insuffisante avec nlist=2048

**Symptôme** : Kernel crash ou `MemoryError`.

**Solutions** :
1. Réduire le dataset : `base_vectors = base_vectors[:500000]`
2. Utiliser `m=4` au lieu de `m=8`
3. Supprimer les index temporaires entre les itérations

---

##  Checklist de validation

Avant de considérer le benchmark terminé :

- [ ] Notebook s'exécute sans erreur (Run All)
- [ ] Vérification des fichiers SIFT1M affiche 3 × "OK"
- [ ] 3 visualisations PNG générées
- [ ] `ivfpq_nprobe_results.csv` et `ivfpq_nlist_results.csv` créés
- [ ] Recall@10 > 0.60 pour nprobe=10 (nlist=1024, m=8)
- [ ] Recall@10 > 0.85 pour nprobe=100 (nlist=1024, m=8)
- [ ] Build time affiché pour nlist=1024 (référence benchmark)
- [ ] Empreinte mémoire estimée cohérente (< 50 MB pour m=8, N=1M)

---

##  Contribution

Ce projet fait partie d'un benchmark comparatif d'algorithmes ANN pour un projet de recherche académique.

**Structure du projet parent** :
```
Project-ANN-Comparative-Method/
├── lsh/          # LSH benchmark (Semaine 1)
├── ivf/          # IVF-PQ benchmark (Semaine 4A) ← CE PROJET
├── chatodit/     # HNSW benchmark (Semaine 3)
└── analysis/     # Analyse comparative finale (Semaine 5)
```



