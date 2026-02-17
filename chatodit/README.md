# Benchmark HNSW pour la Recherche de Voisins Approximatifs (ANN)

## 📋 Table des matières
1. [Introduction](#introduction)
2. [Concepts fondamentaux](#concepts-fondamentaux)
3. [Architecture du notebook](#architecture-du-notebook)
4. [Hyperparamètres HNSW](#hyperparamètres-hnsw)
5. [Métriques de performance](#métriques-de-performance)
6. [Installation et utilisation](#installation-et-utilisation)
7. [Interprétation des résultats](#interprétation-des-résultats)

---

## 🎯 Introduction

Ce projet implémente un **benchmark complet de l'algorithme HNSW** (Hierarchical Navigable Small World) pour la recherche approximative des plus proches voisins (ANN - Approximate Nearest Neighbors).

### Objectif
Analyser les **performances**, les **compromis qualité/vitesse**, et l'**impact des hyperparamètres** de HNSW sur le dataset SIFT1M (1 million de vecteurs de dimension 128).

### Pourquoi HNSW ?
- ⚡ **Très rapide** : recherche en millisecondes sur des millions de vecteurs
- 🎯 **Haute précision** : recall > 99% possible
- 📈 **Scalable** : fonctionne sur des milliards de vecteurs
- 🔧 **Flexible** : plusieurs hyperparamètres pour ajuster le compromis vitesse/précision

---

## 🧠 Concepts fondamentaux

### 1. Recherche de plus proches voisins (Nearest Neighbors)

**Problème** : Étant donné un vecteur requête `q` et une base de données de `N` vecteurs, trouver les `K` vecteurs les plus similaires à `q`.

**Exemple concret** :
- 🖼️ Recherche d'images similaires (Google Images)
- 🎵 Recommandation musicale (Spotify)
- 📝 Recherche sémantique de documents
- 🧬 Analyse de séquences ADN

### 2. Recherche exacte vs approximative

#### Recherche exacte (Brute Force)
```python
# Compare q avec TOUS les vecteurs de la base
distances = []
for vector in database:
    distances.append(distance(q, vector))
return top_k(distances)
```
- ✅ **Résultat parfait** : trouve toujours les vrais voisins
- ❌ **Très lent** : O(N × D) où N = taille base, D = dimension
- ❌ **Non scalable** : impossible sur des millions/milliards de vecteurs

#### Recherche approximative (ANN)
```python
# Utilise une structure de données intelligente
# pour ne visiter qu'une fraction des vecteurs
index = build_smart_index(database)
neighbors = index.search(q, k=10)  # Rapide !
return neighbors  # Peut manquer quelques vrais voisins
```
- ✅ **Très rapide** : O(log N) ou mieux
- ✅ **Scalable** : fonctionne sur des milliards de vecteurs
- ⚠️ **Approximatif** : peut manquer quelques vrais voisins

### 3. HNSW : Hierarchical Navigable Small World

#### Principe de base

HNSW construit un **graphe multi-couches** où :
- Chaque vecteur = un nœud du graphe
- Les nœuds similaires sont reliés par des arêtes
- La recherche "navigue" dans le graphe en suivant les arêtes

#### Architecture en couches

```
Couche 2 (top) :     A -------- B             (peu de nœuds, sauts longs)
                     |          |
Couche 1 :          A -- C -- B -- D           (plus de nœuds)
                     |   |    |   |
Couche 0 (base) :   A-C-E-F-B-G-H-D-I-J       (tous les nœuds)
```

**Recherche en 3 étapes** :
1. **Couche haute** : Sauts longs pour se rapprocher rapidement de la zone cible
2. **Couches intermédiaires** : Affiner la recherche
3. **Couche 0** : Recherche locale précise

#### Analogie : Voyager dans un pays

Imaginez que vous cherchez un restaurant à Paris :

1. **Couche haute** : Autoroutes (A1, A6...) → arriver rapidement dans la bonne région
2. **Couche intermédiaire** : Routes nationales → arriver dans le bon arrondissement
3. **Couche basse** : Rues locales → trouver l'adresse exacte

HNSW fonctionne pareil : il utilise des "autoroutes" (connexions longue distance) pour s'approcher rapidement, puis des "rues locales" (connexions courte distance) pour affiner.

### 4. Mesure de similarité

Le notebook utilise la **distance euclidienne (L2)** :

```python
distance(A, B) = sqrt(sum((A[i] - B[i])^2 for i in range(dimension)))
```

**Exemple** :
```python
A = [1, 2, 3]
B = [4, 5, 6]
distance = sqrt((4-1)² + (5-2)² + (6-3)²) = sqrt(27) ≈ 5.2
```

Plus la distance est **petite**, plus les vecteurs sont **similaires**.

---

## 📓 Architecture du notebook

### Structure en 8 phases

#### **Phase 1-2 : Configuration et imports**
```python
DIM = 128              # Dimension des vecteurs SIFT
K = 10                 # Nombre de voisins à rechercher
DATA_PATH = "data/"    # Chemin vers SIFT1M
```

#### **Phase 3 : Implémentation des fonctions de chargement**

##### `load_fvecs(filename)` : Charger les vecteurs
Le format `.fvecs` est un format binaire compact :
```
[dim][v1][v2]...[vdim] [dim][v1][v2]...[vdim] ...
 4B   4B  4B     4B     ...
```

Chaque vecteur est précédé de sa dimension (toujours 128 pour SIFT).

##### `load_ivecs(filename)` : Charger le ground truth
Même principe, mais contient des **indices** (entiers) au lieu de floats.

**Important** : Le fichier `sift_groundtruth.ivecs` contient 100 voisins, mais on n'en garde que K=10 :
```python
ground_truth = ground_truth[:, :K]  # Tronquer à 10 colonnes
```

#### **Phase 4 : Chargement du dataset SIFT1M**

**SIFT1M** contient :
- 🗂️ **1 million** de vecteurs de base (base_vectors)
- 🔍 **10,000** vecteurs de requête (query_vectors)
- ✅ **Ground truth** pré-calculé (vrais plus proches voisins)

**SIFT** = Scale-Invariant Feature Transform, des descripteurs visuels extraits d'images.

#### **Phase 5 : Préparation du ground truth**

Si les fichiers SIFT1M ne sont pas disponibles, on calcule le ground truth avec une recherche exacte :

```python
def compute_ground_truth(base, query, k=10):
    # Index exact (brute force)
    index = faiss.IndexFlatL2(dim)
    index.add(base)

    # Recherche exacte
    distances, neighbors = index.search(query, k)
    return neighbors  # Ces résultats sont parfaits (recall=100%)
```

**Note** : Cette étape peut prendre 2-3 minutes car elle compare chaque requête avec TOUS les vecteurs.

#### **Phase 6 : Construction de l'index HNSW**

```python
def build_hnsw_index(vectors, M=32, efConstruction=40):
    """
    Construit un index HNSW.

    Args:
        vectors: Les vecteurs à indexer
        M: Nombre de connexions par nœud (plus = meilleur graphe)
        efConstruction: Taille de la liste dynamique lors de la construction

    Returns:
        index: L'index HNSW construit
        build_time: Temps de construction (secondes)
        memory_mb: Mémoire utilisée (MB)
    """
    dim = vectors.shape[1]

    # Créer l'index
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = efConstruction

    # Mesurer le temps et la mémoire
    start_time = time.time()
    memory_before = psutil.Process().memory_info().rss / (1024**2)

    # Ajouter les vecteurs
    index.add(vectors)

    # Calculer les métriques
    build_time = time.time() - start_time
    memory_after = psutil.Process().memory_info().rss / (1024**2)
    memory_mb = memory_after - memory_before

    return index, build_time, memory_mb
```

#### **Phase 7 : Métrique de qualité (Recall@K)**

```python
def calculate_recall_at_k(predicted, ground_truth, k=10):
    """
    Calcule le recall@k : proportion de vrais voisins retrouvés.

    Exemple :
        ground_truth[0] = [5, 12, 8, 3, 99, ...]  # Vrais 10 voisins
        predicted[0]    = [5, 12, 7, 3, 100, ...] # Trouvés par HNSW

        Communs = {5, 12, 3} = 3 voisins corrects
        Recall = 3/10 = 0.30 (30%)
    """
    recalls = []
    for pred, true in zip(predicted, ground_truth):
        # Intersection entre prédiction et vérité
        correct = len(set(pred[:k]) & set(true[:k]))
        recalls.append(correct / k)

    return np.mean(recalls)
```

**Interprétation** :
- Recall@10 = 0.90 → on retrouve **90% des vrais voisins** en moyenne
- Recall@10 = 0.99 → **qualité quasi-parfaite** ✅

#### **Phase 8-9 : Expériences**

##### **Expérience 1 : Impact de M**

Test M = [16, 32, 64] avec efConstruction=40 et efSearch=100 fixes.

**Ce qu'on mesure** :
- `build_time` : Temps de construction de l'index
- `memory_mb` : Mémoire consommée
- `recall@1` et `recall@10` : Qualité de la recherche
- `latency_ms` : Temps de recherche par requête

**Hypothèse** : M plus grand → meilleur graphe → meilleur recall MAIS plus lent à construire et plus de mémoire.

##### **Expérience 2 : Impact de efSearch**

Test efSearch = [10, 20, 50, 100, 200, 400] avec M=32 et efConstruction=40 fixes.

**Ce qu'on mesure** :
- `recall@1` et `recall@10` : Qualité
- `latency_ms` : Vitesse

**Hypothèse** : efSearch plus grand → explore plus de candidats → meilleur recall MAIS plus lent.

**C'est l'expérience clé** car efSearch est le paramètre qu'on ajuste en production !

---

## ⚙️ Hyperparamètres HNSW

### 1. **M** : Nombre de connexions bidirectionnelles

```
M=16 : A -- B -- C        (graphe peu connecté)
M=32 : A ≈≈ B ≈≈ C        (graphe moyennement connecté)
M=64 : A ≡≡ B ≡≡ C        (graphe très connecté)
```

**Impact** :
- ⬆️ M augmente → ⬆️ Recall, ⬆️ Build time, ⬆️ Memory, ⬆️ Latency
- Valeur typique : **M=32** (bon compromis)
- Mémoire : ~40 bytes/vecteur pour M=32

**Quand modifier M ?**
- Applications **haute précision** (recherche médicale) : M=64
- Applications **contraintes mémoire** (mobile) : M=16
- **Standard** : M=32

### 2. **efConstruction** : Qualité de la construction

Taille de la liste dynamique lors de l'ajout de chaque vecteur.

**Impact** :
- ⬆️ efConstruction augmente → ⬆️ Qualité du graphe, ⬆️ Build time
- Valeur typique : **efConstruction=40-80**

**Analogie** : Construire une maison
- efConstruction faible = construction rapide mais fondations moyennes
- efConstruction élevé = construction lente mais fondations solides

**Note** : On ajuste rarement ce paramètre en production (fixé à la construction).

### 3. **efSearch** : Compromis vitesse/précision ⭐ LE PLUS IMPORTANT

Taille de la liste dynamique lors de la recherche.

```python
# efSearch = 10 : explore peu de candidats
index.hnsw.efSearch = 10
neighbors = index.search(query, k=10)  # Rapide mais imprécis

# efSearch = 200 : explore beaucoup de candidats
index.hnsw.efSearch = 200
neighbors = index.search(query, k=10)  # Lent mais précis
```

**Impact** :
- ⬆️ efSearch augmente → ⬆️ Recall, ⬆️ Latency
- Valeur typique : **efSearch=100-200**

**Quand modifier efSearch ?**

| Use case | efSearch | Recall@10 | Latency |
|----------|----------|-----------|---------|
| **Pré-filtrage rapide** | 10-20 | ~90% | < 0.1 ms |
| **Production standard** | 100-200 | 99%+ | 0.2-0.5 ms |
| **Qualité maximale** | 400+ | 99.9% | 0.5-1 ms |

**C'est LE paramètre à ajuster** selon vos besoins !

---

## 📊 Métriques de performance

### 1. Recall@K

**Définition** : Proportion des vrais K voisins retrouvés.

```
Recall@10 = (nombre de vrais voisins trouvés parmi les 10 premiers) / 10
```

**Exemple concret** :

Requête : "Trouver les 10 images les plus similaires à cette photo de chat"

```
Vrais 10 voisins (ground truth) : [img_5, img_12, img_8, img_3, img_99, img_45, img_67, img_23, img_89, img_34]

HNSW trouve               : [img_5, img_12, img_7, img_3, img_100, img_45, img_67, img_23, img_89, img_34]
                                    ✅    ✅    ❌    ✅     ❌      ✅     ✅     ✅     ✅     ✅

Corrects : 8 sur 10
Recall@10 = 8/10 = 0.80 (80%)
```

**Interprétation** :
- **Recall@10 < 0.90** : ❌ Qualité insuffisante pour la production
- **Recall@10 = 0.90-0.95** : ⚠️ Acceptable pour certaines applications
- **Recall@10 > 0.95** : ✅ Excellente qualité
- **Recall@10 > 0.99** : 🌟 Quasi-parfait

### 2. Latence (Latency)

**Définition** : Temps moyen pour traiter une requête (en millisecondes).

```python
latency_ms = (temps_total_10000_requêtes / 10000) × 1000
```

**Benchmark** :
- **< 0.5 ms** : ⚡ Très rapide (temps réel)
- **0.5-2 ms** : ✅ Rapide (production)
- **2-10 ms** : ⚠️ Acceptable (batch processing)
- **> 10 ms** : ❌ Trop lent (revoir paramètres)

### 3. Build Time

Temps de construction de l'index.

**Note** : Opération faite **une seule fois** (ou rarement), donc moins critique que la latence de recherche.

**Acceptable** : < 5 minutes pour 1M vecteurs

### 4. Memory Usage

Mémoire RAM consommée par l'index.

**Estimation** : ~40 bytes/vecteur pour M=32
- 1M vecteurs : ~40 MB + vecteurs bruts (~512 MB) = **~550 MB total**

---

## 🚀 Installation et utilisation

### Installation des dépendances

```bash
pip install faiss-cpu numpy matplotlib seaborn pandas psutil jupyter
```

### Télécharger le dataset SIFT1M

**Option 1 : Script automatique**
```bash
python download_sift1m.py
```

**Option 2 : Manuel**
1. Télécharger depuis http://corpus-texmex.irisa.fr/
2. Placer dans `data/` :
   - `sift_base.fvecs` (1M vecteurs)
   - `sift_query.fvecs` (10k requêtes)
   - `sift_groundtruth.ivecs` (vrais voisins)

### Lancer le notebook

```bash
# Ouvrir Jupyter
jupyter notebook

# Puis naviguer vers notebooks/hnsw_benchmark.ipynb
```

**⏱️ Temps d'exécution total** : ~8-12 minutes sur CPU standard

**Cellules les plus longues** :
- Construction des index HNSW (M=16,32,64) : ~5 minutes
- Calcul du ground truth (si nécessaire) : ~2-3 minutes

### Mode développement rapide

Pour tester rapidement, utilisez un sous-ensemble :

```python
# Dans la cellule de configuration
USE_MOCK_DATA = True  # Génère 10k vecteurs synthétiques

# Ou limiter SIFT1M
base_vectors = base_vectors[:100000]  # 100k au lieu de 1M
```

**Temps réduit** : ~1-2 minutes

---

## 📈 Interprétation des résultats

### Résultats attendus sur SIFT1M

#### Expérience 1 : Impact de M

| M  | Build Time | Memory | Recall@10 | Latency |
|----|-----------|--------|-----------|---------|
| 16 | ~60s      | ~300MB | 0.96-0.97 | 0.12ms  |
| 32 | ~120s     | ~750MB | 0.99+     | 0.19ms  |
| 64 | ~140s     | ~1GB   | 0.99+     | 0.27ms  |

**Observations** :
1. ⬆️ M augmente → ⬆️ Build time (quasi-linéaire)
2. ⬆️ M augmente → ⬆️ Memory (linéaire)
3. M=32 vs M=64 : **gain marginal en recall** (+0.01%) mais **coût significatif** en temps/mémoire

**Conclusion** : **M=32 est optimal** pour la plupart des cas d'usage.

#### Expérience 2 : Impact de efSearch (⭐ Expérience clé)

| efSearch | Recall@10 | Latency | Note |
|----------|-----------|---------|------|
| 10       | ~0.82     | 0.07ms  | ⚡ Très rapide mais imprécis |
| 20       | ~0.91     | 0.08ms  | ⚠️ Limite basse acceptable |
| 50       | ~0.97     | 0.13ms  | ✅ Sweet spot pour balance |
| 100      | ~0.99     | 0.23ms  | ✅ Production standard |
| 200      | ~0.997    | 0.47ms  | 🎯 Haute précision |
| 400      | ~0.999    | 0.56ms  | 🌟 Quasi-parfait (rendements décroissants) |

**Courbe de Pareto** : recall@10 vs latence

```
Recall
1.00 |                   ●────●  (efSearch=200-400)
     |              ●           Rendements décroissants
0.99 |         ●               (efSearch=100)
     |      ●                  Sweet spot ! ⭐
0.97 |   ●
0.91 | ●
0.82 |●
     +----------------------------------------> Latence (ms)
     0.07  0.13      0.23    0.47    0.56
```

**Sweet spot** : **efSearch = 50-100**
- Recall@10 > 0.97 (excellent)
- Latence < 0.25 ms (très rapide)

**Trade-off clé** : Après efSearch=100, doubler la valeur donne :
- ❌ Latence ×2
- ✅ Recall +0.3% seulement

### Graphiques générés

#### 1. Impact de M (`impact_M.png`)

Deux subplots :
- **Gauche** : Build time vs M → montre la croissance du temps de construction
- **Droite** : Memory vs M → montre la consommation mémoire

#### 2. Courbe de Pareto (`pareto_curve_hnsw.png`)

Graphique principal montrant le **trade-off qualité/vitesse** :
- **Axe X** : Latence (ms/query)
- **Axe Y** : Recall@10
- **Points annotés** : Chaque valeur de efSearch
- **Lignes de référence** :
  - Recall = 0.90 (seuil "acceptable")
  - Recall = 0.95 (seuil "excellent")

**Comment lire la courbe** :
1. Points **en haut à gauche** = meilleurs (haute qualité + rapide)
2. La courbe montre les **rendements décroissants** : après un certain point, plus de latence donne peu de gain en recall
3. Le **sweet spot** est marqué avec une étoile ⭐

### Fichiers CSV

#### `results_M.csv`
```csv
M,build_time,memory_mb,recall_at_1,recall_at_10,latency_ms
16,61.30,300.5,0.95,0.967,0.121
32,121.96,749.9,0.99,0.994,0.192
64,137.80,992.2,0.99,0.993,0.273
```

#### `results_efSearch.csv`
```csv
efSearch,recall_at_1,recall_at_10,latency_ms
10,0.89,0.821,0.065
20,0.95,0.906,0.080
50,0.96,0.969,0.134
100,0.98,0.994,0.229
200,0.99,0.997,0.471
400,1.00,0.999,0.565
```

---

## 🎓 Recommandations pratiques

### Configuration par cas d'usage

#### 1. Application temps réel (ex: recherche d'images en direct)
```python
M = 32
efConstruction = 40
efSearch = 50-100
```
- Recall@10 : 97-99%
- Latence : < 0.25 ms
- **Priorité** : Vitesse

#### 2. Application batch (ex: génération de recommandations nocturne)
```python
M = 64
efConstruction = 80
efSearch = 200-400
```
- Recall@10 : 99.7%+
- Latence : 0.5-1 ms (acceptable en batch)
- **Priorité** : Qualité maximale

#### 3. Application mobile/contrainte mémoire
```python
M = 16
efConstruction = 40
efSearch = 50
```
- Recall@10 : 96-97%
- Latence : < 0.15 ms
- Mémoire : ~300 MB pour 1M vecteurs
- **Priorité** : Empreinte mémoire faible

#### 4. Production standard (équilibré)
```python
M = 32              # Bon compromis
efConstruction = 40 # Standard
efSearch = 100      # Ajustable dynamiquement
```
- Recall@10 : 99%+
- Latence : ~0.2 ms
- **Le plus courant** ✅

### Ajustement dynamique de efSearch

HNSW permet d'ajuster efSearch **sans reconstruire l'index** :

```python
# Construction une seule fois
index = build_hnsw_index(vectors, M=32, efConstruction=40)

# Ajuster selon le besoin
index.hnsw.efSearch = 50   # Recherche rapide
results_fast = index.search(query, k=10)

index.hnsw.efSearch = 200  # Recherche précise
results_accurate = index.search(query, k=10)
```

**Use case** : API avec deux endpoints
- `/search/fast` → efSearch=50 (temps réel)
- `/search/accurate` → efSearch=200 (qualité maximale)

---

## 🔍 Analyse comparative

### HNSW vs autres algorithmes ANN

| Algorithme | Recall@10 | Latence | Memory | Build Time |
|------------|-----------|---------|--------|------------|
| **Brute Force** | 100% | 100ms | 512MB | Instant |
| **HNSW** (M=32, ef=100) | 99.4% | 0.23ms | 750MB | 2min |
| LSH | 85-90% | 0.5ms | 300MB | 30s |
| IVF (nlist=1000) | 95% | 1ms | 600MB | 1min |

**HNSW domine** sur le compromis recall/latence !

### Pourquoi HNSW est-il si efficace ?

1. **Structure hiérarchique** : Permet des sauts longs (comme les autoroutes)
2. **Graphe navigable** : Suit toujours la direction du voisin le plus proche
3. **Recherche locale efficace** : Une fois dans la bonne zone, explore finement
4. **Pas de quantization** : Contrairement à IVF, garde toute la précision

---

## 📚 Ressources complémentaires

### Papers de référence

1. **HNSW original** : "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018)
   - https://arxiv.org/abs/1603.09320

2. **FAISS** : "Billion-scale similarity search with GPUs" (Johnson et al., 2019)
   - https://arxiv.org/abs/1702.08734

### Documentation

- **FAISS** : https://github.com/facebookresearch/faiss/wiki
- **SIFT1M dataset** : http://corpus-texmex.irisa.fr/

### Concepts clés à approfondir

- **Small World Networks** : Graphes avec propriétés de navigation efficace
- **Skip Lists** : Structure de données hiérarchique similaire (inspiration de HNSW)
- **Greedy Search** : Algorithme de recherche gloutonne dans les graphes

---

## 🐛 Troubleshooting

### Erreur : `AssertionError: shape attendue (10000, 10), obtenue (10000, 100)`

**Cause** : Le ground truth contient 100 voisins, pas 10.

**Solution** : Ajouter après le chargement :
```python
ground_truth = ground_truth[:, :K]
```

### Cellules très longues

**Normal** : Construction HNSW sur 1M vecteurs prend ~5 minutes.

**Solution rapide** : Utiliser un sous-ensemble
```python
base_vectors = base_vectors[:100000]  # 100k seulement
```

### Mémoire insuffisante

**Symptôme** : `MemoryError` ou kernel crash.

**Solutions** :
1. Réduire la taille : `base_vectors = base_vectors[:500000]`
2. Utiliser M=16 au lieu de M=32
3. Fermer les autres applications

### Résultats différents de ceux attendus

**Possible** : Variations dues au CPU, version FAISS, ou dataset.

**Vérifications** :
1. Version FAISS : `print(faiss.__version__)` (recommandé >= 1.7)
2. Shape des données : `print(base_vectors.shape)` doit être (1000000, 128)
3. Seed : `np.random.seed(42)` pour reproductibilité

---

## ✅ Checklist de validation

Avant de considérer le benchmark terminé :

- [ ] Notebook s'exécute sans erreur (Run All)
- [ ] 3 visualisations générées dans `results/`
- [ ] `results_M.csv` et `results_efSearch.csv` créés
- [ ] Recall@10 > 0.90 pour M=32, efSearch=100
- [ ] Latence < 2 ms pour efSearch=100
- [ ] Courbe de Pareto montre le sweet spot
- [ ] Analyse textuelle inclut des chiffres concrets (pas de placeholder)

---

## 👥 Contribution

Ce projet fait partie d'un benchmark comparatif d'algorithmes ANN pour un projet de recherche académique.

**Structure du projet parent** :
```
Project-ANN-Comparative-Method/
├── lsh/          # LSH benchmark (Semaine 1)
├── ivf/          # IVF benchmark (Semaine 2)
├── chatodit/     # HNSW benchmark (Semaine 3) ← CE PROJET
└── analysis/     # Analyse comparative finale
```

---

## 📄 Licence

Ce projet utilise :
- **FAISS** (MIT License) - Facebook AI Research
- **Dataset SIFT1M** - INRIA Rennes (usage académique)

---

## 📧 Contact

Pour questions ou suggestions concernant ce benchmark, consultez la documentation dans `CLAUDE.md` ou les prompts dans `prompts/`.

---

**🚀 Prêt à benchmarker HNSW ? Lancez le notebook et découvrez les performances de cet algorithme révolutionnaire !**
