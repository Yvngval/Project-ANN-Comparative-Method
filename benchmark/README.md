# Benchmark ANN — Dossier unifié

Comparaison expérimentale de trois méthodes de recherche approximative de plus proches voisins (ANN) sur le dataset **SIFT1M** (1 million de vecteurs, 128 dimensions).

---

## Installation

### 1. Installer les dépendances Python

```bash
pip install faiss-cpu scikit-learn numpy matplotlib seaborn psutil pandas jupyterlab
```

### 2. Télécharger le dataset SIFT1M

Les données ne sont **pas incluses dans le dépôt** (trop volumineuses, ~500 MB). Il faut les télécharger une fois :

```bash
cd benchmark
python download_data.py
```

Le script télécharge et extrait automatiquement les fichiers dans `benchmark/data/`. Si le téléchargement FTP échoue, tu peux aussi les récupérer manuellement :

> Télécharge `sift.tar.gz` depuis http://corpus-texmex.irisa.fr/  
> Extrais les 4 fichiers dans `benchmark/data/` :
> - `sift_base.fvecs`
> - `sift_query.fvecs`
> - `sift_groundtruth.ivecs`
> - `sift_learn.fvecs`

### 3. Lancer Jupyter

```bash
cd benchmark
jupyter lab
```

---

## Notebooks

Les 4 notebooks sont **indépendants** — exécutables dans n'importe quel ordre.

### `hnsw_benchmark.ipynb` — HNSW
Hierarchical Navigable Small World — index basé sur un graphe hiérarchique.

**Paramètres étudiés :**
- `M` — nombre de connexions par nœud (trade-off mémoire / recall)
- `efSearch` — taille de la file de recherche (trade-off vitesse / recall)

**Outputs :** `results/results_M.csv`, `results/results_efSearch.csv`, `results/impact_M.png`, `results/pareto_curve_hnsw.png`

---

### `lsh_benchmark.ipynb` — LSH
Locality Sensitive Hashing — projection aléatoire en codes binaires.

**Paramètre étudié :**
- `nbits` — nombre de bits de hachage `[32, 64, 128, 256, 512, 1024]`

**Outputs :** `results/lsh_benchmark_results.csv`, `results/lsh_analysis_charts.png`

---

### `ivfpq_benchmark.ipynb` — IVF-PQ
Inverted File + Product Quantization — clustering IVF et compression PQ.

**Paramètres étudiés :**
- `nprobe` — nombre de cellules visitées à la recherche
- `nlist` — nombre de cellules Voronoi

**Outputs :** `results/ivfpq_nprobe_results.csv`, `results/ivfpq_nlist_results.csv`, + graphiques

---

### `pca_benchmark.ipynb` — Compression PCA (comparatif)
Évalue l'impact d'une réduction de dimension PCA (128D → 32D) sur les trois méthodes.

**Ce que mesure ce notebook :**
- Recall@10, latence, mémoire et build time **avec vs sans PCA** pour HNSW, LSH et IVF-PQ
- Variance expliquée par les composantes principales
- Courbe de Pareto globale

**Outputs :** `results/pca_comparative_results.csv`, `results/pca_impact_all_methods.png`, `results/pca_tradeoff_scatter.png`, `results/pca_deltas_summary.png`

---

## Structure du dossier

```
benchmark/
├── data/                    # Dataset SIFT1M — à générer via download_data.py
│   ├── sift_base.fvecs      # 1 000 000 vecteurs de base (~488 MB)
│   ├── sift_query.fvecs     # 10 000 vecteurs de requête
│   ├── sift_groundtruth.ivecs
│   └── sift_learn.fvecs
├── results/                 # CSV et PNG générés à l'exécution (gitignorés)
├── download_data.py         # Script de téléchargement SIFT1M
├── hnsw_benchmark.ipynb
├── lsh_benchmark.ipynb
├── ivfpq_benchmark.ipynb
└── pca_benchmark.ipynb
```
