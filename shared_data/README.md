# Données Compressées Partagées

Ce dossier contient les vecteurs SIFT1M compressés, générés par le notebook IVF-PQ (Valentin).

## Fichiers

| Fichier | Description | Shape | Dtype |
|---------|-------------|-------|-------|
| `base_compressed.npy` | 1M vecteurs de base compressés (stockage réel) | (1000000, 128) | uint8 |
| `query_compressed.npy` | 10k vecteurs de requête compressés (stockage réel) | (10000, 128) | uint8 |
| `base_reconstructed.npy` | 1M vecteurs reconstruits en float32 (pour FAISS) | (1000000, 128) | float32 |
| `query_reconstructed.npy` | 10k vecteurs reconstruits en float32 (pour FAISS) | (10000, 128) | float32 |
| `compression_params.npy` | Paramètres de compression (v_min et v_max par dimension) | (2, 128) | float32 |

## Méthode de compression

**Quantification uniforme 8 bits par dimension** (float32 → uint8) :
- Taux de compression : **4x** en mémoire (488 Mo → 122 Mo)
- `v_min` et `v_max` calculés **par dimension** sur `base_vectors` (128 valeurs chacun)
- Les queries sont compressées avec ces mêmes paramètres (pas de fuite)

> **Important** : les méthodes ANN (FAISS) travaillent sur des vecteurs float32.
> Charger `base_reconstructed.npy` et `query_reconstructed.npy` pour les benchmarks.
> Les fichiers `*_compressed.npy` (uint8) servent à mesurer le gain mémoire réel.

## Comment charger les données (pour Sedik et chatodit)

```python
import numpy as np

SHARED_DATA = "../shared_data/"   # adapter le chemin selon ton dossier

# Pour les benchmarks ANN (FAISS)
base_reconstructed  = np.load(SHARED_DATA + "base_reconstructed.npy")
query_reconstructed = np.load(SHARED_DATA + "query_reconstructed.npy")

# Pour mesurer le gain mémoire réel
base_compressed  = np.load(SHARED_DATA + "base_compressed.npy")
query_compressed = np.load(SHARED_DATA + "query_compressed.npy")

# Paramètres de compression (si besoin de recompresser d'autres données)
params = np.load(SHARED_DATA + "compression_params.npy")
v_min, v_max = params[0], params[1]  # shape (128,) chacun

print(f"Base reconstruite  : {base_reconstructed.shape}  dtype={base_reconstructed.dtype}")
print(f"Query reconstruite : {query_reconstructed.shape} dtype={query_reconstructed.dtype}")
print(f"Taux de compression : {base_reconstructed.nbytes / base_compressed.nbytes:.1f}x")
```

## À faire dans chaque notebook (LSH, HNSW)

Après avoir chargé les données, **recalculer la ground truth** sur `base_reconstructed` :

```python
import faiss

K_GT = 100  # ou K selon la méthode
index_flat = faiss.IndexFlatL2(128)
index_flat.add(base_reconstructed)
_, ground_truth_compressed = index_flat.search(query_reconstructed[:NUM_QUERIES], K_GT)
del index_flat
```

## Généré par

`Valentin/IVF_PQ_Benchmark.ipynb`, Section 11
