# Checklist de développement autokluster

## Phase 1 - MVP

### 1.1 Infrastructure (numpy_adapter.py)
- [x] `cosineSimilarity` : normalisation L2 + produit matriciel
- [x] `normalizedLaplacian` : L_sym = I - D^(-1/2) × S × D^(-1/2)
- [x] `eigendecomposition` : scipy.linalg.eigh pour les k plus petites valeurs propres

### 1.2 Infrastructure (file_adapter.py)
- [x] `writeJson` : sérialisation JSON avec gestion numpy arrays
- [x] `readNpy` : chargement fichiers .npy

### 1.3 Domain (spectral_clustering.py)
- [x] `computeSimilarityMatrix` : utilise cosineSimilarity, clip valeurs négatives à 0
- [x] `computeNormalizedLaplacian` : utilise normalizedLaplacian
- [x] `computeEigendecomposition` : utilise eigendecomposition, retourne SpectralResult
- [x] `clusterEigenvectors` : sklearn KMeans sur les vecteurs propres normalisés

### 1.4 Domain (cohesion_ratio.py)
- [x] `compute_global_mean_similarity` : μ_G = moyenne de toutes les paires
- [x] `compute_intra_cluster_mean_similarity` : μ_I = moyenne des paires intra-cluster
- [x] `compute_cohesion_ratio` : ρ_C = μ_I / μ_G

### 1.5 Application (embedding_loader.py)
- [ ] `loadNpy` : charge et valide format numpy
- [ ] `loadEmbeddings` : détecte extension, dispatch vers le bon loader

### 1.6 Application (clustering_service.py)
- [ ] `cluster` : orchestration complète avec k fixe
  - Calcul similarité → Laplacien → Eigendecomposition → KMeans → Cohesion Ratio
  - Retourne ClusterResult complet

### 1.7 Exposition (cli.py)
- [ ] `main` : charge embeddings, appelle cluster, écrit JSON
- [ ] Support --format standard/detailed

### 1.8 Tests Phase 1
- [ ] Tests unitaires numpy_adapter (3 fonctions)
- [ ] Tests unitaires spectral_clustering (4 méthodes)
- [ ] Tests unitaires cohesion_ratio (3 fonctions)
- [ ] Tests intégration clustering_service avec blobs synthétiques
- [ ] Test CLI avec fichier npy temporaire

---

## Phase 2 - v1.0

### 2.1 Domain (eigen_gap.py)
- [ ] `computeAdaptiveGaps` : δ_i = |λ_i - λ_(i-1)| / moyenne_fenêtre
- [ ] `computeGapThreshold` : seuil data-driven E[δ] × (1 + σ[δ]/E[δ] + ε)
- [ ] `findOptimalK` : premier i où δ_i > seuil, avec min_k/max_k

### 2.2 Application (clustering_service.py)
- [ ] Modifier `cluster` : si k=None, appeler eigen-gap pour déterminer k automatiquement
- [ ] Ajouter eigengapIndex dans ClusterResult

### 2.3 Infrastructure (file_adapter.py)
- [ ] `readCsv` : pandas ou numpy.loadtxt
- [ ] `readParquet` : pyarrow ou pandas

### 2.4 Application (embedding_loader.py)
- [ ] `loadCsv` : support fichiers CSV
- [ ] `loadParquet` : support fichiers Parquet
- [ ] Mettre à jour `loadEmbeddings` pour les nouveaux formats

### 2.5 Domain (échantillonnage adaptatif)
- [ ] Créer `adaptive_sampling.py` si n > 1000
- [ ] Implémenter n_replicates = log₂(n) × 10 sous-échantillons
- [ ] Agrégation des résultats

### 2.6 Tests Phase 2
- [ ] Tests eigen_gap avec eigenvalues synthétiques connues
- [ ] Tests auto-estimation k vs k réel sur blobs
- [ ] Tests chargement CSV/Parquet
- [ ] Tests échantillonnage grands datasets (>1000 samples)

---

## Phase 3 - Production

### 3.1 CI/CD
- [ ] GitHub Actions : tests + coverage sur push
- [ ] GitHub Actions : publication PyPI sur tag release
- [ ] Badge coverage dans README

### 3.2 Documentation
- [ ] Exemples d'usage dans README
- [ ] Docstrings pour API publique (cluster, ClusterResult)
- [ ] Notebook exemple Jupyter/Colab

### 3.3 Benchmarks
- [ ] Script benchmark vs sklearn KMeans
- [ ] Script benchmark vs HDBSCAN
- [ ] Résultats sur 20 Newsgroups embeddings
- [ ] Résultats sur AG News embeddings

### 3.4 Intégrations optionnelles
- [ ] Dépendance optionnelle sentence-transformers
- [ ] Fonction helper `clusterTexts(texts, model="all-MiniLM-L6-v2")`

### 3.5 Publication
- [ ] Vérifier pyproject.toml metadata
- [ ] `pip install autokluster` fonctionne localement
- [ ] Publication test sur TestPyPI
- [ ] Publication finale sur PyPI

### 3.6 Tests Phase 3
- [ ] Tests intégration sentence-transformers (si installé)
- [ ] Tests end-to-end CLI avec tous formats
- [ ] Vérifier couverture ≥ 80%
