# CLAUDE.md - autokluster

## Projet

Librairie Python de clustering automatique pour embeddings textuels avec estimation automatique de k via eigen-gap et métrique Cohesion Ratio.

## Références

### Spécification projet
- **[autokluster-spec.md](autokluster-spec.md)** : Spécification complète du projet (architecture, algorithmes, paramètres, roadmap)

### Article source
- **arXiv 2511.19350** : https://arxiv.org/abs/2511.19350
  - Introduit le Cohesion Ratio (ρ_C)
  - Licence CC-BY 4.0 (code réutilisable)

### Documentation technique
- **Tutorial von Luxburg** (spectral clustering) : https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf
- **scikit-learn SpectralClustering** : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
- **scipy.linalg.eigh** : https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html

## Architecture

Architecture hexagonale (Ports & Adapters) :

```
src/autokluster/
├── domain/           # Logique métier pure (pas de dépendances externes)
│   ├── spectral_clustering.py
│   ├── cohesion_ratio.py
│   └── eigen_gap.py
├── application/      # Orchestration, use cases
│   ├── clustering_service.py
│   └── embedding_loader.py
├── infrastructure/   # Adapters (numpy, fichiers)
│   ├── numpy_adapter.py
│   └── file_adapter.py
└── exposition/       # Interface utilisateur
    └── cli.py
```

## Formules mathématiques

### Cohesion Ratio
```
ρ_C = μ_I / μ_G

μ_G = (1 / C(n,2)) × Σ S_ij        # moyenne globale
μ_I = (1 / P) × Σ_k Σ_(i,j∈C_k) S_ij  # moyenne intra-cluster
```
- ρ_C = 1 → cohésion égale au bruit (mauvais)
- ρ_C > 1 → clusters cohérents (bon)

### Laplacien normalisé
```
L_sym = I - D^(-1/2) × S × D^(-1/2)

S = matrice de similarité cosinus (clip négatifs à 0)
D = matrice diagonale des degrés : D_ii = Σ_j S_ij
```

### Eigen-gap adaptatif
```
δ_i = |λ_i - λ_(i-1)| / [(1/w) × Σ_(j=i-w..i-1) λ_j + ε]

Seuil : k = premier i où δ_i > E[δ] × (1 + σ[δ]/E[δ] + ε)
```

## Conventions de code

### Nommage
- Variables et fonctions : `snake_case` (PEP8)
- Classes : `PascalCase`
- Constantes : `UPPER_SNAKE_CASE`

### Style
- Pas de commentaires dans le code (code auto-documenté)
- Exception : `# TODO: [TICKET-ID]` temporaires
- Ligne max : 100 caractères
- Imports triés (ruff)

### Types
- Type hints obligatoires pour fonctions publiques
- `NDArray[np.float64]` pour les matrices numpy
- `int | None` plutôt que `Optional[int]`

## Bonnes pratiques

### NumPy/SciPy
- Utiliser `scipy.linalg.eigh` (pas `numpy.linalg.eig`) pour matrices symétriques
- Clip valeurs négatives de similarité à 0 avant Laplacien
- Ajouter epsilon (1e-10) pour stabilité numérique dans divisions
- Normaliser les vecteurs propres avant k-means

### Clustering
- Toujours valider que embeddings est 2D : `(n_samples, n_features)`
- min_k=2 obligatoire (un seul cluster n'a pas de sens)
- max_k ne doit pas dépasser n_samples - 1
- Fixer random_state pour reproductibilité des tests

### Tests
- Fixtures avec `make_blobs` de sklearn pour clusters synthétiques
- Tester edge cases : k=2, k proche de n_samples
- Vérifier que cohesion_ratio > 1 pour blobs bien séparés
- Tests paramétrés pour différentes valeurs de k

### Performance
- Échantillonnage adaptatif si n > 1000
- Eigendecomposition partielle (k valeurs propres, pas toutes)
- Éviter boucles Python sur matrices, utiliser vectorisation numpy

## Checklist

Voir [CHECKLIST.md](CHECKLIST.md) pour le suivi du développement.

## Commandes

```bash
# Tests
pytest tests/ -v

# Tests avec couverture
pytest tests/ --cov=autokluster --cov-report=term-missing

# Lint
ruff check src/

# Format
ruff format src/

# CLI
autokluster --input embeddings.npy --output clusters.json
```

## Dépendances

| Package | Usage |
|---------|-------|
| numpy | Opérations matricielles |
| scipy | Eigendecomposition (linalg.eigh) |
| scikit-learn | KMeans, make_blobs (tests) |
| click | CLI |
| sentence-transformers | Optionnel : génération embeddings |
