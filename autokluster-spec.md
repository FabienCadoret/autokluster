# autokluster - Spécification projet

## Problème à résoudre

Quand on veut regrouper des textes automatiquement (clustering), deux problèmes majeurs :

1. **On doit deviner le nombre de groupes (k)** — scikit-learn demande "combien de clusters tu veux ?" alors qu'on ne sait pas
2. **Pas de métrique fiable** — silhouette score et inertie sont approximatifs pour les embeddings de texte

Aujourd'hui les gens bricolent : ils testent k=5, k=10, k=15... regardent à l'œil, utilisent la méthode du coude qui marche mal.

## Solution

Une lib Python qui fait :

```python
from autokluster import cluster

result = cluster(embeddings)

print(result.k)               # 7 (trouvé automatiquement via eigen-gap)
print(result.cohesion_ratio)  # 0.84 (score de qualité fiable)
print(result.labels)          # [0, 2, 1, 0, 3, ...]
```

## Innovations clés

| Fonctionnalité | Outils existants | autokluster |
|----------------|------------------|------------------|
| Estimation automatique de k | Manuel | Via eigen-gap |
| Métrique de qualité fiable | Silhouette (limité) | Cohesion Ratio |
| Optimisé pour embeddings texte | Générique | Oui |

### Cohesion Ratio

Nouvelle métrique du papier arXiv [2511.19350](https://arxiv.org/abs/2511.19350) :
- Mesure compacité interne des clusters vs séparation entre eux
- N'existe dans aucune lib Python actuelle
- Plus fiable que silhouette pour embeddings de texte

### Eigen-gap

Méthode pour estimer k automatiquement :
- Analyse les valeurs propres du Laplacien
- Le "gap" (saut) entre valeurs propres indique le nombre naturel de clusters

## Concurrence

| Outil | Limite |
|-------|--------|
| scikit-learn SpectralClustering | Pas d'estimation auto de k, pas de Cohesion Ratio |
| huggingface/text-clustering | UMAP + HDBSCAN, pas spectral |
| wq2012/SpectralCluster | Audio (speaker diarization), pas texte |
| GateNLP/cluster-embeddings | Script simple, pas de métrique custom |

**Gap identifié** : Aucune lib n'implémente Cohesion Ratio + eigen-gap pour clustering de texte.

## Public cible

1. **Data Scientists** — segmentation de données textuelles sans catégories prédéfinies
2. **Développeurs RAG/LLM** — pré-organisation de chunks documentaires
3. **Chercheurs/Analystes** — exploration de corpus (topic discovery)

Profil type : sait utiliser Python + sentence-transformers, veut du clustering sans bricoler les hyperparamètres.

## Cas d'usage concrets

### Triage de tickets support
```
500 tickets/jour → embeddings → autokluster → 8 groupes découverts
```

### Chunking intelligent pour RAG
```
2000 chunks → autokluster → groupes thématiques → recherche plus précise
```

### Analyse de feedback produit
```
10 000 avis clients → autokluster → clusters "UX confuse", "App lente", etc.
```

### Détection de doublons sémantiques
```
"Comment réinitialiser mon mot de passe"
"Procédure de reset password"
→ Même cluster = doublons à fusionner
```

## Architecture

```
autokluster/
    src/
        domain/
            spectral_clustering.py     # Algorithme spectral
            cohesion_ratio.py          # Métrique originale
            eigen_gap.py               # Estimation auto de k
        application/
            clustering_service.py      # Orchestration
            embedding_loader.py        # Chargement embeddings
        infrastructure/
            numpy_adapter.py           # Opérations matricielles
            file_adapter.py            # I/O fichiers
        exposition/
            cli.py                     # Interface ligne de commande
    tests/
    pyproject.toml
```

## Stack technique

- Python 3.10+
- NumPy/SciPy (opérations matricielles, eigendecomposition)
- scikit-learn (k-means final, prétraitement)
- Click (CLI)
- sentence-transformers (optionnel: génération embeddings)

## Algorithme simplifié

1. **Input** : matrice d'embeddings (n_samples, n_features)
2. **Matrice de similarité** : cosine similarity entre tous les vecteurs
3. **Laplacien normalisé** : L = D^(-1/2) * (D - W) * D^(-1/2)
4. **Eigendecomposition** : calcul des k plus petites valeurs/vecteurs propres
5. **Eigen-gap** : trouver le k optimal via le plus grand saut entre valeurs propres
6. **K-means** : clustering dans l'espace des vecteurs propres
7. **Cohesion Ratio** : calcul de la métrique de qualité
8. **Output** : labels, k, cohesion_ratio

## Formules mathématiques

### Cohesion Ratio (ρ_C)

```
ρ_C = μ_I / μ_G
```

- **μ_G** = moyenne globale de similarité : `1/C(n,2) × Σ S_ij` pour toutes les paires
- **μ_I** = moyenne intra-cluster : `1/P × Σ_k Σ_(i,j∈C_k) S_ij`
- ρ_C = 1 → cohésion égale au bruit de fond (mauvais)
- ρ_C > 1 → clusters cohérents (bon)
- Quand `sampled=true` (n > τ), ρ_C est calculé sur le sous-échantillon spectral (τ points). L'échantillon aléatoire est statistiquement représentatif du dataset complet.

### Laplacien normalisé

```
L_sym = I - D^(-1/2) × S × D^(-1/2)
```

- S = matrice de similarité cosinus (valeurs négatives clippées à 0)
- D = matrice diagonale des degrés : `D_ii = Σ_j S_ij`

### Eigen-gap adaptatif

```
δ_i = |λ_i - λ_(i-1)| / [(1/w) × Σ_(j=i-w..i-1) λ_j + ε]
```

Seuil data-driven :
```
k = premier i où δ_i > E[δ] × (1 + σ[δ]/E[δ] + ε)
```

## Paramètres

### Configurables (API Python)

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `window_size` | int | 3 | Fenêtre glissante (w) pour eigen-gap |
| `epsilon` | float | 1e-10 | Stabilité numérique (ε) |
| `k` | int \| None | None | Forcer k (None = auto via eigen-gap) |
| `min_k` | int | 2 | Nombre minimum de clusters |
| `max_k` | int | 50 | Nombre maximum de clusters |
| `random_state` | int \| None | None | Seed pour reproductibilité |

### Fixes (basés sur le papier)

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `sampling_threshold` | 1000 | Seuil τ au-delà duquel l'échantillonnage adaptatif s'active |
| `n_replicates` | log₂(n) × 10 | Nombre de sous-échantillons pour grands datasets |

## CLI

### Usage

```bash
autokluster --input embeddings.npy --output clusters.json
autokluster --input embeddings.npy --k 5  # forcer k=5
autokluster --input embeddings.npy --format detailed
```

### Format de sortie (JSON)

**Standard** (`--format standard`, défaut) :
```json
{
  "k": 7,
  "labels": [0, 2, 1, 0, 3, ...],
  "cohesion_ratio": 1.84
}
```

**Détaillé** (`--format detailed`) :
```json
{
  "k": 7,
  "labels": [0, 2, 1, 0, 3, ...],
  "cohesion_ratio": 1.84,
  "cluster_sizes": [45, 32, 28, 21, 19, 15, 12],
  "eigenvalues": [0.0, 0.012, 0.018, ...],
  "eigengap_index": 7,
  "n_samples": 172,
  "sampled": false
}
```

## Données de test

### Fixtures synthétiques
- Blobs bien séparés (k=3, 5, 10) pour validation basique
- Embeddings aléatoires (k inconnu) pour tester l'eigen-gap

### Datasets publics (benchmarks)
- **20 Newsgroups** : ~18k documents, 20 catégories
- **AG News** : ~120k documents, 4 catégories
- Embeddings pré-calculés avec `all-MiniLM-L6-v2`

## Roadmap

### Phase 1 - MVP (2 semaines)
- SpectralClustering avec Laplacien normalisé
- Cohesion Ratio
- CLI basique : `autokluster --input embeddings.npy --output clusters.json`
- Tests unitaires sur 2 datasets publics (20 Newsgroups, AG News)

### Phase 2 - v1.0 (1 semaine)
- Estimation automatique de k via eigen-gap
- Support formats multiples (npy, csv, parquet)
- Échantillonnage adaptatif pour grands datasets
- Documentation et exemples

### Phase 3 - Production (1 semaine)
- Publication PyPI
- Intégration optionnelle sentence-transformers
- Benchmarks vs K-Means, HDBSCAN
- GitHub Actions CI/CD

## Ressources

### Papier source
- arXiv [2511.19350](https://arxiv.org/abs/2511.19350) - Cohesion Ratio
- Licence : CC-BY 4.0 (code réutilisable)

### Références techniques
- [Tutorial von Luxburg](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf) - Spectral clustering
- [scikit-learn SpectralClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)

## Promotion post-publication

1. Reddit r/MachineLearning, r/Python
2. Hacker News "Show HN"
3. Twitter/X (tag @huggingface, @_akhaliq)
4. Article Medium/Dev.to
5. PR intégration LangChain ou sentence-transformers
6. Notebook Kaggle avec démo

## Critères de succès

- `pip install autokluster` fonctionne
- README avec exemple en 5 lignes
- Benchmark montrant avantage vs K-Means classique
- 100+ stars GitHub dans les 3 premiers mois
