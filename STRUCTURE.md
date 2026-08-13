# Architecture du projet — Atlantic Haven Hotels

```
atlantic-haven-projet/
├── README.md                     # Rapport complet (canevas rempli avec les résultats réels)
├── requirements.txt              # Dépendances Python
├── notebook.ipynb                # Notebook exécutable de bout en bout (16 sections)
├── submission.csv                # Prédictions finales (2000 lignes, 3 colonnes)
├── .gitignore
│
├── data/                         # Données d'entrée (fournies)
│   ├── reservations_train.csv    #   8000 réservations + cible
│   ├── reservations_test.csv     #   2000 réservations à prédire
│   ├── sample_submission.csv     #   format de soumission attendu
│   └── data_dictionary.csv       #   dictionnaire des variables
│
├── src/
│   └── pipeline.py               # Script Python autonome (même logique que le notebook)
│
├── reports/
│   ├── script_video.md           # Script de présentation vidéo (3–5 min)
│   └── figures/
│       ├── fig_seuil_confusion.png   # Courbe F1 vs seuil + matrice de confusion
│       └── fig_importance.png        # Permutation importance + coefficients
│
└── docs/
    └── readme-model.md           # Canevas original du rapport (référence)
```

## Comment exécuter

1. Installer les dépendances : `pip install -r requirements.txt`
2. Ouvrir le notebook : `jupyter notebook notebook.ipynb`, puis **Kernel → Restart & Run All**
   (ou en ligne de commande : `jupyter nbconvert --to notebook --execute notebook.ipynb`)
3. Le notebook lit les CSV depuis `data/` et régénère `submission.csv` à la racine.

Alternative sans Jupyter : `python src/pipeline.py` (depuis la racine du projet).

## Principes de conception

- **Données** isolées dans `data/` — jamais modifiées.
- **Code** dans `notebook.ipynb` (livrable principal) et `src/pipeline.py` (version script).
- **Sorties** : `submission.csv` à la racine, figures dans `reports/figures/`.
- **Documentation** : `README.md` (rapport), `STRUCTURE.md` (ce fichier), `docs/` (canevas).
- Graine fixée (`RANDOM_STATE = 42`), aucune fuite de données, validation temporelle.
