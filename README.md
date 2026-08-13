# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

 **ISPM — Madagascar** ([www.ispm-edu.com](https://www.ispm-edu.com))

---

### **1. Informations sur le Groupe**


#### Membre 1

- Nom : RAKOTOARIVELO 
- Prénom(s) : Anjaratiana Mendrika 
- Classe : IMTICIA 4
- Numéro : 25
- Rôle : Ingénieur données, nettoyage et prétraitement

#### Membre 2

- Nom : RABEHARISAINA 
- Prénoms : Mamy Fanojo
- Classe : IMTICIA 4  
- Numéro : 22  
- Rôle : Ingénieur machine learning, entraînement des modèles

#### Membre 3

- Nom    : RANDRIAMORASATA  
- Prénom : Ezra 
- Classe : IMTICIA 4
- numéro : 8  
- Rôles  : Analyste évaluation & validation

  #### Membre 4

- Nom    : RALISAONA  
- Prénom : Solonirina Fanomezana 
- Classe : IMTICIA 4
- numéro : 21  
- Rôles  : Documentation, préparation de la vidéo de présentation

#### Membre 5

- Nom    : ANDRIAMALALA RANJA
- Prénom : Ken Andi
- Classe : IMTICIA 4
- numéro : 13
- Rôles  : Chef de projet, coordination générale et intégration des livrables




### **2. Résumé du Travail**

#### Problématique

Atlantic Haven Hotels subit un taux d'annulation de **25,8 %** sur ses réservations, ce qui désorganise
la gestion des chambres, fausse les prévisions de revenu et entraîne des pertes lorsque des chambres
bloquées restent finalement vides. Prédire suffisamment tôt qu'une réservation en cours risque d'être
annulée permettrait à l'hôtel d'agir en amont (relance, confirmation, sur-réservation maîtrisée) plutôt
que de subir l'annulation.

#### Méthodologie adoptée

Nous avons suivi une chaîne complète et **sans fuite de données** : exploration (EDA), prétraitement par
`ColumnTransformer` + `Pipeline` (imputation, standardisation, encodage one-hot appris uniquement sur le
train), **feature engineering** de variables disponibles au moment de la réservation, **validation
temporelle** (holdout chronologique des 20 % de réservations les plus récentes + `TimeSeriesSplit` pour le
tuning), une **baseline** en régression logistique, la comparaison avec **Random Forest** et
**HistGradientBoosting**, l'optimisation des hyperparamètres, puis **l'optimisation du seuil de décision**
pour maximiser le F1 de la classe « annulation ».

#### Résultats obtenus

Sur le jeu de validation temporelle (1 600 réservations les plus récentes du train), le modèle final —
une **régression logistique régularisée (C = 0,05)** avec seuil de décision **0,22** — atteint :

- **F1-score (classe annulation) = 0,480**
- Précision = 0,356 · Rappel = 0,740 · ROC-AUC = 0,664

Découverte principale : les annulations sont surtout expliquées par des **conditions contractuelles et
comportementales** (absence d'acompte, tarif remboursable, long délai de réservation, historique
d'annulation du client), et non par la région ou le profil du client en tant que tel.

#### Mots-clés

classification binaire · annulation hôtelière · validation temporelle · F1-score · feature engineering ·
régression logistique · optimisation du seuil · déséquilibre de classes

---

### **3. Contenu du Repository**

- **notebook.ipynb** : code complet (EDA, prétraitement, modélisation, évaluation), exécutable de bout en bout ;
- **submission.csv** : prédictions sur `reservations_test.csv` (2 000 lignes, 3 colonnes) ;
- **README.md** : présent rapport ;
- **requirements.txt** : dépendances nécessaires à la reproduction.

**🔗 Liens utiles :**

- [**LIEN VERS LA VIDÉO DE PRÉSENTATION**](https://www.youtube.com/) *(à compléter)*
- [Lien vers le dépôt GitHub](https://github.com/) *(à compléter)*

---

### **4. Résultats de Modélisation**

Tous les résultats ci-dessous sont mesurés sur **le même holdout temporel** (20 % de réservations les plus
récentes du train, du 2024-11-28 au 2025-05-24). Sauf mention contraire, les métriques sont au **seuil 0,5**
afin de comparer les modèles à conditions égales ; le modèle final est ensuite évalué au **seuil optimisé**.

| Modèle | Paramètres principaux | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Régression logistique — baseline (sans FE) | `class_weight=balanced` | 0,454 | 0,362 | 0,608 | 0,663 |
| Random Forest (avec FE) | `n_estimators=300, balanced` | 0,043 | 0,312 | 0,023 | 0,644 |
| HistGradientBoosting (avec FE) | `balanced` (défaut) | 0,454 | 0,396 | 0,534 | 0,644 |
| Régression logistique + FE (seuil 0,5) | `C=0,05` | 0,462 | 0,369 | 0,620 | 0,662 |
| **Modèle final — LogReg + FE (seuil 0,22)** | `C=0,05`, seuil optimisé | **0,480** | 0,356 | 0,740 | 0,664 |

> Note : les Random Forest s'effondrent au seuil 0,5 (rappel ≈ 2–3 %) car leurs probabilités sont mal
> calibrées sur ces données ; à seuil optimisé elles remontent à ~0,48 de F1, sans dépasser la régression
> logistique, qui reste plus stable, plus interprétable et plus rapide.

**Seuil de décision retenu :** **0,22** (choisi pour maximiser le F1 de la classe 1 sur la validation).

**Justification du choix du modèle final :** la régression logistique offre le **meilleur ROC-AUC**
(0,664), le **meilleur F1**, un **seuil stable**, un **temps d'entraînement < 1 s**, et une
**interprétabilité directe** par ses coefficients — atouts décisifs pour un déploiement défendable et un
diagnostic métier. Les modèles d'arbres n'apportent aucun gain de performance tout en étant plus lourds et
plus opaques ; sur un signal modéré (AUC ≈ 0,66) et un décalage temporel, le modèle linéaire généralise
mieux.

---

### **5. Réponses aux Questions d'Analyse**

#### **Q1. Pourquoi le F1-score plutôt que l'accuracy ?**

La cible est déséquilibrée : seules **25,8 %** des réservations sont annulées. Un modèle trivial prédisant
« jamais annulé » obtiendrait déjà **74 % d'accuracy** tout en étant inutile (rappel = 0 sur la classe qui
nous intéresse). Le **F1-score de la classe 1** combine précision et rappel sur les annulations : il ne
récompense pas l'inaction et mesure réellement la capacité à détecter les annulations.

#### **Q2. Faux positif ou faux négatif : lequel est le plus grave ?**

- **Faux positif (FP)** : on prédit une annulation qui n'arrive pas → une relance inutile, un léger coût
  commercial, un risque de sur-réservation.
- **Faux négatif (FN)** : on rate une annulation réelle → chambre bloquée puis libérée trop tard,
  perte de revenu directe, impossibilité de revendre.

Dans l'hôtellerie, le **faux négatif est généralement plus coûteux** (revenu perdu non récupérable), ce qui
justifie un seuil bas privilégiant le **rappel** (0,740). Nuance : au-delà d'un certain volume, trop de FP
sature les équipes et dégrade l'expérience client ; le seuil doit donc rester un curseur ajustable selon la
capacité opérationnelle.

#### **Q3. Quelles features de feature engineering ont le plus aidé ?**

Par rapport à la baseline (F1 = 0,454), le feature engineering porte le F1 à **0,462** (seuil 0,5) puis
**0,480** (seuil optimisé). Les variables créées les plus utiles (permutation importance) :

- **`delai_long`** (délai ≥ 60 j) et le délai continu : 2ᵉ variable la plus importante ;
- **`taux_annul_client`** = annulations passées / réservations passées : capture l'habitude d'annulation
  sans utiliser la cible courante ;
- **`reservation_directe`** (agent_id manquant) : les réservations directes annulent moins (20,5 % vs 29,7 %) ;
- variables de saison/mois d'arrivée et **`prix_total_par_nuit`**.

#### **Q4. Pourquoi un découpage aléatoire serait trompeur ?**

Les données sont **chronologiques** et le test est **postérieur** au train (train jusqu'au 2025-05-24, test
du 2025-05-24 au 2025-12-31). Un découpage aléatoire mélangerait des réservations futures dans
l'entraînement (**fuite temporelle**) et donnerait un score optimiste irréaliste. Nous avons donc utilisé
un **holdout chronologique** : entraînement sur les **80 % anciennes** réservations (jusqu'au 2024-11-28),
validation sur les **20 % récentes** (2024-11-28 → 2025-05-24), et un **`TimeSeriesSplit`** (4 plis) pour le
tuning — reproduisant fidèlement « entraîner sur le passé, prédire le futur ».

#### **Q5. Quels scénarios sont le plus associés aux annulations ?**

*(circonstances observables, jamais une population « intrinsèquement à risque »)*

- réservation **sans acompte** ET **tarif remboursable** (aucun engagement financier) ;
- **long délai** entre réservation et arrivée (plus de temps pour changer d'avis) ;
- client dont **l'historique montre déjà des annulations** ;
- réservation via **plateforme en ligne** plutôt qu'en direct ou en entreprise.

Ces facteurs se **cumulent** : c'est leur combinaison (remboursable + sans acompte + long délai) qui
caractérise les dossiers les plus volatils.

#### **Q6. Traitement des valeurs manquantes et des catégories jamais vues ?**

Tout est encapsulé dans un `Pipeline` **appris uniquement sur le train** : imputation numérique par la
**médiane**, catégorielle par le **mode**, encodage `OneHotEncoder(handle_unknown="ignore")` qui gère sans
erreur toute catégorie absente de l'entraînement, et `min_frequency=10` qui regroupe les modalités rares.
`agent_id` manquant est traité comme information métier (`reservation_directe`), pas comme du bruit. Aucune
statistique du test n'est utilisée pour ajuster les transformations → **pas de fuite**.

#### **Q7. Quelle action lorsqu'une réservation présente une forte probabilité d'annulation ?**

Une intervention **proportionnée et non punitive** : relance personnalisée de confirmation, rappel des
conditions, incitation douce à verser un acompte ou à passer sur un tarif partiellement engageant, offre de
flexibilité (modification plutôt qu'annulation). En parallèle, alimenter une **sur-réservation maîtrisée**
sur les créneaux les plus à risque. **Ne jamais annuler automatiquement** la réservation du client.

#### **Q8. Performances comparables selon les régions ?**

Non, il existe des écarts. Sur la validation, le F1 par région varie de **0,408 (Campania)** et **0,420
(Toscana)** à **0,558 (Trentino-Alto-Adige)** et **0,545 (Sicilia)**. Ces écarts doivent être lus avec
prudence : certaines régions comptent peu de réservations en validation (Sardegna n = 67, Puglia n = 99),
ce qui rend leur F1 instable. Aucune région n'est « intrinsèquement » plus annulatrice ; ce sont des
contextes touristiques/saisonniers différents.

#### **Q9. Analyse des erreurs**

**5 faux positifs** (prédits annulés, en réalité honorés) — profil type : **tarif remboursable + aucun
acompte**, souvent via plateforme/agence :
`R006922`, `R009568`, `R001349` (délai 429 j !), `R008395`, `R001316`.

**5 faux négatifs** (annulations ratées) — profil type : **tarif non remboursable + acompte total**, via
site hôtel/entreprise, donc « en apparence sûrs » :
`R009204`, `R001223`, `R008135`, `R001185`, `R002885` (9 948 € !).

*Raisons.* Le modèle s'appuie fortement sur l'acompte et la flexibilité tarifaire ; il se trompe quand un
client engagé annule quand même (FN) ou quand un client non engagé honore finalement (FP). Ce sont des
comportements individuels que les variables disponibles n'expliquent pas.

*Piste d'amélioration.* Enrichir les données par des signaux comportementaux (nombre de connexions au
dossier, délai depuis la dernière modification, historique de paiement) et tester une calibration des
probabilités ; explorer des features d'interaction explicites (remboursable × sans acompte × délai).

---

### **6. Conclusion et Recommandations**

Le modèle détecte **74 % des annulations** (rappel) au prix d'une précision de 0,356, pour un **F1 = 0,480**
sur des données futures — un résultat honnête compte tenu d'un signal modéré (AUC ≈ 0,66) sur données
synthétiques. Ses forces : simplicité, rapidité, interprétabilité, absence de fuite, validation temporelle
rigoureuse. Sa limite : la précision reste modérée, donc l'outil est un **système d'alerte d'aide à la
décision**, pas un automate.

**Recommandation opérationnelle finale :** utiliser le score comme **file de priorisation** des relances de
confirmation sur les réservations à risque (sans acompte, remboursables, à long délai), avec un seuil
**ajustable** selon la capacité des équipes — plus bas en basse saison (maximiser le rappel), plus haut en
forte affluence (limiter les fausses alertes). Réévaluer le modèle chaque trimestre sur les données les plus
récentes.

---

### **7. Reproductibilité**

- version de Python : 3.11
- principales bibliothèques : scikit-learn 1.8, pandas 3.0, numpy, matplotlib
- graine(s) aléatoire(s) : `RANDOM_STATE = 42` (numpy + tous les estimateurs)
- procédure d'exécution : placer les CSV à côté du notebook puis « Run All » (`jupyter nbconvert --to notebook --execute notebook.ipynb`)
- durée approximative d'entraînement : < 10 secondes (tuning inclus)
- environnement : local / Jupyter (aucune dépendance externe à scikit-learn)

---

### **8. Bibliographie**

- Documentation scikit-learn — Pipeline, ColumnTransformer, TimeSeriesSplit, métriques de classification.
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011.
- Outil d'IA générative (assistant Claude) : aide à la structuration du pipeline, à la rédaction du code
  commenté et du présent rapport ; l'ensemble des résultats a été calculé par exécution réelle du code.
