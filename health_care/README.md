# 🏥 Healthcare Test Results Classification

## 🎯 Contexte

Ce projet analyse un **dataset synthétique d'admissions hospitalières** pour prédire les résultats de tests médicaux et identifier les facteurs associés aux résultats anormaux. L'objectif est de développer un système de classification multi-classes capable de distinguer les résultats **Normal**, **Abnormal** et **Inconclusive**.

⚠️ **Note importante** : Ce dataset est synthétique et utilisé uniquement à des fins pédagogiques. Les conclusions ne constituent pas des recommandations médicales.

### Problématique

Les établissements de santé génèrent d'énormes volumes de données. Ce projet explore comment le machine learning peut aider à :

- **Prédire les résultats de tests médicaux** avant leur réalisation
- **Identifier les profils de patients** associés à des résultats anormaux
- **Optimiser l'allocation des ressources** en anticipant les besoins
- **Comprendre les patterns** entre conditions médicales et résultats de tests

---

## 📁 Description du Dataset

**Source** : Healthcare Dataset (Kaggle)

### Caractéristiques

- **Taille** : 55,500 lignes (admissions hospitalières)
- **Variables** : 15 colonnes
- **Qualité** : 0% de valeurs manquantes
- **Période** : Données synthétiques représentant des admissions 2024

### Variables Principales

| Variable | Type | Description |
|----------|------|-------------|
| `Name` | Identifiant | Nom du patient (supprimé lors du traitement) |
| `Age` | Numérique | Âge du patient (13-89 ans) |
| `Gender` | Catégorielle | Genre (Male, Female) |
| `Blood Type` | Catégorielle | Groupe sanguin (A+, A-, B+, B-, AB+, AB-, O+, O-) |
| `Medical Condition` | Catégorielle | Condition médicale principale (Diabetes, Cancer, Arthritis, Hypertension, Obesity, Asthma) |
| `Date of Admission` | Date | Date d'admission à l'hôpital |
| `Doctor` | Identifiant | Nom du médecin (supprimé lors du traitement) |
| `Hospital` | Identifiant | Nom de l'hôpital (supprimé lors du traitement) |
| `Insurance Provider` | Catégorielle | Assureur (Aetna, Blue Cross, Cigna, UnitedHealthcare, Medicare) |
| `Billing Amount` | Numérique | Montant de la facturation ($) |
| `Room Number` | Identifiant | Numéro de chambre (supprimé lors du traitement) |
| `Admission Type` | Catégorielle | Type d'admission (Emergency, Elective, Urgent) |
| `Discharge Date` | Date | Date de sortie de l'hôpital |
| `Medication` | Texte | Médicament prescrit |
| `Test Results` | Catégorielle | **Résultat du test (Normal, Abnormal, Inconclusive)** ⭐ |

### Variable Cible

**`Test Results`** : Classification multi-classes (3 catégories)
- `Normal` : ~33.4% (18,517 cas)
- `Abnormal` : ~33.6% (18,627 cas)
- `Inconclusive` : ~33.1% (18,356 cas)

**Distribution** : Dataset parfaitement équilibré entre les 3 classes.

---

## 🔬 Workflow du Projet

### 1. **Exploration des Données (EDA)**

#### Nettoyage
- Suppression des colonnes identifiantes (Name, Doctor, Hospital, Room Number)
- Conversion des dates en format datetime
- Création de features temporelles (durée de séjour)

#### Analyses Démographiques
- Distribution de l'âge : moyenne de 51.5 ans (13-89 ans)
- Répartition équilibrée par genre (50% Male / 50% Female)
- 8 groupes sanguins représentés de manière homogène

#### Analyses Médicales
- **6 conditions médicales** principales, distribution équilibrée
- **3 types d'admission** : Elective (33.6%), Urgent (33.5%), Emergency (32.9%)
- **5 assureurs** principaux avec répartition homogène

#### Analyses Croisées
- Test Results par Medical Condition (heatmap)
- Test Results par Admission Type
- Test Results par Insurance Provider
- Distribution des montants de facturation par résultat de test

**Visualisations** : Histogrammes, boxplots, barplots, heatmaps, pie charts

### 2. **Feature Engineering**

- **Création de features temporelles** :
  - `Length of Stay` : Durée de séjour en jours (Discharge Date - Admission Date)
  
- **Encodage des variables catégorielles** :
  - One-Hot Encoding pour : Gender, Blood Type, Medical Condition, Admission Type, Insurance Provider
  - Passage de 8 features à ~20+ features encodées

- **Features finales sélectionnées** :
  - Age, Gender, Blood Type, Medical Condition, Admission Type, Insurance Provider, Billing Amount, Length of Stay

### 3. **Modélisation Machine Learning**

Deux modèles de classification multi-classes ont été développés :

#### **Modèle 1 : Régression Logistique Multinomiale (Baseline)**
- Algorithme linéaire adapté aux problèmes multi-classes
- Solver : `lbfgs` (Limited-memory BFGS)
- Features normalisées avec StandardScaler
- Interprétable et rapide

#### **Modèle 2 : Random Forest Classifier (Avancé)**
- Ensemble de 100 arbres de décision
- Max depth : 15, Min samples split : 10
- Capture les interactions non-linéaires
- Fournit l'importance des features

**Configuration du train/test** :
- 80% entraînement (44,400 patients) / 20% test (11,100 patients)
- Stratification pour conserver la distribution des 3 classes
- Random state fixé (42) pour la reproductibilité

### 4. **Évaluation des Modèles**

**Métriques utilisées** :
- **Accuracy** : Taux global de prédictions correctes
- **F1-Score (Macro)** : Moyenne non pondérée des F1 par classe
- **F1-Score (Micro)** : F1 calculé globalement
- **F1-Score (Weighted)** : Moyenne pondérée par le nombre d'exemples

**Visualisations** :
- Matrices de confusion 3x3
- Rapports de classification détaillés par classe
- Graphiques comparatifs des performances
- Importance des features (Random Forest)

---

## 📈 Résultats Clés

### Performances des Modèles

| Modèle | Accuracy | F1-Score (Macro) | F1-Score (Weighted) |
|--------|----------|------------------|---------------------|
| **Logistic Regression** | ~0.33 | ~0.33 | ~0.33 |
| **Random Forest** | ~0.33 | ~0.33 | ~0.33 |

### Observation Importante

Les performances sont **proches du hasard** (~33% pour 3 classes équilibrées), ce qui suggère que :

1. **Dataset synthétique** : Les résultats de tests sont probablement générés aléatoirement, sans corrélation forte avec les features
2. **Absence de signal prédictif** : Aucune relation causale évidente entre les caractéristiques des patients et les résultats de tests
3. **Valeur pédagogique** : Le projet démontre néanmoins toutes les étapes d'un workflow ML complet

### Top Features (Random Forest)

L'analyse de l'importance des features révèle :

1. **Length of Stay** (durée de séjour)
2. **Billing Amount** (montant de facturation)
3. **Age** (âge du patient)
4. Features encodées de **Medical Condition**
5. Features encodées d'**Admission Type**

> ⚠️ Même si le modèle identifie ces features comme importantes, les performances faibles indiquent qu'elles ne permettent pas de prédire efficacement les résultats dans ce dataset synthétique.

---

## 💡 Insights et Interprétation

### Analyses Comparatives

Les moyennes par type de résultat montrent des valeurs très similaires :

| Métrique | Normal | Abnormal | Inconclusive |
|----------|--------|----------|--------------|
| **Âge moyen** | ~51.5 ans | ~51.5 ans | ~51.5 ans |
| **Durée de séjour** | ~15 jours | ~15 jours | ~15 jours |
| **Facturation** | ~$25,500 | ~$25,500 | ~$25,500 |

Cette homogénéité confirme l'absence de patterns discriminants.

### Applications Potentielles (Contexte Réel)

Dans un environnement avec des **données réelles** et après **validation médicale**, ce type d'approche pourrait servir à :

1. **Priorisation des ressources** : Identifier les patients à risque nécessitant des examens complémentaires
2. **Optimisation des parcours** : Adapter les protocoles de soins selon les profils
3. **Gestion administrative** : Anticiper les besoins en lits, équipements, personnel
4. **Support à la décision** : Fournir des insights aux équipes médicales (sans remplacer leur expertise)

### Limites Reconnues

- ⚠️ Dataset synthétique sans valeur médicale réelle
- ⚠️ Corrélations ≠ causalité (même avec de bonnes performances)
- ⚠️ Un modèle ML ne peut jamais remplacer l'expertise clinique
- ⚠️ Validation et certification médicales obligatoires avant tout usage réel

---

## 🛠️ Technologies Utilisées

- **Python 3.10+**
- **Pandas** : Manipulation et analyse de données
- **NumPy** : Calculs numériques et matrices
- **Matplotlib / Seaborn** : Visualisations statistiques
- **Scikit-learn** : 
  - Preprocessing (StandardScaler, One-Hot Encoding)
  - Modèles (LogisticRegression, RandomForestClassifier)
  - Métriques (accuracy, F1-score, confusion matrix)
  - Train/test split

---

## 📂 Structure du Projet

```
data-science-portfolio/
│
├── health_care/
│   ├── healthcare_dataset.csv
│   └── README.md (ce fichier)
│
├── notebooks/
│   ├── 01_ecommerce_churn_analysis.ipynb
│   └── 02_healthcare_test_results_classification.ipynb  ⭐
│
└── README.md
```

---

## 🚀 Comment Utiliser ce Projet

### Prérequis

Installer les dépendances :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Exécution

1. Cloner le repository
2. Ouvrir le notebook Jupyter :

```bash
jupyter notebook notebooks/02_healthcare_test_results_classification.ipynb
```

3. Exécuter les cellules séquentiellement pour reproduire l'analyse

---

## 📊 Améliorations Futures

### Modélisation Avancée

- [ ] Tester XGBoost, LightGBM, CatBoost
- [ ] Optimisation des hyperparamètres (GridSearchCV, Optuna)
- [ ] Validation croisée stratifiée k-fold (5 ou 10 folds)
- [ ] Ensembles de modèles (Voting, Stacking)

### Feature Engineering

- [ ] Interactions entre features (polynomiales)
- [ ] Binning de variables continues (age groups)
- [ ] Features temporelles avancées (jour de la semaine, mois)
- [ ] Agrégations par groupes (moyenne par condition, etc.)

### Interprétabilité

- [ ] SHAP values pour expliquer les prédictions
- [ ] LIME pour interprétabilité locale
- [ ] Partial Dependence Plots
- [ ] Analyse de sensibilité

### Déploiement

- [ ] API REST avec Flask/FastAPI
- [ ] Dashboard interactif avec Streamlit
- [ ] Containerisation avec Docker
- [ ] CI/CD pour l'entraînement et le déploiement

---

## 👨‍🎓 Auteur

**Étudiant L3 Informatique** - Université de Lille  
Portfolio Data Science / Machine Learning  
Recherche : Stage (12 semaines) puis Alternance

---

## 📝 Licence

Ce projet est développé dans un cadre académique et de portfolio professionnel.

---

## 🙏 Remerciements

- Dataset : Kaggle Healthcare Dataset
- Ressources : Documentation Scikit-learn, Towards Data Science, Stack Overflow
- Inspiration : Projets open-source de la communauté ML

---

## 📞 Contact

Pour toute question sur ce projet ou collaboration :
- Portfolio GitHub : [Votre lien]
- LinkedIn : [Votre lien]
- Email : [Votre email]
