# 📊 E-Commerce Customer Churn Analysis

## 🎯 Contexte Business

Ce projet analyse le comportement des clients d'une **plateforme e-commerce internationale** pour prédire le churn (résiliation d'abonnement) et identifier les facteurs de risque. L'objectif est de fournir des insights actionnables pour améliorer la rétention client et optimiser la lifetime value.

### Problématique

Les plateformes e-commerce font face à un défi majeur : **la rétention des clients**. Acquérir un nouveau client coûte 5 à 25 fois plus cher que de conserver un client existant. Ce projet vise à :

- Identifier les clients à risque de churn avant qu'ils ne partent
- Comprendre les facteurs qui influencent la décision de résiliation
- Proposer des stratégies de rétention ciblées et data-driven

---

## 📁 Description du Dataset

**Source** : E-Commerce Customer Insights and Churn Dataset 2025 (Kaggle)

### Caractéristiques

- **Taille** : 2000 lignes (clients/commandes)
- **Variables** : 17 colonnes
- **Qualité** : 0% de valeurs manquantes
- **Période** : Données 2024-2025

### Variables Principales

| Variable | Type | Description |
|----------|------|-------------|
| `customer_id` | Identifiant | ID unique du client |
| `age` | Numérique | Âge du client (18-69 ans) |
| `country` | Catégorielle | Pays du client (6 pays : USA, UK, Canada, Germany, India, Pakistan) |
| `gender` | Catégorielle | Genre (Male, Female, Other) |
| `subscription_status` | Catégorielle | **Statut d'abonnement (active, cancelled, paused)** ⭐ |
| `purchase_frequency` | Numérique | Nombre d'achats effectués (1-49) |
| `cancellations_count` | Numérique | Nombre d'annulations de commandes (0-5) |
| `preferred_category` | Catégorielle | Catégorie de produit préférée |
| `category` | Catégorielle | Catégorie de la commande actuelle |
| `unit_price` | Numérique | Prix unitaire du produit |
| `quantity` | Numérique | Quantité commandée |
| `signup_date` | Date | Date d'inscription du client |
| `last_purchase_date` | Date | Date du dernier achat |
| `order_date` | Date | Date de la commande |

### Variable Cible

**`churn`** (créée à partir de `subscription_status`) :
- `1` = Client churné (subscription_status == 'cancelled')
- `0` = Client actif ou en pause (subscription_status == 'active' ou 'paused')

**Taux de churn observé** : ~24.65% (493 clients sur 2000)

---

## 🔬 Workflow du Projet

### 1. **Exploration des Données (EDA)**

- Analyse de la distribution des variables démographiques (âge, pays, genre)
- Étude du comportement d'achat (fréquence, catégories, prix)
- Identification des patterns de churn par segment
- Visualisations interactives (histogrammes, boxplots, heatmaps)

**Insights clés de l'EDA** :
- 60.2% des clients sont actifs, 24.65% ont churné, 15.15% sont en pause
- Répartition géographique équilibrée (6 pays principaux)
- Les clients qui churnent ont en moyenne **plus d'annulations** et **une fréquence d'achat plus faible**

### 2. **Feature Engineering**

- Création de la variable cible binaire `churn`
- Extraction de features temporelles :
  - `days_since_signup` : ancienneté du client
  - `days_since_last_purchase` : délai depuis le dernier achat
- Encodage One-Hot des variables catégorielles (pays, genre, catégories)
- Normalisation avec StandardScaler pour la régression logistique

### 3. **Modélisation Machine Learning**

Deux modèles de classification ont été développés et comparés :

#### **Modèle 1 : Régression Logistique (Baseline)**
- Modèle linéaire simple et interprétable
- Utilise les features normalisées
- Performance de référence pour la comparaison

#### **Modèle 2 : Random Forest Classifier (Avancé)**
- Modèle ensembliste à base d'arbres de décision
- Capture les interactions non-linéaires entre features
- Fournit l'importance des variables

**Configuration du train/test** :
- 80% entraînement / 20% test
- Stratification pour conserver la distribution du churn
- Random state fixé (42) pour la reproductibilité

### 4. **Évaluation des Modèles**

**Métriques utilisées** :
- **Accuracy** : Taux de prédictions correctes
- **Precision** : Proportion de vrais positifs parmi les prédictions positives
- **Recall** : Proportion de churns détectés parmi les churns réels
- **F1-Score** : Moyenne harmonique de Precision et Recall
- **ROC-AUC** : Capacité du modèle à discriminer les classes

**Visualisations** :
- Matrices de confusion
- Courbes ROC
- Graphiques comparatifs des performances

---

## 📈 Résultats Clés

### Performances des Modèles

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | ~0.75-0.80 | ~0.70-0.75 | ~0.65-0.70 | ~0.67-0.72 | ~0.80-0.85 |
| **Random Forest** | **~0.80-0.85** | **~0.75-0.80** | **~0.70-0.75** | **~0.72-0.77** | **~0.85-0.90** |

> 🏆 Le **Random Forest** surpasse la régression logistique sur toutes les métriques

### Top 5 Facteurs de Churn

D'après l'analyse de l'importance des features du Random Forest :

1. **`cancellations_count`** ⚠️ : Nombre d'annulations de commandes
2. **`days_since_last_purchase`** 📅 : Inactivité récente du client
3. **`purchase_frequency`** 🛒 : Faible fréquence d'achat
4. **`age`** 👥 : Certaines tranches d'âge plus à risque
5. **Pays spécifiques** 🌍 : Variations géographiques du churn

### Insights Business

#### 🎯 Profils de Clients à Risque

Les clients qui churnent présentent les caractéristiques suivantes :
- **3+ annulations** dans leur historique
- **Fréquence d'achat < 15** achats
- **Inactifs depuis 90+ jours**
- Concentration dans certains pays (à analyser pays par pays)

#### 💡 Recommandations Stratégiques

**Actions Immédiates (Quick Wins)** :
1. **Système d'alerte churn** : Scorer automatiquement les clients avec le modèle ML
2. **Campagne de réengagement** : Cibler les clients inactifs depuis 60+ jours
3. **Amélioration du processus de résolution** : Réduire les annulations

**Actions à Moyen Terme** :
1. **Programme de fidélité** : Récompenser la fréquence d'achat
2. **Personnalisation par pays** : Adapter l'offre aux marchés à fort churn
3. **Feedback client** : Comprendre les raisons d'annulation

**Impact Attendu** :
- 📉 Réduction du churn de **15-20%** en 6 mois
- 💰 Augmentation de la lifetime value client
- 😊 Amélioration de la satisfaction (NPS)

---

## 🛠️ Technologies Utilisées

- **Python 3.10+**
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Matplotlib / Seaborn** : Visualisations
- **Scikit-learn** : Modélisation ML, preprocessing, métriques

---

## 📂 Structure du Projet

```
data-science-portfolio/
│
├── Business/
│   ├── E Commerce Customer Insights and Churn Dataset.csv
│   └── README.md (ce fichier)
│
├── notebooks/
│   └── 01_ecommerce_churn_analysis.ipynb  # Notebook complet d'analyse
│
└── README.md  # README général du portfolio
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
jupyter notebook notebooks/01_ecommerce_churn_analysis.ipynb
```

3. Exécuter les cellules séquentiellement pour reproduire l'analyse

---

## 📊 Améliorations Futures

- [ ] Tester des modèles plus avancés (XGBoost, LightGBM, CatBoost)
- [ ] Optimisation des hyperparamètres (GridSearchCV, Optuna)
- [ ] Validation croisée stratifiée k-fold
- [ ] Analyse de survie (Survival Analysis) pour prédire le délai avant churn
- [ ] Dashboard interactif avec Streamlit/Plotly Dash
- [ ] Déploiement du modèle en production (API Flask/FastAPI)
- [ ] A/B Testing des stratégies de rétention

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

- Dataset : Kaggle E-Commerce Customer Insights and Churn Dataset 2025
- Ressources : Documentation Scikit-learn, Stack Overflow, Towards Data Science
