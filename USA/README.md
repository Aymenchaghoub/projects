# 🏠 USA Housing Price Prediction

## Contexte Métier

Ce projet vise à **prédire les prix des logements aux États-Unis** en utilisant des algorithmes de machine learning. L'objectif est d'aider les acheteurs, vendeurs et agents immobiliers à estimer la valeur des propriétés basée sur leurs caractéristiques (localisation, superficie, nombre de chambres, etc.).

## Dataset

- **Source** : USA Housing Dataset
- **Taille** : ~5,000 propriétés
- **Variables principales** :
  - Average Area Income (revenu moyen de la zone)
  - Average Area House Age (âge moyen des maisons)
  - Average Area Number of Rooms (nombre moyen de pièces)
  - Average Area Number of Bedrooms (nombre moyen de chambres)
  - Area Population (population de la zone)
  - Price (prix de vente - **variable cible**)

## Approche

### 1. Analyse Exploratoire (EDA)
- Visualisation des distributions de prix et des features
- Analyse des corrélations entre variables
- Identification des outliers et patterns
- Résultats sauvegardés dans `house_eda.png`

### 2. Feature Engineering
- Analyse de la multicolinéarité
- Normalisation des features si nécessaire

### 3. Modélisation
Plusieurs modèles de régression testés :
- **Linear Regression** (baseline)
- **Ridge Regression** (régularisation L2)
- **Lasso Regression** (régularisation L1)
- **Random Forest Regressor**

### 4. Évaluation
- Métriques : RMSE, R², MAE
- Validation croisée
- **Meilleur modèle** : Ridge Regression (sauvegardé dans `best_house_price_model_ridge_regression.pkl`)

## Résultats

- **R² Score** : ~0.92 (le modèle explique 92% de la variance des prix)
- **RMSE** : Faible erreur moyenne de prédiction
- Le modèle Ridge Regression offre le meilleur compromis entre précision et généralisation
- Visualisations des résultats dans `house_model_results.png`
- Rapport détaillé dans `rapport_final.txt`

## Structure du Dossier

```
USA/
├── Predect_housing.py                              # Script principal d'analyse et modélisation
├── USA_Housing.csv                                 # Dataset
├── best_house_price_model_ridge_regression.pkl     # Modèle entraîné sauvegardé
├── model_metadata.pkl                              # Métadonnées du modèle
├── house_eda.png                                   # Visualisations de l'analyse exploratoire
├── house_model_results.png                         # Graphiques des résultats du modèle
├── rapport_final.txt                               # Rapport complet de l'analyse
└── README.md                                       # Ce fichier
```

## Comment Exécuter

### Prérequis
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Lancer l'analyse
```bash
python USA/Predect_housing.py
```

Le script va :
1. Charger et analyser le dataset
2. Entraîner les modèles de régression
3. Générer les visualisations (EDA et résultats)
4. Sauvegarder le meilleur modèle

### Utiliser le modèle entraîné
```python
import pickle

# Charger le modèle
with open('USA/best_house_price_model_ridge_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Prédire (exemple)
# features = [income, house_age, rooms, bedrooms, population]
prediction = model.predict([[75000, 5.5, 7, 3, 35000]])
print(f"Prix prédit : ${prediction[0]:,.2f}")
```

## Technologies Utilisées

- **Python 3.10+**
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Scikit-learn** : Modèles de régression, métriques
- **Matplotlib / Seaborn** : Visualisations
- **Pickle** : Sauvegarde du modèle

## Insights Clés

1. **Average Area Income** est le facteur le plus prédictif du prix
2. La **population de la zone** et le **nombre de chambres** influencent significativement le prix
3. Le modèle Ridge Regression évite l'overfitting grâce à la régularisation
4. Les prédictions sont fiables pour des propriétés dans la gamme de prix du dataset

---

**Projet réalisé par** : Aymen Chaghoub - L3 Informatique, Université de Lille
