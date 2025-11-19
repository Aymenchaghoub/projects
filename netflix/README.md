# 🎬 Netflix Content Analysis

## Contexte Métier

Ce projet analyse le **catalogue Netflix** pour comprendre les tendances de contenu, les patterns de production et prédire certaines caractéristiques des titres. L'objectif est d'extraire des insights business sur la stratégie de contenu de Netflix (types de contenu, pays producteurs, genres populaires, évolution temporelle).

## Dataset

- **Source** : Netflix Titles Dataset (Kaggle)
- **Taille** : ~8,800 titres (films et séries TV)
- **Variables principales** :
  - show_id, type (Movie/TV Show)
  - title, director, cast
  - country, date_added, release_year
  - rating, duration
  - listed_in (genres), description

## Approche

### 1. Analyse Exploratoire (EDA)
- Distribution des types de contenu (Movies vs TV Shows)
- Évolution temporelle des ajouts Netflix
- Analyse géographique (pays producteurs)
- Analyse des genres les plus populaires
- Analyse des durées et ratings
- Visualisations sauvegardées dans `netflix_eda.png`

### 2. Feature Engineering
- Extraction et nettoyage des données de date
- Parsing des genres multiples
- Traitement des données manquantes
- Encodage des variables catégorielles

### 3. Modélisation
Construction de modèles pour prédire :
- Le type de contenu (Movie vs TV Show)
- Le rating du contenu
- Classification par genre

Algorithmes utilisés :
- Logistic Regression
- Random Forest Classifier
- Naive Bayes (pour classification textuelle)

### 4. Visualisations Business
- Graphiques d'évolution temporelle
- Heatmaps de corrélations
- Word clouds des descriptions
- Résultats dans `netflix_model_results.png`

## Résultats

- **Croissance du contenu** : Augmentation significative des ajouts après 2015
- **Dominance des films** : ~70% du catalogue sont des films
- **Top pays producteurs** : États-Unis, Inde, Royaume-Uni
- **Genres populaires** : International Movies, Dramas, Comedies
- **Accuracy des modèles** : ~75-85% pour la classification du type de contenu

## Structure du Dossier

```
netflix/
├── project_netflix.py        # Script principal d'analyse et modélisation
├── netflix_titles.csv        # Dataset Netflix
├── netflix_eda.png           # Visualisations de l'analyse exploratoire
├── netflix_model_results.png # Graphiques des résultats des modèles
└── README.md                 # Ce fichier
```

## Comment Exécuter

### Prérequis
```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud
```

### Lancer l'analyse
```bash
python netflix/project_netflix.py
```

Le script va :
1. Charger et nettoyer le dataset Netflix
2. Effectuer l'analyse exploratoire complète
3. Entraîner les modèles de classification
4. Générer les visualisations (EDA et résultats)

## Technologies Utilisées

- **Python 3.10+**
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Scikit-learn** : Modèles de classification, preprocessing
- **Matplotlib / Seaborn** : Visualisations statistiques
- **WordCloud** : Nuages de mots pour analyse textuelle

## Insights Clés

### Business Intelligence
1. **Stratégie de contenu** : Netflix privilégie les films courts (90-120 min) et les séries de 1-2 saisons
2. **Internationalisation** : Forte croissance du contenu non-anglophone (Inde, Corée, Espagne)
3. **Genres tendance** : Documentaries et Stand-Up Comedy en forte progression
4. **Targeting** : Contenu majoritairement classé TV-MA et TV-14

### Modélisation
1. Le **pays de production** et le **genre** sont les meilleurs prédicteurs du type de contenu
2. Les **descriptions textuelles** contiennent des patterns distincts entre films et séries
3. La **durée** est un excellent indicateur : <100 min = film, sinon série

---

**Projet réalisé par** : Aymen Chaghoub - L3 Informatique, Université de Lille
