# 🐦 Twitter Sentiment Analysis

## Contexte Métier

Ce projet analyse les **sentiments exprimés dans les tweets** pour classifier automatiquement les opinions (positif, négatif, neutre). L'objectif est de comprendre l'opinion publique sur des sujets variés et de développer un système de classification de sentiments basé sur le machine learning et le NLP (Natural Language Processing).

## Dataset

- **Source** : Twitter Sentiment Dataset
- **Taille** : ~1.6 million de tweets
- **Variables principales** :
  - tweet_id : Identifiant unique du tweet
  - text : Contenu textuel du tweet
  - sentiment : Label de sentiment (0 = négatif, 4 = positif)
  - user, date : Métadonnées du tweet

## Approche

### 1. Analyse Exploratoire (EDA)
- Distribution des sentiments dans le dataset
- Analyse de la longueur des tweets
- Mots les plus fréquents par sentiment
- Word clouds pour visualiser le vocabulaire
- Visualisations sauvegardées dans `twitter_eda.png` et `twitter_wordclouds.png`

### 2. Preprocessing NLP
- Nettoyage du texte (suppression URLs, mentions, hashtags)
- Tokenization
- Suppression des stop words
- Lemmatization / Stemming
- Vectorisation : TF-IDF et CountVectorizer

### 3. Feature Engineering
- Extraction de features textuelles :
  - Longueur du tweet
  - Nombre de hashtags, mentions, emojis
  - Présence de mots-clés de sentiment
- Création de n-grams (unigrammes, bigrammes)

### 4. Modélisation
Algorithmes de classification testés :
- **Naive Bayes** (baseline pour classification textuelle)
- **Logistic Regression** avec TF-IDF
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

### 5. Évaluation
- Métriques : Accuracy, Precision, Recall, F1-Score
- Matrice de confusion
- Courbes ROC-AUC
- Résultats dans `twitter_model_results.png`

## Résultats

- **Accuracy** : ~75-82% selon le modèle
- **Meilleur modèle** : Logistic Regression avec TF-IDF
- **F1-Score** : ~0.78 (bon équilibre precision/recall)
- Les modèles détectent efficacement les sentiments extrêmes (très positifs/négatifs)
- Difficulté avec les tweets sarcastiques ou ambigus

## Structure du Dossier

```
x/
├── project_x.py               # Script principal d'analyse et modélisation
├── Tweets.csv                 # Dataset de tweets
├── twitter_eda.png            # Visualisations de l'analyse exploratoire
├── twitter_model_results.png  # Graphiques des résultats des modèles
├── twitter_wordclouds.png     # Nuages de mots par sentiment
└── README.md                  # Ce fichier
```

## Comment Exécuter

### Prérequis
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud
```

### Télécharger les ressources NLTK (première fois)
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### Lancer l'analyse
```bash
python x/project_x.py
```

Le script va :
1. Charger et préprocesser les tweets
2. Effectuer l'analyse exploratoire et générer les word clouds
3. Entraîner les modèles de classification de sentiments
4. Générer les visualisations des résultats

## Technologies Utilisées

- **Python 3.10+**
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Scikit-learn** : TF-IDF, modèles de classification, métriques
- **NLTK** : Preprocessing NLP (tokenization, stopwords, lemmatization)
- **Matplotlib / Seaborn** : Visualisations
- **WordCloud** : Génération de nuages de mots

## Insights Clés

### Analyse Textuelle
1. **Vocabulaire positif** : "love", "great", "happy", "good", "awesome"
2. **Vocabulaire négatif** : "hate", "bad", "sad", "worst", "terrible"
3. **Longueur moyenne** : Tweets négatifs légèrement plus longs que les positifs
4. **Emojis** : Forte corrélation entre emojis positifs et sentiment positif

### Performance des Modèles
1. **TF-IDF** surpasse CountVectorizer pour cette tâche
2. **Bigrammes** améliorent la performance (captent la négation)
3. Les modèles linéaires (LogReg, SVM) performent mieux que les arbres pour du texte
4. **Stop words** : Leur suppression améliore légèrement l'accuracy

### Applications Pratiques
- Monitoring de la réputation de marque
- Analyse de campagnes marketing
- Détection de crises sur les réseaux sociaux
- Analyse de feedback produit

---

**Projet réalisé par** : Aymen Chaghoub - L3 Informatique, Université de Lille
