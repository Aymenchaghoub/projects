# 📰 Fake News Detection

## Contexte Métier

Ce projet vise à **détecter automatiquement les fake news** (fausses informations) en utilisant des techniques de NLP et de machine learning. À l'heure de la désinformation sur les réseaux sociaux, ce système aide à classifier les articles en "vrais" ou "faux" pour lutter contre la propagation de fausses nouvelles.

## Dataset

- **Source** : Fake News Dataset
- **Taille** : ~45,000 articles
- **Composition** :
  - `Fake.csv` : ~23,000 articles de fake news
  - `True.csv` : ~22,000 articles d'informations vérifiées
- **Variables principales** :
  - title : Titre de l'article
  - text : Contenu complet de l'article
  - subject : Catégorie/sujet de l'article
  - date : Date de publication

## Approche

### 1. Preprocessing NLP
- Nettoyage du texte (suppression ponctuation, caractères spéciaux)
- Conversion en minuscules
- Tokenization
- Suppression des stop words (mots courants sans valeur sémantique)
- Stemming / Lemmatization

### 2. Feature Engineering
- **Vectorisation textuelle** :
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - CountVectorizer
  - N-grams (unigrammes, bigrammes, trigrammes)
- **Features additionnelles** :
  - Longueur du titre et du texte
  - Nombre de mots en majuscules (indicateur de clickbait)
  - Présence de mots sensationnalistes

### 3. Modélisation
Algorithmes de classification binaire testés :
- **Naive Bayes** (MultinomialNB - excellent pour classification textuelle)
- **Logistic Regression** avec TF-IDF
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **Passive Aggressive Classifier** (adapté aux gros volumes)

### 4. Évaluation
- Métriques : Accuracy, Precision, Recall, F1-Score
- Matrice de confusion
- Validation croisée
- Analyse des erreurs de classification

## Résultats

- **Accuracy** : ~92-98% selon le modèle
- **Meilleur modèle** : Passive Aggressive Classifier avec TF-IDF (~98%)
- **Precision** : ~97% (peu de faux positifs)
- **Recall** : ~96% (détecte la majorité des fake news)
- Les modèles s'appuient principalement sur le vocabulaire et le style d'écriture

## Structure du Dossier

```
fake/
├── fake_news_detection.py    # Script principal de classification
├── Fake.csv                  # Articles de fake news
├── True.csv                  # Articles d'informations vérifiées
└── README.md                 # Ce fichier
```

## Comment Exécuter

### Prérequis
```bash
pip install pandas numpy scikit-learn nltk
```

### Télécharger les ressources NLTK (première fois)
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Lancer la détection
```bash
python fake/fake_news_detection.py
```

Le script va :
1. Charger et combiner les datasets Fake.csv et True.csv
2. Préprocesser les textes (nettoyage, tokenization)
3. Entraîner les modèles de classification
4. Afficher les métriques de performance
5. Tester le modèle sur de nouveaux articles

### Utiliser le modèle pour prédire
```python
from fake_news_detection import predict_fake_news

article = "Titre sensationnel : Une découverte incroyable..."
prediction = predict_fake_news(article)
print(f"Prédiction : {'FAKE' if prediction == 1 else 'RÉEL'}")
```

## Technologies Utilisées

- **Python 3.10+**
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Scikit-learn** : TF-IDF, modèles de classification, métriques
- **NLTK** : Preprocessing NLP (tokenization, stopwords)

## Insights Clés

### Caractéristiques des Fake News
1. **Vocabulaire sensationnaliste** : Usage fréquent de mots comme "shocking", "unbelievable", "scandal"
2. **Titres clickbait** : Titres excessivement longs ou courts
3. **Ponctuation excessive** : Utilisation de !!! et ???
4. **Sources vagues** : Absence de citations vérifiables
5. **Style émotionnel** : Appel aux émotions plutôt qu'aux faits

### Performance par Sujet
- Les fake news politiques sont les plus faciles à détecter
- Les fake news scientifiques nécessitent plus de contexte
- Le modèle performe mieux sur les articles récents (vocabulaire actuel)

### Limites Reconnues
- ⚠️ Le modèle détecte le **style**, pas la **véracité factuelle**
- ⚠️ Sensible aux biais du dataset d'entraînement
- ⚠️ Ne remplace pas le fact-checking humain
- ⚠️ Nécessite mise à jour régulière (évolution du langage)

### Applications Pratiques
- **Plateformes sociales** : Flagging automatique de contenu suspect
- **Médias** : Aide aux journalistes pour vérification
- **Éducation** : Outil pédagogique de media literacy
- **Entreprises** : Protection contre la désinformation

---

**Projet réalisé par** : Aymen Chaghoub - L3 Informatique, Université de Lille
