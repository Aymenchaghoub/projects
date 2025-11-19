# 📊 Data Science & Machine Learning Portfolio

**Aymen Chaghoub** | Étudiant L3 Informatique - Université de Lille  
🎯 Recherche : Stage 12 semaines + Alternance en Data Science / IA / Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Aymenchaghoub/data-science-portfolio.svg)](https://github.com/Aymenchaghoub/data-science-portfolio/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Aymenchaghoub/data-science-portfolio.svg)](https://github.com/Aymenchaghoub/data-science-portfolio/network)

---

## 👨‍🎓 À Propos

Étudiant passionné de data science et d'intelligence artificielle en L3 Informatique à l'Université de Lille. Ce portfolio présente mes compétences en analyse de données, machine learning et développement de solutions data-driven à travers **8 projets complets** couvrant différents domaines (e-commerce, santé, immobilier, médias, NLP).

**Compétences clés** : Python, Scikit-learn, Pandas, NLP, Classification/Régression, Dashboards, Deep Learning

---

## 🚀 Projets Principaux

| # | Projet | Dossier | Description | Stack Technique |
|---|--------|---------|-------------|-----------------|
| 1 | **E-Commerce Churn Prediction** | [`Business/`](Business/) | Prédiction du churn clients et analyse des facteurs de risque pour une plateforme e-commerce (2000 clients) | Python, Pandas, Scikit-learn, Logistic Regression, Random Forest |
| 2 | **Healthcare Test Results Classification** | [`health_care/`](health_care/) | Classification multi-classes des résultats de tests médicaux (Normal/Abnormal/Inconclusive) sur 55,500 patients | Python, Scikit-learn, Classification multi-classes, Régression logistique multinomiale |
| 3 | **USA Housing Price Prediction** | [`USA/`](USA/) | Prédiction des prix immobiliers aux États-Unis avec analyse de corrélation et régression (R² ~0.92) | Python, Scikit-learn, Ridge Regression, Feature Engineering |
| 4 | **Netflix Content Analysis** | [`netflix/`](netflix/) | Analyse du catalogue Netflix et prédiction du type de contenu (8800+ titres) | Python, Pandas, EDA approfondie, Classification, Visualisations |
| 5 | **Twitter Sentiment Analysis** | [`x/`](x/) | Classification de sentiments sur 1.6M tweets avec NLP (accuracy ~80%) | Python, NLTK, TF-IDF, Naive Bayes, SVM, Word Clouds |
| 6 | **Sales Dashboard** | [`Sales/`](Sales/) | Dashboard interactif d'analyse des ventes avec backend Python et frontend React | Python, Pandas, React, Plotly, Visualisations business |
| 7 | **Fake News Detection** | [`fake/`](fake/) | Détection automatique de fake news avec NLP (accuracy ~98%) | Python, NLTK, TF-IDF, Passive Aggressive Classifier |
| 8 | **Bike Sharing App** | [`velo/`](velo/) | Application de partage de vélos avec prédiction de la demande et interface web | Python, Flask/Streamlit, ML Prédictif, Optimisation |

### 📓 Notebooks Jupyter

Les analyses complètes sont disponibles dans le dossier [`notebooks/`](notebooks/) :
- `01_ecommerce_churn_analysis.ipynb` : Analyse de churn e-commerce (39 cellules)
- `02_healthcare_test_results_classification.ipynb` : Classification médicale (44 cellules)

---

## 🛠️ Stack Technique

### **Data Science & Machine Learning**
- **Python 3.10+** : Langage principal
- **Pandas & NumPy** : Manipulation et analyse de données
- **Scikit-learn** : ML (classification, régression, clustering)
- **NLTK** : Natural Language Processing
- **Matplotlib, Seaborn, Plotly** : Visualisations

### **Développement Web & Dashboards**
- **Flask / Streamlit** : Applications web
- **React (JSX)** : Frontend interactif
- **Dash** : Dashboards analytiques

### **Installation**

1. **Cloner le repository**
```bash
git clone https://github.com/Aymenchaghoub/data-science-portfolio.git
cd data-science-portfolio
```

2. **Créer un environnement virtuel (recommandé)**
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
venv\Scripts\activate     # Sur Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

### **Lancer un Projet Spécifique**

Chaque projet possède son propre README avec des instructions détaillées. Exemples :

```bash
# E-Commerce Churn
jupyter notebook notebooks/01_ecommerce_churn_analysis.ipynb

# Healthcare Classification
jupyter notebook notebooks/02_healthcare_test_results_classification.ipynb

# USA Housing Price Prediction
python USA/Predect_housing.py

# Netflix Analysis
python netflix/project_netflix.py

# Twitter Sentiment Analysis
python x/project_x.py

# Sales Dashboard
python Sales/sales_dashboard.py

# Fake News Detection
python fake/fake_news_detection.py

# Bike Sharing App
python velo/app.py
```

---

## 📁 Organisation du Dépôt

```
data-science-portfolio/
│
├── 📂 Business/                  # Projet 1: E-Commerce Churn Prediction
│   ├── E Commerce Customer Insights and Churn Dataset.csv
│   └── README.md
│
├── 📂 health_care/               # Projet 2: Healthcare Test Classification
│   ├── healthcare_dataset.csv
│   └── README.md
│
├── 📂 USA/                       # Projet 3: USA Housing Price Prediction
│   ├── Predect_housing.py
│   ├── USA_Housing.csv
│   ├── best_house_price_model_ridge_regression.pkl
│   ├── house_eda.png
│   ├── house_model_results.png
│   └── README.md
│
├── 📂 netflix/                   # Projet 4: Netflix Content Analysis
│   ├── project_netflix.py
│   ├── netflix_titles.csv
│   ├── netflix_eda.png
│   └── README.md
│
├── 📂 x/                         # Projet 5: Twitter Sentiment Analysis
│   ├── project_x.py
│   ├── Tweets.csv
│   ├── twitter_eda.png
│   └── README.md
│
├── 📂 Sales/                     # Projet 6: Sales Dashboard
│   ├── sales_dashboard.py
│   ├── SalesDashboard.jsx
│   └── README.md
│
├── 📂 fake/                      # Projet 7: Fake News Detection
│   ├── fake_news_detection.py
│   ├── Fake.csv / True.csv
│   └── README.md
│
├── 📂 velo/                      # Projet 8: Bike Sharing App
│   ├── app.py
│   ├── web.py
│   └── README.md
│
├── 📂 notebooks/                 # Notebooks Jupyter d'analyse complète
│   ├── 01_ecommerce_churn_analysis.ipynb
│   └── 02_healthcare_test_results_classification.ipynb
│
├── 📂 archive_internal/          # Code interne et versions techniques
│   ├── src/                     # Versions backend des projets
│   ├── outputs/                 # Artefacts techniques
│   └── visualizations/          # Graphiques internes
│
├── requirements.txt              # Dépendances Python
├── LICENSE                       # Licence MIT
└── README.md                     # Ce fichier
```

### 📖 Note sur l'Organisation

- **Projets principaux** : Chaque dossier de projet contient son code, données et README spécifique
- **`notebooks/`** : Analyses Jupyter complètes pour les projets Business et health_care
- **`archive_internal/`** : Code technique et versions avancées (non essentiel pour les recruteurs)

---

## 🎯 Compétences Démontrées

### **Machine Learning**
- ✅ Classification binaire et multi-classes (Logistic Regression, Random Forest, SVM, Naive Bayes)
- ✅ Régression (Linear, Ridge, Lasso, Gradient Boosting)
- ✅ Feature engineering et sélection de variables
- ✅ Validation croisée et optimisation des hyperparamètres
- ✅ Évaluation de modèles (accuracy, F1, ROC-AUC, RMSE, R²)

### **Natural Language Processing**
- ✅ Preprocessing textuel (tokenization, stopwords, lemmatization)
- ✅ Vectorisation (TF-IDF, CountVectorizer, n-grams)
- ✅ Sentiment analysis et classification de texte
- ✅ Word clouds et visualisations textuelles

### **Data Analysis & Visualization**
- ✅ EDA complète (distributions, corrélations, outliers)
- ✅ Visualisations statistiques (Matplotlib, Seaborn, Plotly)
- ✅ Dashboards interactifs (React, Streamlit, Dash)
- ✅ Business Intelligence et insights actionnables

### **Développement**
- ✅ Code propre et documenté (docstrings, comments, READMEs)
- ✅ Notebooks Jupyter structurés (sections markdown + code)
- ✅ Applications web (Flask, React)
- ✅ Git/GitHub et bonnes pratiques de versioning

---

## 📫 Contact & Liens

**Aymen Chaghoub**  
Étudiant L3 Informatique - Université de Lille  
Recherche : Stage 12 semaines + Alternance en Data Science / ML

- 🌐 **GitHub** : [@Aymenchaghoub](https://github.com/Aymenchaghoub)
- 💼 **LinkedIn** : [Aymen Chaghoub](https://www.linkedin.com/in/aymen-chaghoub-1a7796279/)
- 📧 **Email** : ensm.chaghoub.aymen@gmail.com

---

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

<div align="center">

### ⭐ Si ce portfolio vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐

**Développé avec passion par Aymen Chaghoub**  
*Portfolio Data Science - Novembre 2025*

</div>