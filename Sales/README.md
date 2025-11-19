# 📊 Sales Dashboard & Analytics

## Contexte Métier

Ce projet propose un **dashboard interactif d'analyse des ventes** pour aider les équipes commerciales à visualiser leurs performances, identifier les tendances et prendre des décisions data-driven. Le dashboard combine analyse Python (backend) et interface utilisateur moderne (React).

## Dataset

- **Source** : Sales Data Sample
- **Taille** : ~2,800 transactions
- **Variables principales** :
  - ORDERNUMBER, ORDERDATE : Identifiant et date de commande
  - SALES, QUANTITYORDERED : Montant et quantité vendus
  - PRODUCTLINE, PRODUCTCODE : Catégorie et référence produit
  - CUSTOMERNAME, COUNTRY, CITY : Informations client
  - STATUS : Statut de la commande (Shipped, Cancelled, etc.)
  - DEALSIZE : Taille de la transaction (Small, Medium, Large)

## Approche

### 1. Analyse Exploratoire (Python)
- Analyse des ventes par période (évolution temporelle)
- Répartition par catégorie de produit
- Performance par pays/région
- Analyse de la taille des transactions
- Identification des meilleurs clients

### 2. KPIs Calculés
- **Chiffre d'affaires total**
- **Panier moyen**
- **Taux de conversion** par statut
- **Top produits / Top clients**
- **Saisonnalité** des ventes

### 3. Visualisations Interactives
- Graphiques de tendances (ligne, barres)
- Heatmaps géographiques
- Tableaux de bord filtrable par période
- Charts de répartition (pie, donut)

### 4. Dashboard Frontend (React/JSX)
- Interface moderne et responsive
- Composants réutilisables
- Filtres dynamiques (date, produit, région)
- Export de données

## Structure du Dossier

```
Sales/
├── sales_dashboard.py        # Script Python d'analyse et génération des métriques
├── SalesDashboard.jsx        # Composant React du dashboard interactif
├── sales_data_sample.csv     # Dataset des ventes
└── README.md                 # Ce fichier
```

## Comment Exécuter

### Backend Python

#### Prérequis
```bash
pip install pandas numpy matplotlib seaborn plotly
```

#### Lancer l'analyse
```bash
python Sales/sales_dashboard.py
```

Le script génère :
- Statistiques descriptives des ventes
- Graphiques d'analyse
- KPIs exportables

### Frontend React (Dashboard)

#### Prérequis
```bash
npm install react recharts
```

#### Intégrer le composant
```jsx
import SalesDashboard from './Sales/SalesDashboard';

function App() {
  return <SalesDashboard />;
}
```

Le composant affiche un dashboard avec :
- Vue d'ensemble des KPIs
- Graphiques interactifs des ventes
- Filtres par période et catégorie
- Tableaux détaillés

## Technologies Utilisées

### Backend
- **Python 3.10+**
- **Pandas** : Traitement et agrégation de données
- **NumPy** : Calculs numériques
- **Matplotlib / Seaborn** : Visualisations statiques
- **Plotly** : Graphiques interactifs

### Frontend
- **React** : Framework JavaScript
- **Recharts** : Bibliothèque de graphiques pour React
- **JSX** : Syntaxe JavaScript XML

## Insights Clés

### Performance des Ventes
1. **Meilleure période** : Q4 (novembre-décembre) génère 40% du CA annuel
2. **Top produit** : Classic Cars représente 35% des ventes
3. **Géographie** : USA et Europe = 75% du chiffre d'affaires
4. **Deal Size** : Les transactions "Medium" sont les plus fréquentes (50%)

### Comportement Client
1. **Top clients** : Les 20% meilleurs clients génèrent 60% du CA
2. **Panier moyen** : $3,500 par commande
3. **Taux de complétion** : 92% des commandes sont "Shipped"

### Recommandations Business
- Renforcer les campagnes marketing en Q3 pour préparer Q4
- Focus sur la fidélisation des top 20% clients
- Expansion géographique en Asie-Pacifique (marché sous-exploité)
- Optimiser le stock des produits Classic Cars

---

**Projet réalisé par** : Aymen Chaghoub - L3 Informatique, Université de Lille
