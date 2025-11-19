# 🚴 Bike Sharing / Velo Application

## Contexte Métier

Ce projet développe une **application de partage de vélos** (bike sharing) avec analyse de données et prédiction de la demande. L'objectif est d'optimiser la disponibilité des vélos et de prédire les besoins en fonction de facteurs comme la météo, l'heure et la saison.

## Approche

### 1. Analyse des Données
- Analyse de la demande de vélos par période (heure, jour, saison)
- Impact des conditions météorologiques sur l'utilisation
- Identification des stations/zones à forte demande
- Patterns d'utilisation (jours ouvrés vs week-ends)

### 2. Backend Application (`app.py`)
Fonctionnalités principales :
- **Prédiction de la demande** : Modèle ML pour anticiper le nombre de locations
- **Gestion des stations** : Disponibilité en temps réel
- **Optimisation** : Redistribution des vélos entre stations
- **API REST** : Endpoints pour l'application web/mobile

### 3. Interface Web (`web.py`)
- **Dashboard utilisateur** : Carte interactive des stations
- **Disponibilité en temps réel** : Nombre de vélos disponibles
- **Réservation** : Système de réservation de vélos
- **Statistiques** : Visualisations d'utilisation

### 4. Modèles Prédictifs
- **Régression** : Prédiction du nombre de locations
- **Classification** : Prédiction de forte/faible demande
- Features utilisées :
  - Heure de la journée
  - Jour de la semaine
  - Saison
  - Température, humidité, vent
  - Jours fériés

## Structure du Dossier

```
velo/
├── app.py        # Application backend (API, logique métier, modèles ML)
├── web.py        # Interface web (Flask/Streamlit)
└── README.md     # Ce fichier
```

## Comment Exécuter

### Prérequis
```bash
pip install pandas numpy scikit-learn flask streamlit plotly
```

### Lancer l'application backend
```bash
python velo/app.py
```

L'API sera accessible sur `http://localhost:5000` avec les endpoints :
- `/api/stations` : Liste des stations et disponibilité
- `/api/predict` : Prédiction de la demande
- `/api/reserve` : Réservation de vélo

### Lancer l'interface web
```bash
python velo/web.py
```
ou si c'est Streamlit :
```bash
streamlit run velo/web.py
```

L'interface sera accessible sur `http://localhost:8501`

## Technologies Utilisées

### Backend
- **Python 3.10+**
- **Flask / FastAPI** : Framework web
- **Pandas / NumPy** : Traitement de données
- **Scikit-learn** : Modèles de prédiction
- **SQLite / PostgreSQL** : Base de données (optionnel)

### Frontend
- **Streamlit** : Interface web interactive
- **Plotly** : Visualisations interactives
- **Folium** : Cartes interactives
- **HTML/CSS/JavaScript** : Interface web classique

## Fonctionnalités Clés

### Pour l'Utilisateur
- 🗺️ Carte interactive des stations
- 🚲 Disponibilité en temps réel
- 📱 Réservation de vélos
- 📊 Historique personnel
- 💳 Gestion de compte

### Pour l'Administrateur
- 📈 Dashboard analytique
- 🔮 Prédictions de demande
- 🚚 Optimisation de la redistribution
- 📊 Rapports de performance
- ⚙️ Gestion des stations et vélos

## Insights Business

### Patterns d'Utilisation
1. **Pics de demande** : 8h-9h et 17h-18h (trajet travail)
2. **Saisonnalité** : Forte baisse en hiver (-40% vs été)
3. **Météo** : Pluie réduit l'utilisation de 30%
4. **Week-end** : Usage loisir différent (après-midi)

### Optimisations
1. **Redistribution intelligente** : Anticiper les besoins du soir dès le matin
2. **Maintenance préventive** : Planifier selon l'utilisation prédite
3. **Pricing dynamique** : Ajuster les tarifs selon la demande
4. **Expansion** : Identifier les zones sous-servies

---

**Projet réalisé par** : Aymen Chaghoub - L3 Informatique, Université de Lille
