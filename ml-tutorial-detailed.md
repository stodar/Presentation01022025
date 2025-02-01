# Introduction au Machine Learning avec Python - Guide détaillé
## Un tutoriel pratique avec scikit-learn

## Table des matières
1. [Installation et préparation](#1-installation-et-préparation)
2. [Premier exemple : Dataset Iris](#2-premier-exemple--dataset-iris)
3. [Préparation des données](#3-préparation-des-données)
4. [Modélisation avec Random Forest](#4-modélisation-avec-random-forest)
5. [Exercice principal : California Housing](#5-exercice-principal--california-housing)
6. [Comparaison de modèles](#6-comparaison-de-modèles)
7. [Analyse des features](#7-analyse-des-features)
8. [Validation croisée](#8-validation-croisée)
9. [Exercices proposés](#9-exercices-proposés)

## 1. Installation et préparation

### Installation des packages nécessaires
```python
!pip install numpy pandas matplotlib seaborn scikit-learn
```

### Imports Python
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
```

**Rôle de chaque package** :
- `numpy` : Calculs numériques et manipulations de tableaux
- `pandas` : Manipulation et analyse de données
- `matplotlib` et `seaborn` : Visualisation de données
- `scikit-learn` : Algorithmes de machine learning

## 2. Premier exemple : Dataset Iris

### Chargement des données
```python
iris = load_iris()
X = iris.data
y = iris.target

# Conversion en DataFrame pour une meilleure visualisation
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['target'] = y
```

**Pourquoi le dataset Iris ?**
- Dataset classique et simple
- 4 caractéristiques (longueur et largeur des sépales et pétales)
- 3 classes de fleurs à prédire
- Données déjà nettoyées et équilibrées

### Visualisation des données
```python
# Création d'un pair plot
sns.pairplot(iris_df, hue='target', diag_kind='hist')
plt.show()
```

**Intérêt de la visualisation** :
- Voir les relations entre variables
- Observer la séparation des classes
- Identifier les patterns potentiels

## 3. Préparation des données

### Split Train/Test
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20% pour le test
    random_state=42   # Pour la reproductibilité
)
```

**Pourquoi splitter les données ?**
- Évaluer la performance sur des données non vues
- Éviter le surapprentissage
- Simuler un cas réel d'utilisation

### Standardisation
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Importance de la standardisation** :
- Met toutes les variables à la même échelle
- Améliore la convergence des algorithmes
- Nécessaire pour certains algorithmes (ex: SVM)

## 4. Modélisation avec Random Forest

### Création et entraînement du modèle
```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
```

### Évaluation
```python
y_pred = rf_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

**Métriques importantes** :
- Precision : Exactitude des prédictions positives
- Recall : Capacité à trouver tous les cas positifs
- F1-score : Moyenne harmonique precision/recall

## 5. Exercice principal : California Housing

### Chargement des données
```python
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
```

**Caractéristiques du dataset** :
- Problème de régression (prédiction de prix)
- 8 variables explicatives
- Plus complexe que Iris

### Analyse exploratoire
```python
plt.figure(figsize=(10, 6))
plt.hist(y, bins=50)
plt.title('Distribution des prix des maisons')
plt.show()
```

## 6. Comparaison de modèles

### Régression linéaire
```python
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
```

### Random Forest
```python
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
```

**Pourquoi comparer les modèles ?**
- Établir une baseline
- Comprendre le compromis complexité/performance
- Choisir le meilleur modèle pour le problème

## 7. Analyse des features

### Importance des features
```python
feature_importance = pd.DataFrame({
    'feature': housing.feature_names,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
```

**Utilité de l'analyse** :
- Identifier les variables importantes
- Simplifier potentiellement le modèle
- Guider la collecte future de données

## 8. Validation croisée

```python
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"Scores CV: {cv_scores}")
print(f"Moyenne: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

**Avantages de la validation croisée** :
- Évaluation plus robuste
- Détection du surapprentissage
- Estimation de la variabilité du modèle

## 9. Exercices proposés

1. **Optimisation des hyperparamètres**
   - Utiliser GridSearchCV
   - Tester différentes configurations
   - Évaluer l'impact sur les performances

2. **Feature Engineering**
   - Créer de nouvelles variables
   - Transformer les variables existantes
   - Analyser l'impact sur le modèle

3. **Test d'autres algorithmes**
   - SVM
   - XGBoost
   - Réseaux de neurones

4. **Analyse approfondie des erreurs**
   - Identifier les cas problématiques
   - Comprendre les limitations du modèle
   - Proposer des améliorations

## Ressources supplémentaires

- Documentation scikit-learn : [https://scikit-learn.org/](https://scikit-learn.org/)
- Cours en ligne : Coursera, edX
- Plateformes d'exercices : Kaggle
- Forums : Stack Overflow, Reddit r/MachineLearning

## Notes importantes

- Toujours explorer les données avant de modéliser
- Valider les hypothèses des modèles
- Tester différentes approches
- Documenter les choix et résultats
- Interpréter les résultats avec précaution
