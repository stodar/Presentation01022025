# Guide détaillé : Prédiction des prix immobiliers en Californie

## 1. Chargement et exploration des données

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Chargement des données
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
```

### Explication des features :
- MedInc : revenu médian du bloc
- HouseAge : âge médian des maisons dans le bloc
- AveRooms : nombre moyen de pièces
- AveBedrms : nombre moyen de chambres
- Population : population du bloc
- AveOccup : nombre moyen d'occupants
- Latitude : latitude du bloc
- Longitude : longitude du bloc

La variable cible (y) représente la valeur médiane des maisons en centaines de milliers de dollars.

### Analyse exploratoire :
```python
# Statistiques descriptives
print("Description des features :")
print(X.describe())

# Distribution des prix
plt.figure(figsize=(10, 6))
plt.hist(y, bins=50)
plt.title('Distribution des prix des maisons')
plt.xlabel('Prix (100k$)')
plt.ylabel('Fréquence')
plt.show()

# Corrélations
correlation_matrix = pd.concat([X, pd.Series(y, name='Price')], axis=1).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.show()
```

## 2. Prétraitement des données

### Split train/test :
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20% pour le test
    random_state=42   # Pour la reproductibilité
)
```

Pourquoi faire un split ?
- Éviter le surapprentissage
- Évaluer les performances réelles du modèle
- Simuler des données "inconnues"

### Standardisation :
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Pourquoi standardiser ?
- Mettre toutes les features à la même échelle
- Améliorer la convergence des algorithmes
- Éviter qu'une feature domine les autres
- IMPORTANT : On fit le scaler uniquement sur les données d'entraînement !

## 3. Modélisation

### Régression linéaire :
```python
from sklearn.linear_model import LinearRegression

# Création et entraînement
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Prédictions
lr_pred = lr_model.predict(X_test_scaled)
```

Caractéristi