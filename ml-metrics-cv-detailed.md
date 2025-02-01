# Métriques d'évaluation et Validation Croisée en Machine Learning

## 1. Métriques d'évaluation

### 1.1 La Matrice de Confusion

La base pour comprendre toutes les métriques :

```
                     Prédictions
                 Positif    Négatif
Réalité  Positif    VP        FN
         Négatif    FP        VN
```

- VP (Vrais Positifs) : Correctement prédits comme positifs
- VN (Vrais Négatifs) : Correctement prédits comme négatifs
- FP (Faux Positifs) : Incorrectement prédits comme positifs
- FN (Faux Négatifs) : Incorrectement prédits comme négatifs

Code pour afficher la matrice de confusion :
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Prédictions')
plt.ylabel('Réalité')
plt.show()
```

### 1.2 Précision (Precision)

```python
Précision = VP / (VP + FP)
```

- Mesure la qualité des prédictions positives
- Répond à la question : "Parmi les cas que j'ai prédits positifs, combien le sont réellement ?"
- Important quand le coût des faux positifs est élevé

Exemple :
- Détection de fraude : Une précision élevée signifie que quand on prédit une fraude, on a rarement tort
- Code :
```python
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average='weighted')
```

### 1.3 Rappel (Recall)

```python
Rappel = VP / (VP + FN)
```

- Mesure la capacité à trouver tous les cas positifs
- Répond à la question : "Parmi tous les cas positifs réels, combien ai-je réussi à en identifier ?"
- Important quand manquer un cas positif est coûteux

Exemple :
- Diagnostic médical : Un rappel élevé signifie qu'on manque rarement une maladie
- Code :
```python
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred, average='weighted')
```

### 1.4 F1-Score

```python
F1 = 2 * (Précision * Rappel) / (Précision + Rappel)
```

- Moyenne harmonique de la précision et du rappel
- Équilibre entre précision et rappel
- Varie entre 0 (pire) et 1 (meilleur)
- Particulièrement utile pour les datasets déséquilibrés

Code :
```python
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='weighted')
```

### 1.5 Rapport de classification complet

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## 2. Validation Croisée

### 2.1 Principe de base

La validation croisée divise les données en K sous-ensembles (folds) :
- K-1 folds pour l'entraînement
- 1 fold pour la validation
- Répété K fois en changeant le fold de validation

### 2.2 Types de validation croisée

#### K-Fold classique
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
```

#### Validation croisée stratifiée
Maintient la proportion des classes dans chaque fold :
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 2.3 Utilisation avec scikit-learn

#### Méthode simple
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Scores: {scores}")
print(f"Moyenne: {scores.mean():.3f} (±{scores.std() * 2:.3f})")
```

#### Validation croisée avec plusieurs métriques
```python
from sklearn.model_selection import cross_validate

scoring = {'precision': 'precision_weighted',
           'recall': 'recall_weighted',
           'f1': 'f1_weighted'}

scores = cross_validate(model, X, y, scoring=scoring, cv=5)
```

### 2.4 Bonnes pratiques

1. Choix du nombre de folds :
   - K=5 ou K=10 sont des valeurs courantes
   - Plus K est grand, plus le temps de calcul augmente
   - Plus K est petit, plus la variance des estimations augmente

2. Stratification :
   - Toujours utiliser StratifiedKFold pour la classification
   - Assure une distribution équilibrée des classes

3. Shuffle :
   - Toujours mélanger les données (shuffle=True)
   - Fixez random_state pour la reproductibilité

### 2.5 Example complet

```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

# Définition du modèle
model = RandomForestClassifier(random_state=42)

# Définition des métriques
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted'
}

# Validation croisée
cv_results = cross_validate(
    model, X, y,
    scoring=scoring,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    return_train_score=True
)

# Affichage des résultats
for metric in scoring.keys():
    train_scores = cv_results[f'train_{metric}']
    test_scores = cv_results[f'test_{metric}']
    print(f"\n{metric.capitalize()}:")
    print(f"Train: {train_scores.mean():.3f} (±{train_scores.std() * 2:.3f})")
    print(f"Test: {test_scores.mean():.3f} (±{test_scores.std() * 2:.3f})")
```

## 3. Points clés à retenir

1. Métriques :
   - La précision mesure la qualité des prédictions positives
   - Le rappel mesure la capacité à trouver tous les cas positifs
   - Le F1-score est un compromis entre précision et rappel

2. Validation croisée :
   - Évaluation plus robuste que simple train/test split
   - Permet d'estimer la variance du modèle
   - Aide à détecter le surapprentissage
   - Stratification importante pour les données déséquilibrées

3. Choix des métriques :
   - Dépend du contexte métier
   - Considérer le coût des erreurs
   - Utiliser plusieurs métriques complémentaires
