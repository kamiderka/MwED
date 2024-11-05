import pandas as pd
import numpy as np
import random
from decision_tree import DecisionTree

LICZBA_DRZEW = 10
LICZBA_STRAPOW = LICZBA_DRZEW
LICZBA_RZEDOW_NA_STRP = 10
las = []  # Lista drzew

ETYKIETA_KLASYFIKATORA = "variety"

dane = pd.read_csv("iris.csv")

# Podział na zbiór treningowy 90% i testowy 10%
dane_sampled = dane.sample(frac=0.9, random_state=1)
dane_test = dane.drop(dane_sampled.index.tolist())
dane = dane_sampled

# Zamiana kolumn na string
for col in dane.select_dtypes(include='object').columns:
    dane[col] = dane[col].astype("string")

X = dane.drop(ETYKIETA_KLASYFIKATORA, axis=1).values
y = dane[ETYKIETA_KLASYFIKATORA].astype("category").cat.codes.values

X_test = dane_test.drop(ETYKIETA_KLASYFIKATORA, axis=1).values
y_test = dane_test[ETYKIETA_KLASYFIKATORA].astype("category").cat.codes.values

# Budowa drzew
for i in range(LICZBA_STRAPOW):
    # Bootstrap
    indices = np.random.choice(len(X), size=LICZBA_RZEDOW_NA_STRP, replace=True)
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]

    # Trenowanie drzewa na samplach
    tree = DecisionTree(max_depth=5)
    tree.fit(X_bootstrap, y_bootstrap)

    # Dodaj drzewo do lasu
    las.append(tree)

# Predykcja poprzez głosowanie większościowe
predictions = []
for sample in X_test:
    tree_predictions = [tree._predict_sample(sample, tree.tree) for tree in las]
    final_prediction = np.bincount(tree_predictions).argmax()  #Głosowanie większościowe
    predictions.append(final_prediction)

# Sprawdzenie dokładności
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
