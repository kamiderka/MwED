import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from time import time
from joblib import Parallel, delayed
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.random_forest as md

# Przygotowanie danych z mniejszą próbką
X, y = make_classification(n_samples=500, n_features=20, random_state=42)

# Parametry do testowania (zmniejszona liczba kombinacji)
n_estimators_list = [3, 10]  # Mniejsza liczba drzew
max_depth_list = [5, 7]  # Wybrane ograniczone głębokości
max_features_list = ['sqrt', 'log2']  # Dwa najczęściej używane parametry

# Funkcja do przeprowadzenia pojedynczego testu dla danego zestawu parametrów
def single_test(X, y, n_estimators, max_depth, max_features):
    params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features}

    # Testowanie Twojego modelu
    my_model = md.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    start_time = time()
    my_model.fit(X, y)
    my_time = time() - start_time
    my_pred = my_model.predict(X)
    my_accuracy = accuracy_score(y, my_pred)

    # Testowanie modelu sklearn
    sklearn_model = SklearnRandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    start_time = time()
    sklearn_model.fit(X, y)
    sklearn_time = time() - start_time
    sklearn_pred = sklearn_model.predict(X)
    sklearn_accuracy = accuracy_score(y, sklearn_pred)

    return {
        'params': params,
        'my_model': (my_accuracy, my_time),
        'sklearn_model': (sklearn_accuracy, sklearn_time)
    }

# Uruchomienie testów równolegle
results = Parallel(n_jobs=-1)(delayed(single_test)(X, y, n_estimators, max_depth, max_features)
                              for n_estimators in n_estimators_list
                              for max_depth in max_depth_list
                              for max_features in max_features_list)

# Formatowanie wyników do dalszej analizy
formatted_results = {'my_model': [], 'sklearn_model': []}
for result in results:
    formatted_results['my_model'].append((result['params'], result['my_model'][0], result['my_model'][1]))
    formatted_results['sklearn_model'].append((result['params'], result['sklearn_model'][0], result['sklearn_model'][1]))

def plot_results(results):
    fig, axs = plt.subplots(2, len(n_estimators_list), figsize=(18, 10))

    for idx, n_estimators in enumerate(n_estimators_list):
        my_results = [result for result in results['my_model'] if result[0]['n_estimators'] == n_estimators]
        sklearn_results = [result for result in results['sklearn_model'] if result[0]['n_estimators'] == n_estimators]

    # Grupujemy wyniki według `max_features`
    for feature_idx, max_features in enumerate(max_features_list):
        # Filtrujemy wyniki tylko dla danego max_features
        my_results_filtered = [result for result in my_results if result[0]['max_features'] == max_features]
        sklearn_results_filtered = [result for result in sklearn_results if result[0]['max_features'] == max_features]

        # Pobieramy dokładności i czasy wykonania dla Twojego modelu
        my_accuracies = [result[1] for result in my_results_filtered]
        my_times = [result[2] for result in my_results_filtered]

        # Pobieramy dokładności i czasy wykonania dla modelu sklearn
        sklearn_accuracies = [result[1] for result in sklearn_results_filtered]
        sklearn_times = [result[2] for result in sklearn_results_filtered]

        # Wykres dokładności
        axs[0, idx].plot(
            max_depth_list, my_accuracies, label=f'My RandomForest - max_features={max_features}', marker='o'
        )
        axs[0, idx].plot(
            max_depth_list, sklearn_accuracies, label=f'Sklearn RandomForest - max_features={max_features}', marker='x'
        )
        axs[0, idx].set_title(f'n_estimators = {n_estimators} - Accuracy')
        axs[0, idx].set_xlabel('max_depth')
        axs[0, idx].set_ylabel('Accuracy')
        axs[0, idx].legend()

        # Wykres czasu
        axs[1, idx].plot(
            max_depth_list, my_times, label=f'My RandomForest - max_features={max_features}', marker='o'
        )
        axs[1, idx].plot(
            max_depth_list, sklearn_times, label=f'Sklearn RandomForest - max_features={max_features}', marker='x'
        )
        axs[1, idx].set_title(f'n_estimators = {n_estimators} - Time')
        axs[1, idx].set_xlabel('max_depth')
        axs[1, idx].set_ylabel('Time (s)')
        axs[1, idx].legend()

    plt.tight_layout()
    plt.show()

# Rysujemy wykresy
plot_results(formatted_results)
