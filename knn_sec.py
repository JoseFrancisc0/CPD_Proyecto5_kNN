from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import pandas as pd
import time
import os

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

# Cargar dataset
digits = load_digits()
k = 3

# Subsets de N
data_fractions = [0.25, 0.5, 0.75, 1.0]
all_results = []

print("=== PRUEBAS SECUENCIALES (p=1) ===")
print("Subset % | N_train | N_test | T_total (s) | GFLOP/s | Accuracy")
print("-" * 65)

for frac in data_fractions:
    n_samples = int(len(digits.data) * frac)
    X_subset = digits.data[:n_samples]
    y_subset = digits.target[:n_samples]


    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, test_size=0.2, random_state=42
    )

    # Medir tiempo de ejecución
    start_time = time.time()
    y_pred = [knn_predict(x, X_train, y_train, k) for x in X_test]
    end_time = time.time()

    # Evaluar
    accuracy = np.mean(y_pred == y_test)
    t_total = end_time - start_time

    # FLOPS
    flops_per_dist = 192
    flops = len(X_test) * len(X_train) * flops_per_dist
    gflops = flops / ((t_total) * 1e9)

    print(f"{int(frac*100):6} % | {len(X_train):7} | {len(X_test):6} | {t_total:11.4f} | {gflops:7.4f} | {accuracy:.4f}")

    all_results.append({
        'Processes': 1,
        'Subset_%': 100,
        'N_train': len(X_train),
        'N_test': len(X_test),
        'T_comp (s)': t_total,
        'T_comm (s)': 0.0,
        'T_total (s)': t_total,
        'GFLOP/s': gflops,
        'Accuracy': accuracy
    })

filename = "results/knn_digits_results.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
results_df = pd.DataFrame(all_results)
results_df.to_csv(filename, mode='w', header=True, index=False)