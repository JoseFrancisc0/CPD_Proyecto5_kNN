from collections import Counter
import numpy as np
import time
import os
import csv

def knn_predict(test_point, X_train, y_train, k):
    distances = np.linalg.norm(X_train - test_point, axis=1)
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

# Cargar dataset
mnist = np.load('mnist_784.npz', allow_pickle=True)
X_full = mnist['data']
Y_full = mnist['target']
k = 3

# Subsets de N
data_fractions = [0.25, 0.5, 0.75, 1.0]
all_results = []

print("=== PRUEBAS SECUENCIALES (p=1) ===")
print("Subset % | N_train | N_test | T_total (s) | GFLOP/s | Accuracy")
print("-" * 65)

for frac in data_fractions:
    n_samples = int(len(X_full) * frac)
    X_subset = X_full[:n_samples]
    y_subset = Y_full[:n_samples]

    # Generando splits
    np.random.seed(42)
    indices = np.random.permutation(len(X_subset))
    split_idx = int(len(X_subset) * 0.8)

    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X_subset[train_idx], X_subset[test_idx]
    y_train, y_test = y_subset[train_idx], y_subset[test_idx]

    # Medir tiempo de ejecución
    start_time = time.time()
    y_pred = []
    for i, x in enumerate(X_test):
        if i % 1000 == 0 and i > 0:
            print(f"   -> Procesadas {i}/{len(X_test)} imágenes de prueba...")
        y_pred.append(knn_predict(x, X_train, y_train, k))
    end_time = time.time()

    # Evaluar
    accuracy = np.mean(y_pred == y_test)
    t_total = end_time - start_time

    # FLOPS
    flops_per_dist = 2352
    flops = len(X_test) * len(X_train) * flops_per_dist
    gflops = flops / ((t_total) * 1e9)

    print(f"{int(frac*100):6} % | {len(X_train):7} | {len(X_test):6} | {t_total:11.4f} | {gflops:7.4f} | {accuracy:.4f}")

    all_results.append({
        'Processes': 1,
        'Subset_%': int(frac*100),
        'N_train': len(X_train),
        'N_test': len(X_test),
        'T_comp (s)': t_total,
        'T_comm (s)': 0.0,
        'T_total (s)': t_total,
        'GFLOP/s': gflops,
        'Accuracy': accuracy
    })

filename = "results/knn_cluster_results.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)

with open(filename, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)