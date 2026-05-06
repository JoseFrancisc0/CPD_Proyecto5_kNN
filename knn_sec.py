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

# Cargar y dividir los datos
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Parámetro
k = 3

# Medir tiempo de ejecución
start_time = time.time()

# Realizar predicciones
y_pred = [knn_predict(x, X_train, y_train, k) for x in X_test]

# Evaluar
accuracy = np.mean(y_pred == y_test)
end_time = time.time()
t_total = end_time - start_time

# FLOPS
flops_per_dist = 192
flops = len(X_test) * len(X_train) * flops_per_dist
gflops = flops / ((end_time - start_time) * 1e9)

print(f"Accuracy: {accuracy:.4f}")
print(f"Execution time (sequential): {t_total:.4f} sec")
print(f"GFLOPs: {gflops:.4f} GFLOPs/s")

results_df = pd.DataFrame([{
    'Processes': 1,
    'Subset_%': 100,
    'N_train': len(X_train),
    'N_test': len(X_test),
    'T_comp (s)': t_total,
    'T_comm (s)': 0.0,
    'T_total (s)': t_total,
    'GFLOP/s': gflops,
    'Accuracy': accuracy
}])

filename = "knn_results.csv"
write_header = not os.path.exists(filename)
results_df.to_csv(filename, mode='a', header=write_header, index=False)