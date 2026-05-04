from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    distances = np.zeros(len(X_train))
    for i, x in enumerate(X_train):
        distances[i] = euclidean_distance(test_point, x)
    
    k_indices = np.argsort(distances)[:k]

    k_labels = np.zeros(len(k_indices), dtype=int)
    for idx, i in enumerate(k_indices):
        k_labels[idx] = y_train[i]

    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

# Configuracion MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
X_train, y_train, k, chunks_X_test = None, None, None, None

if rank == 0:
    digits = load_digits()
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )

    X_train = X_train_full
    y_train = y_train_full
    k = 3
    chunks_X_test = np.array_split(X_test_full, size)
    y_test_global = y_test_full

comm.barrier()
start_total = MPI.Wtime()

# Broadcast y Scatter
start_comm = MPI.Wtime()

X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
k = comm.bcast(k, root=0)
local_X_test = comm.scatter(chunks_X_test, root=0)

end_comm_initial = MPI.Wtime()
time_comm = end_comm_initial - start_comm

# Computo local
start_comp = MPI.Wtime()

local_y_pred = np.zeros(len(local_X_test), dtype=int)
for i, x in enumerate(local_X_test):
    local_y_pred[i] = knn_predict(x, X_train, y_train, k)

end_comp = MPI.Wtime()
time_comp = end_comp - start_comp

# FLOPs locales
flops_per_dist = 192
n_tr = len(X_train)
n_te_local = len(local_X_test)
local_flops = n_te_local * n_tr * flops_per_dist

# Gather
start_gather = MPI.Wtime()
gather_preds = comm.gather(local_y_pred, root=0)
end_gather = MPI.Wtime()
time_comm += (end_gather - start_gather)

if rank == 0:
    final_y_preds = np.concatenate(gather_preds)
    end_total = MPI.Wtime()

    accuracy = np.mean(final_y_preds == y_test_global)
    time_total = end_total - start_total

    print("\n--- kNN Paralelo ---")
    print(f"Procesos (p): {size}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tiempo Total: {time_total:.4f} seg")
    print(f"Tiempo Computo (Maestro): {time_comp:.4f} seg")
    print(f"Tiempo Comunicacion (Maestro): {time_comm:.4f} seg")

    if time_comp > 0:
        gflops_local = (local_flops) / (time_comp * 1e9)
        print(f"Rendimiento de Computo (Maestro): {gflops_local:.4f} GFLOP/s")