from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import pandas as pd
import os

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
digits = None
if rank == 0:
    digits = load_digits()

data_fractions = [0.25, 0.5, 0.75, 1.0]
if rank == 0:
    print(f"=== PRUEBAS DE ESCALABILIDAD CON {size} PROCESOS ===")
    print("Subset % | N_Train | N_Test | T_Comp(s) | T_Comm(s) | T_Total(s) | GFLOP/s | Accuracy")
    print("-" * 86)

for frac in data_fractions:
    X_train, y_train, k, chunks_X_test = None, None, None, None
    y_test_global = None

    if rank == 0:
        n_samples = int(len(digits.data) * frac)
        X_subset = digits.data[:n_samples]
        y_subset = digits.target[:n_samples]

        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
                X_subset, y_subset, test_size=0.2, random_state=42
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
    time_comm = MPI.Wtime() - start_comm

    # Computo local
    start_comp = MPI.Wtime()
    local_y_pred = np.zeros(len(local_X_test), dtype=int)
    for i, x in enumerate(local_X_test):
        local_y_pred[i] = knn_predict(x, X_train, y_train, k)
    time_comp = MPI.Wtime() - start_comp

    # FLOPs locales
    flops_per_dist = 192
    local_flops = len(local_X_test) * len(X_train) * flops_per_dist

    # Gather
    start_gather = MPI.Wtime()
    gather_preds = comm.gather(local_y_pred, root=0)
    time_comm += (MPI.Wtime() - start_gather)

    if rank == 0:
        final_y_preds = np.concatenate(gather_preds)
        time_total = MPI.Wtime() - start_total
        accuracy = np.mean(final_y_preds == y_test_global)
        
        gflops_local = 0
        if time_comp > 0:
            gflops_local = (local_flops) / (time_comp * 1e9)

        print(f"{int(frac*100):6} % | {len(X_train):7} | {len(y_test_global):6} | {time_comp:9.4f} | {time_comm:9.4f} | {time_total:10.4f} | {gflops_local:7.4f} | {accuracy:.4f}")

        results_df = pd.DataFrame([{
            'Processes': size,
            'Subset_%': int(frac*100),
            'N_train': len(X_train),
            'N_test': len(y_test_global),
            'T_comp (s)': time_comp,
            'T_comm (s)': time_comm,
            'T_total (s)': time_total,
            'GFLOP/s': gflops_local,
            'Accuracy': accuracy
        }])

        filename = f"knn_results_{size}.csv"
        write_header = not os.path.exists(filename)
        results_df.to_csv(filename, mode='a', header=write_header, index=False)

    comm.Barrier()