from mpi4py import MPI
from collections import Counter
import numpy as np
import os
import csv

def knn_predict(test_point, X_train, y_train, k):
    distances = np.linalg.norm(X_train - test_point, axis=1)
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

# Configuracion MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
mnist = None

if rank == 0:
    mnist = np.load('mnist_784.npz', allow_pickle=True)
    X_full = mnist['data']
    y_full = mnist['target']

data_fractions = [0.25, 0.5, 0.75, 1.0]
if rank == 0:
    print(f"=== PRUEBAS DE ESCALABILIDAD CON {size} PROCESOS ===")
    print("Subset % | N_Train | N_Test | T_Comp(s) | T_Comm(s) | T_Total(s) | GFLOP/s | Accuracy")
    print("-" * 86)

for frac in data_fractions:
    X_train, y_train, k, chunks_X_test = None, None, None, None
    y_test_global = None

    if rank == 0:
        n_samples = int(len(X_full) * frac)
        X_subset = X_full[:n_samples]
        y_subset = y_full[:n_samples]

        # Generando splits
        np.random.seed(42)
        indices = np.random.permutation(len(X_subset))
        split_idx = int(len(X_subset) * 0.8)
        
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        X_train_full, X_test_full = X_subset[train_idx], X_subset[test_idx]
        y_train_full, y_test_full = y_subset[train_idx], y_subset[test_idx]

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
        if rank == 0 and i % 100 == 0 and i > 0:
            print(f"   -> [Rank 0] Procesadas {i}/{len(local_X_test)} imágenes de su bloque...")
        local_y_pred[i] = int(knn_predict(x, X_train, y_train, k))
    time_comp = MPI.Wtime() - start_comp

    # FLOPs locales
    flops_per_dist = 2352
    local_flops = len(local_X_test) * len(X_train) * flops_per_dist

    # Gather
    start_gather = MPI.Wtime()
    gather_preds = comm.gather(local_y_pred, root=0)
    time_comm += (MPI.Wtime() - start_gather)

    if rank == 0:
        final_y_preds = np.concatenate(gather_preds)
        time_total = MPI.Wtime() - start_total
        accuracy = np.mean(final_y_preds == y_test_global.astype(int))
        
        gflops_local = 0
        if time_comp > 0:
            gflops_local = (local_flops) / (time_comp * 1e9)

        print(f"{int(frac*100):6} % | {len(X_train):7} | {len(y_test_global):6} | {time_comp:9.4f} | {time_comm:9.4f} | {time_total:10.4f} | {gflops_local:7.4f} | {accuracy:.4f}")

        results = {
            'Processes': size,
            'Subset_%': int(frac*100),
            'N_train': len(X_train),
            'N_test': len(y_test_global),
            'T_comp (s)': time_comp,
            'T_comm (s)': time_comm,
            'T_total (s)': time_total,
            'GFLOP/s': gflops_local,
            'Accuracy': accuracy
        }

        filename = "results/knn_cluster_results.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        write_header = not os.path.exists(filename)

        with open(filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(results)

    comm.Barrier()