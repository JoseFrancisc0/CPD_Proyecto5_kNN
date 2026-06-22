from mpi4py import MPI
from collections import Counter
import numpy as np
import os
import csv

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
    print("Subset % | N_Train | N_Test | T_Comp(s) | T_Comm(s) | T_Reduce(s) | T_Total(s) | GFLOP/s | Accuracy")
    print("-" * 86)

for frac in data_fractions:
    chunks_X_train, chunks_y_train, X_test, k = None, None, None, None
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

        k = 3
        X_test = X_test_full
        chunks_X_train = np.array_split(X_train_full, size)
        chunks_y_train = np.array_split(y_full, size)
        y_test_global = y_test_full

    comm.barrier()
    start_total = MPI.Wtime()

    # 1 y 2. Broadcast y Scatter
    start_comm = MPI.Wtime()
    X_test = comm.bcast(X_test, root=0)
    k = comm.bcast(k, root=0)
    local_X_train = comm.scatter(chunks_X_train, root=0)
    local_y_train = comm.scatter(chunks_y_train, root=0)
    time_comm = MPI.Wtime() - start_comm

    # 3. Computo local
    start_comp = MPI.Wtime()
    n_test = len(X_test)

    local_top_k_dists = np.zeros((n_test, k), dtype=float)
    local_top_k_labels = np.zeros((n_test, k), dtype=int)

    for i, x in enumerate(X_test):
        if rank == 0 and i % 1000 == 0 and i > 0:
            print(f"   -> [Progreso Global] Calculando distancias: {i}/{n_test} imágenes de prueba...")
        dists = np.linalg.norm(local_X_train - x, axis=1)
        k_indices = np.argsort(dists)[:k]
        local_top_k_dists[i] = dists[k_indices]
        local_top_k_labels[i] = local_y_train[k_indices]

    time_comp = MPI.Wtime() - start_comp

    # FLOPs locales
    flops_per_dist = 2352
    local_flops = len(X_test) * len(local_X_train) * flops_per_dist

    # 4. Gather
    start_gather = MPI.Wtime()
    gather_dists = comm.gather(local_top_k_dists, root=0)
    gather_labels = comm.gather(local_top_k_labels, root=0)
    time_comm += (MPI.Wtime() - start_gather)

    # 5. Reduccion y validacion global
    if rank == 0:
        start_reduce = MPI.Wtime()
        final_y_preds = np.zeros(n_test, dtype=int)
        
        for i in range(n_test):
            all_dists_i = np.concatenate([gather_dists[p][i] for p in range(size)])
            all_labels_i = np.concatenate([gather_labels[p][i] for p in range(size)])
            global_k_indices = np.argsort(all_dists_i)[:k]
            global_k_labels = all_labels_i[global_k_indices]
            most_common = Counter(global_k_labels).most_common(1)
            final_y_preds[i] = most_common[0][0]
        
        time_reduce = MPI.Wtime() - start_reduce

        time_total = MPI.Wtime() - start_total
        accuracy = np.mean(final_y_preds == y_test_global.astype(int))

        gflops_local = 0
        if time_comp > 0:
            gflops_local = (local_flops) / (time_comp * 1e9)

        n_train_real = sum(len(c) for c in chunks_X_train)

        print(f"{int(frac*100):6} % | {n_train_real:7} | {len(y_test_global):6} | {time_comp:9.4f} | {time_comm:9.4f} | {time_reduce:9.4f} | {time_total:10.4f} | {gflops_local:7.4f} | {accuracy:.4f}")

        results = {
            'Processes': size,
            'Subset_%': int(frac*100),
            'N_train': n_train_real,
            'N_test': len(y_test_global),
            'T_comp (s)': time_comp,
            'T_comm (s)': time_comm,
            'T_reduce (s)': time_reduce,
            'T_total (s)': time_total,
            'GFLOP/s': gflops_local,
            'Accuracy': accuracy
        }

        filename = "results/knn_cluster_results_v2.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        write_header = not os.path.exists(filename)

        with open(filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(results)

    comm.Barrier()