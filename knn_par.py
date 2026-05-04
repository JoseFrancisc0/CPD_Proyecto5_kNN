from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time

# -- 1. Funciones --
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

# -- 2. Configuracion del MPI --
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

X_train = None
y_train = None
k = None
chunks_X_test = None

# -- 3. Fase rank 0 --
if rank == 0:
    print(f"Iniciando kNN paralelo con {size} procesos...")
    digits = load_digits()
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )

    X_train = X_train_full
    y_train = y_train_full
    k = 3
    chunks_X_test = np.array_split(X_test_full, size)
    y_test_global = y_test_full
    start_time = time.time()

# -- 4. Broadcast y Scatter --
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
k = comm.bcast(k, root=0)
local_X_test = comm.scatter(chunks_X_test, root=0)

# -- 5. Computo local -- 
local_y_pred = np.zeros(len(local_X_test), dtype=int)
for i, x in enumerate(local_X_test):
    local_y_pred[i] = knn_predict(x, X_train, y_train, k)

# -- 6. Gather --
gather_preds = comm.gather(local_y_pred, root=0)

# -- 7. Evaluacion --
if rank == 0:
    final_y_preds = np.concatenate(gather_preds)
    end_time = time.time()
    accuracy = np.mean(final_y_preds == y_test_global)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Execution time (paralelo): {end_time - start_time:.4f} sec")