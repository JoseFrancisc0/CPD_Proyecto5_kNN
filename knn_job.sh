#!/bin/bash
#SBATCH --job-name=knn_mnist
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=90G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

# 1. Limpiar y cargar los módulos necesarios
module purge
module load miniconda/3.0
module load gnu12/12.4.0
module load openmpi4/4.1.6

conda activate knn_env

# 2. Crear la carpeta de resultados si no existe
mkdir -p results

# 3. Ejecución Secuencial (p=1)
echo "=== INICIANDO P=1 (SECUENCIAL) ==="
python3 -u knn_sec_cluster.py

# 4. Ejecución Paralela
echo "=== INICIANDO MPI (PARALELO) ==="
for p in 2 4 8 16 32; do
    echo "Lanzando con $p procesos..."
    mpiexec -n $p python3 -u knn_par_cluster.py
done

echo "=== TRABAJO FINALIZADO ==="

# 5. Limpieza
conda deactivate

module unload openmpi4/4.1.6
module unload gnu12/12.4.0
module unload miniconda/3.0