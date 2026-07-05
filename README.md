# Proyecto 5: Paralelización del k-Nearest Neighbors

Este repositorio contiene los archivos de la implementación del algoritmo KNN paralelizado para el proyecto del curso de Computación Paralela y Distribuida.

**Integrantes:**
- Isaac Emanuel Javier Simeón Sarmiento
- Jose Francisco Wong Orrillo

**Universidad**: Universidad de Ingeniería y Tecnología (UTEC)

**Ciclo:** 2026-1

### Ambiente de ejecución paralelizable

Las pruebas del algoritmo se ejecutaron en un nodo del clúster computacional de la universidad, *Khipu*, gestionado mediante el planificador de tareas SLURM. El hardware y software configurado para el experimento cuenta con las siguientes características.

- **Procesador:** Intel Xeon Gold 6226R a 2.90GHz.
- **Núcleos:** 16 núcleos físicos y 32 hilos lógicos (Hyper-Threading habilitado)
- **Memoria RAM:** 98 GB disponibles
- **Entorno Virtual:** Hypervisor VMware
- **Software:** Python 3 con `mpi4py` soportada sobre OpenMPI (v4.1.6)

### Archivos

- `knn_sec_cluster.py`: la implementación del kNN secuencial, preparado para las pruebas de ejecución en el clúster.
- `knn_par_cluster.py`: la implementación del kNN paralelo, preparado para las pruebas de ejecución en el clúster.
- `knn_job.sh`: el job para las pruebas experimentales dentro del clúster.
- `download_mnist.py`: para descargar el dataset MNIST_784 (almacenado en `mnist_784.npz`).
- `analisis_cluster.ipynb`: el notebook utilizado para la sección de análisis de resultados experimentales.
- `results/knn_cluster_results_v2.csv`: los resultados experimentales del kNN ejecutado sobre el clúster.
- `imgs/`: los gráficos de análisis de resultados experimentales.


