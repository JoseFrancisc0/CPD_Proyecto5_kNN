from sklearn.datasets import fetch_openml
import numpy as np

print("Descargando MNIST....")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

print("Comprimiento en npz...")
np.savez_compressed('mnist_784.npz', data=mnist.data, target=mnist.target)
print("MNIST descargado y empaquetado.")