import time
import psutil
import os
import tracemalloc
from functools import wraps

def medir_tiempo(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fin = time.time()
        print(f"Tiempo de ejecución: {fin - inicio:.6f} segundos")
        return resultado
    return wrapper

def medir_memoria():
    proceso = psutil.Process(os.getpid())
    return proceso.memory_info().rss / 1024 / 1024  # MB

class Contador:
    def __init__(self):
        self.comparaciones = 0
        self.swaps = 0
    
    def reset(self):
        self.comparaciones = 0
        self.swaps = 0