import random
import time
import psutil
import os
import statistics

# =============================
# FUNCIONES DE APOYO
# =============================
def medir_recurso(func):
    """Decorador para medir tiempo, memoria, comparaciones y swaps."""
    def wrapper(*args, **kwargs):
        proceso = psutil.Process(os.getpid())
        memoria_inicio = proceso.memory_info().rss
        inicio = time.perf_counter()
        comparaciones, swaps = func(*args, **kwargs)
        fin = time.perf_counter()
        memoria_fin = proceso.memory_info().rss
        print(f"⏱ Tiempo: {fin - inicio:.4f}s | 🔢 Comparaciones: {comparaciones} | 🔁 Swaps: {swaps} | 💾 Memoria: {(memoria_fin - memoria_inicio)/1024:.2f} KB\n")
    return wrapper


# =============================
# a) MERGE SORT
# =============================
def merge_sort(arr):
    comparaciones = 0
    swaps = 0

    def merge(left, right):
        nonlocal comparaciones, swaps
        resultado = []
        i = j = 0
        while i < len(left) and j < len(right):
            comparaciones += 1
            if left[i] < right[j]:
                resultado.append(left[i])
                i += 1
            else:
                resultado.append(right[j])
                j += 1
                swaps += 1
        resultado.extend(left[i:])
        resultado.extend(right[j:])
        return resultado

    if len(arr) <= 1:
        return arr, comparaciones, swaps

    mid = len(arr) // 2
    izquierda, ci, si = merge_sort(arr[:mid])
    derecha, cd, sd = merge_sort(arr[mid:])
    combinado = merge(izquierda, derecha)
    comparaciones += ci + cd
    swaps += si + sd
    return combinado, comparaciones, swaps


@medir_recurso
def mediana_merge_sort(datos):
    ordenado, comp, swaps = merge_sort(datos)
    n = len(ordenado)
    mediana = ordenado[n//2] if n % 2 == 1 else (ordenado[n//2 - 1] + ordenado[n//2]) / 2
    print(f"🔹 Mediana (MergeSort): {mediana}")
    return comp, swaps


# =============================
# b) QUICKSELECT (Lomuto)
# =============================
def quickselect_lomuto(arr, k):
    comparaciones = swaps = 0

    def lomuto(low, high):
        nonlocal comparaciones, swaps
        pivot = arr[high]
        i = low
        for j in range(low, high):
            comparaciones += 1
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                swaps += 1
        arr[i], arr[high] = arr[high], arr[i]
        swaps += 1
        return i

    low, high = 0, len(arr) - 1
    while True:
        p = lomuto(low, high)
        if p == k:
            return arr[p], comparaciones, swaps
        elif p < k:
            low = p + 1
        else:
            high = p - 1


@medir_recurso
def mediana_quickselect_lomuto(datos):
    k = len(datos)//2
    mediana, comp, swaps = quickselect_lomuto(datos.copy(), k)
    print(f"🔹 Mediana (QuickSelect-Lomuto): {mediana}")
    return comp, swaps


# =============================
# c) QUICKSELECT (Hoare)
# =============================
def quickselect_hoare(arr, k):
    comparaciones = swaps = 0

    def hoare(low, high):
        nonlocal comparaciones, swaps
        pivot = arr[(low + high)//2]
        i, j = low - 1, high + 1
        while True:
            i += 1
            while arr[i] < pivot:
                i += 1
                comparaciones += 1
            j -= 1
            while arr[j] > pivot:
                j -= 1
                comparaciones += 1
            if i >= j:
                return j
            arr[i], arr[j] = arr[j], arr[i]
            swaps += 1

    low, high = 0, len(arr)-1
    while True:
        p = hoare(low, high)
        if k <= p:
            high = p
        else:
            low = p + 1
        if low >= high:
            return arr[k], comparaciones, swaps


@medir_recurso
def mediana_quickselect_hoare(datos):
    k = len(datos)//2
    mediana, comp, swaps = quickselect_hoare(datos.copy(), k)
    print(f"🔹 Mediana (QuickSelect-Hoare): {mediana}")
    return comp, swaps


# =============================
# d) BUBBLE SORT + BÚSQUEDA SECUENCIAL
# =============================
def bubble_sort(arr):
    comparaciones = swaps = 0
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            comparaciones += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
    return arr, comparaciones, swaps


@medir_recurso
def mediana_bubble(datos):
    ordenado, comp, swaps = bubble_sort(datos[:10000])  # ⚠️ Limitamos a 10k para que no tarde horas
    n = len(ordenado)
    mediana = ordenado[n//2] if n % 2 == 1 else (ordenado[n//2 - 1] + ordenado[n//2]) / 2
    print(f"🔹 Mediana (Bubble Sort): {mediana}")
    return comp, swaps


# =============================
# EJECUCIÓN PRINCIPAL
# =============================
if __name__ == "__main__":
    print("🧮 Generando datos...")
    datos = [random.randint(0, int(1e7)) for _ in range(1000000)]  # 1 millón para pruebas
    print("Datos generados ✅\n")

    mediana_merge_sort(datos.copy())
    mediana_quickselect_lomuto(datos.copy())
    mediana_quickselect_hoare(datos.copy())
    mediana_bubble(datos.copy())
