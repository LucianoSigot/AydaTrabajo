import random
import time
import tracemalloc
import sys
import math
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)

# ------------------ FUNCIONES AUXILIARES ------------------

def kadane(arr):
    # Devuelve (max_sum, start, end, comparisons)
    max_actual = arr[0]
    max_global = arr[0]
    inicio = fin = temp = 0
    comps = 0
    for i in range(1, len(arr)):
        comps += 1
        if arr[i] > max_actual + arr[i]:
            max_actual = arr[i]
            temp = i
        else:
            max_actual += arr[i]
        if max_actual > max_global:
            max_global = max_actual
            inicio = temp
            fin = i
    return max_global, inicio, fin, comps

def top_k_periodos(arr, k):
    # Heurística: repetir Kadane y "anular" tramo encontrado
    periodos = []
    comps_totales = 0
    arr_temp = arr.copy()
    for _ in range(k):
        max_suma, ini, fin, comps = kadane(arr_temp)
        comps_totales += comps
        if max_suma <= 0:
            break
        periodos.append((ini, fin, max_suma))
        # marcar el tramo como muy negativo para que no se vuelva a elegir
        for i in range(ini, fin + 1):
            arr_temp[i] = -10**15
    return periodos, comps_totales

# QuickSelect Lomuto
def partition_lomuto(arr, low, high, comps_swaps):
    pivot = arr[high]
    i = low
    for j in range(low, high):
        comps_swaps[0] += 1
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            comps_swaps[1] += 1
            i += 1
    arr[i], arr[high] = arr[high], arr[i]
    comps_swaps[1] += 1
    return i

def quickselect_lomuto(arr, k, comps_swaps):
    low, high = 0, len(arr) - 1
    while low <= high:
        pi = partition_lomuto(arr, low, high, comps_swaps)
        if k == pi:
            return arr[k]
        elif k < pi:
            high = pi - 1
        else:
            low = pi + 1
    return arr[k]

# QuickSelect Hoare (corregido)
def quickselect_hoare(arr, k, comps_swaps):
    def partition(a, lo, hi):
        pivot = a[(lo + hi) // 2]
        i, j = lo, hi
        while True:
            while a[i] < pivot:
                comps_swaps[0] += 1
                i += 1
            comps_swaps[0] += 1
            while a[j] > pivot:
                comps_swaps[0] += 1
                j -= 1
            comps_swaps[0] += 1
            if i >= j:
                return j
            a[i], a[j] = a[j], a[i]
            comps_swaps[1] += 1
            i += 1
            j -= 1

    lo, hi = 0, len(arr) - 1
    while True:
        if lo >= hi:
            return arr[k]
        p = partition(arr, lo, hi)
        if k <= p:
            hi = p
        else:
            lo = p + 1

# MergeSort iterativo (baseline)
def merge_sort(arr):
    width = 1
    n = len(arr)
    res = arr.copy()
    while width < n:
        for i in range(0, n, 2 * width):
            left = res[i:i + width]
            right = res[i + width:i + 2 * width]
            merged = []
            l = r = 0
            while l < len(left) and r < len(right):
                if left[l] <= right[r]:
                    merged.append(left[l])
                    l += 1
                else:
                    merged.append(right[r])
                    r += 1
            merged += left[l:]
            merged += right[r:]
            res[i:i + len(merged)] = merged
        width *= 2
    return res

# Helper: downsample una serie para visualizar sin numpy
def downsample_list(xs, max_points=200000):
    n = len(xs)
    if n <= max_points:
        return xs
    step = math.ceil(n / max_points)
    return [xs[i] for i in range(0, n, step)]

# ------------------ EJERCICIO 4 CON GRAFICOS ------------------
def ejercicio4(run_full_n=True, save_plots=True, show_plots=False):
    """
    run_full_n: si True, realiza la medición escalado incluyendo n=1_000_000.
    save_plots: guarda las figuras como PNG.
    show_plots: muestra las figuras con plt.show() (ventana interactiva).
    """
    results_text = []
    random.seed(12345)  # reproducible

    # ---------- Parte A: Escalado n vs tiempo (Kadane) ----------
    ns = [200_000, 400_000, 600_000, 800_000, 1_000_000] if run_full_n else [200_000, 400_000, 600_000]
    times = []

    results_text.append("PARTE A — Escalado: n vs tiempo (Kadane)\n")
    for n in ns:
        # generamos un arreglo aleatorio
        arr = [random.randint(-100, 100) for _ in range(n)]
        tracemalloc.start()
        t0 = time.perf_counter()
        _, _, _, comps = kadane(arr)
        t1 = time.perf_counter()
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = t1 - t0
        times.append(elapsed)
        results_text.append(f"n = {n:,} -> tiempo: {elapsed:.6f}s, comps: {comps}, memoria pico: {mem_peak/1024:.2f} KB")

    # Graficar n vs tiempo
    plt.figure(figsize=(8, 5))
    plt.plot(ns, times, marker='o')
    plt.title("Escalado: n vs Tiempo (Kadane)")
    plt.xlabel("n (tamaño del arreglo)")
    plt.ylabel("Tiempo (s)")
    plt.grid(True)
    if save_plots:
        plt.tight_layout()
        plt.savefig("kadane_scaling.png", dpi=150)
        results_text.append("Gráfico de escalado guardado en 'kadane_scaling.png'")
    if show_plots:
        plt.show()
    plt.close()

    # ---------- Parte A: Ejecución final para n=1_000_000 (subarreglo máximo y gráfico de la serie) ----------
    n_final = 1_000_000
    results_text.append("\nPARTE A — Ejecución final para n = 1_000_000 (Kadane + gráfico)\n")
    arr = [random.randint(-100, 100) for _ in range(n_final)]

    tracemalloc.start()
    t0 = time.perf_counter()
    max_sum, start_idx, end_idx, comps_kadane = kadane(arr)
    t1 = time.perf_counter()
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_kadane = t1 - t0

    results_text.append(f"Mejor periodo: {start_idx} a {end_idx}")
    results_text.append(f"Valor máximo: {max_sum}")
    results_text.append(f"Comparaciones (Kadane): {comps_kadane}")
    results_text.append(f"Tiempo (Kadane): {elapsed_kadane:.6f}s")
    results_text.append(f"Memoria pico: {mem_peak/1024:.2f} KB")

    # Graficar la serie con tramo óptimo resaltado.
    # Para no colapsar el plot, downsample la serie para mostrar un máximo razonable de puntos.
    # Pero aseguramos marcar correctamente la posición del tramo (calculamos escala).
    max_plot_points = 200_000  # máximo de puntos a dibujar
    if n_final <= max_plot_points:
        xs = list(range(n_final))
        ys = arr
        factor = 1
    else:
        # downsample mostrando cada step puntos
        step = math.ceil(n_final / max_plot_points)
        xs = list(range(0, n_final, step))
        ys = [arr[i] for i in xs]
        factor = step

    plt.figure(figsize=(12, 5))
    plt.plot(xs, ys)  # serie general
    # Resaltar tramo óptimo: calculamos indices en la escala de xs:
    # Encontramos la first and last xs indices that fall inside start_idx..end_idx
    seg_x = []
    seg_y = []
    # si factor==1, x==index; si factor>1, xs = 0, step, 2*step...
    # Tomamos puntos dentro del tramo
    for i, x in enumerate(xs):
        true_idx = x
        if start_idx <= true_idx <= end_idx:
            seg_x.append(x)
            seg_y.append(ys[i])
    if seg_x and seg_y:
        # resaltar con línea más gruesa
        plt.plot(seg_x, seg_y, linewidth=2.5)
    plt.title("Serie de utilidades y mejor periodo resaltado")
    plt.xlabel("Día (índice)")
    plt.ylabel("Utilidad diaria")
    plt.grid(True)
    if save_plots:
        plt.tight_layout()
        plt.savefig("series_with_best_period.png", dpi=150)
        results_text.append("Gráfico de serie con mejor periodo guardado en 'series_with_best_period.png'")
    if show_plots:
        plt.show()
    plt.close()

    # ---------- Parte B: Top-k periodos no solapados ----------
    results_text.append("\nPARTE B — Top-10 periodos no solapados (heurística Kadane repetido)\n")
    tracemalloc.start()
    t0 = time.perf_counter()
    topk, comps_topk = top_k_periodos(arr, 10)
    t1 = time.perf_counter()
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_topk = t1 - t0
    for i, (ini, fin, val) in enumerate(topk):
        results_text.append(f"{i+1}) inicio {ini}, fin {fin}, valor {val}")
    results_text.append(f"Comparaciones top-k: {comps_topk}")
    results_text.append(f"Tiempo top-k: {elapsed_topk:.6f}s")
    results_text.append(f"Memoria pico top-k: {mem_peak/1024:.2f} KB")

    # ---------- Parte C: Percentiles y mediana ----------
    results_text.append("\nPARTE C — Percentiles y mediana (QuickSelect Lomuto / Hoare / MergeSort)\n")
    percentiles = [50, 90, 95]
    posiciones = [int(n_final * p / 100) for p in percentiles]

    # Lomuto
    arr_l = arr.copy()
    comps_swaps = [0, 0]
    tracemalloc.start()
    t0 = time.perf_counter()
    vals_l = [quickselect_lomuto(arr_l, pos, comps_swaps) for pos in posiciones]
    t1 = time.perf_counter()
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results_text.append(f"QuickSelect Lomuto - percentiles {percentiles}: {vals_l}")
    results_text.append(f"Comparaciones: {comps_swaps[0]}, Swaps: {comps_swaps[1]}")
    results_text.append(f"Tiempo Lomuto: {t1 - t0:.6f}s")
    results_text.append(f"Memoria pico Lomuto: {mem_peak/1024:.2f} KB")

    # Hoare
    arr_h = arr.copy()
    comps_swaps = [0, 0]
    tracemalloc.start()
    t0 = time.perf_counter()
    vals_h = [quickselect_hoare(arr_h, pos, comps_swaps) for pos in posiciones]
    t1 = time.perf_counter()
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results_text.append(f"QuickSelect Hoare - percentiles {percentiles}: {vals_h}")
    results_text.append(f"Comparaciones: {comps_swaps[0]}, Swaps: {comps_swaps[1]}")
    results_text.append(f"Tiempo Hoare: {t1 - t0:.6f}s")
    results_text.append(f"Memoria pico Hoare: {mem_peak/1024:.2f} KB")

    # MergeSort baseline
    arr_m = arr.copy()
    tracemalloc.start()
    t0 = time.perf_counter()
    sorted_arr = merge_sort(arr_m)
    vals_m = [sorted_arr[pos] for pos in posiciones]
    t1 = time.perf_counter()
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results_text.append(f"MergeSort baseline - percentiles {percentiles}: {vals_m}")
    results_text.append(f"Tiempo MergeSort: {t1 - t0:.6f}s")
    results_text.append(f"Memoria pico MergeSort: {mem_peak/1024:.2f} KB")

    # Verificación por búsqueda secuencial (opcional, simple comprobación)
    # Contar cuántos <= umbral p50 (usando sorted_arr para verificar)
    p50_val = vals_m[0]
    count_le_p50 = sum(1 for v in sorted_arr if v <= p50_val)
    results_text.append(f"Verificación: cantidad <= p50 ({p50_val}): {count_le_p50}")

    # Empaquetar resultado textual
    full_text = "\n".join(results_text)
    return full_text

# Ejecución directa (si corres este archivo)
if __name__ == "__main__":
    # correr todo (puede tardar minutos por n=1_000_000)
    resumen = ejercicio4(run_full_n=True, save_plots=True, show_plots=False)
    print(resumen)
    print("\nListo. Figuras guardadas: 'kadane_scaling.png' y 'series_with_best_period.png'")