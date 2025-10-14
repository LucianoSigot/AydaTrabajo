#!/usr/bin/env python3


import random
import time
import tracemalloc
import sys
import math
from typing import List, Tuple

# -----------------------
# CONFIGURACIÓN
# -----------------------
SEED = 42
random.seed(SEED)

N_MAX = 1_000_000
N_LIST = [200_000, 400_000, 600_000, 800_000, 1_000_000]
K_TOP = 10

# Umbral para usar DP exacto en Part B (evitar O(n*k) memory explosion)
EXACT_DP_LIMIT = 100_000  # si n <= este valor, ejecuta DP exacto; si no, modo práctico

# Umbral para ordenar completamente en Parte C (MergeSort) por defecto
MERGESORT_LIMIT = 200_000

# -----------------------
# UTILIDADES MEDICIÓN
# -----------------------
def now_ms():
    return time.perf_counter() * 1000.0

def medir(func, *args, **kwargs):
    """Ejecuta func y devuelve (resultado, tiempo_ms, memoria_peak_kb)."""
    tracemalloc.start()
    t0 = now_ms()
    result = func(*args, **kwargs)
    t1 = now_ms()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, (t1 - t0), peak / 1024.0

# -----------------------
# GENERADOR DE DATOS P
# -----------------------
def generar_P(n: int, low: int = -1000, high: int = 1000, seed: int = None) -> List[int]:
    """Genera lista de n enteros en [low, high]."""
    if seed is not None:
        random.seed(seed)
    # Usar lista por comprensión (más eficiente que append en grandes volúmenes)
    return [random.randint(low, high) for _ in range(n)]

# -----------------------
# PARTE A: SUB-ARREGLO MÁXIMO
# -----------------------
def kadane(P: List[int]) -> Tuple[int,int,int]:
    """Kadane O(n). Devuelve (max_sum, start_idx, end_idx_inclusive)."""
    n = len(P)
    max_ending = P[0]
    max_so_far = P[0]
    start = 0
    best_start = 0
    best_end = 0
    for i in range(1, n):
        if P[i] > max_ending + P[i]:
            max_ending = P[i]
            start = i
        else:
            max_ending = max_ending + P[i]
        if max_ending > max_so_far:
            max_so_far = max_ending
            best_start = start
            best_end = i
    return max_so_far, best_start, best_end

def max_subarray_divide_conquer(P: List[int]) -> Tuple[int,int,int]:
    """Divide & Conquer clásico. Devuelve (max_sum, start, end)."""
    def helper(l, r):
        # intervalo [l, r) half-open
        if r - l == 1:
            return P[l], l, l
        m = (l + r) // 2
        left_sum, left_l, left_r = helper(l, m)
        right_sum, right_l, right_r = helper(m, r)
        # crossing max
        left_max = -10**18
        s = 0
        left_idx = m-1
        tmp = 0
        for i in range(m-1, l-1, -1):
            tmp += P[i]
            if tmp > left_max:
                left_max = tmp
                left_idx = i
        right_max = -10**18
        tmp = 0
        right_idx = m
        for j in range(m, r):
            tmp += P[j]
            if tmp > right_max:
                right_max = tmp
                right_idx = j
        cross_sum = left_max + right_max
        # elegir mejor
        best = max((left_sum, left_l, left_r),
                   (right_sum, right_l, right_r),
                   (cross_sum, left_idx, right_idx),
                   key=lambda x: x[0])
        return best
    return helper(0, len(P))

# -----------------------
# PARTE B: TOP-k PERIODOS NO SOLAPADOS
# -----------------------
def top_k_greedy_kadane(P: List[int], k: int) -> List[Tuple[int,int,int]]:
    """
    Estrategia práctica: repetir k veces Kadane, y al seleccionar un segmento,
    anularlo (setear valores a -inf) para evitar solapamiento.
    O(k*n) tiempo, O(1) memoria extra. Devuelve lista de (sum, start, end) en orden de elección.
    Nota: no garantiza óptimo global en todos los casos, pero es práctica para n grande y k pequeño.
    """
    n = len(P)
    P_work = P[:]  # copia para modificar
    NEG_INF = -10**18
    results = []
    for iter_i in range(k):
        # Kadane on P_work but must ignore NEG_INF segments; works since NEG_INF dominates
        found = False
        max_ending = P_work[0]
        max_so_far = P_work[0]
        start = 0
        best_start = 0
        best_end = 0
        for i in range(1, n):
            if P_work[i] > max_ending + P_work[i]:
                max_ending = P_work[i]
                start = i
            else:
                max_ending += P_work[i]
            if max_ending > max_so_far:
                max_so_far = max_ending
                best_start = start
                best_end = i
                found = True
        if not found and max_so_far <= NEG_INF/2:
            # no hay más segmentos útiles
            break
        # Si el mejor encontrado es NEG_INF-like (no hay nada razonable), salir
        if max_so_far <= NEG_INF/2:
            break
        results.append((max_so_far, best_start, best_end))
        # anular segmento elegido
        for j in range(best_start, best_end+1):
            P_work[j] = NEG_INF
    return results

def top_k_exact_dp(P: List[int], k: int) -> List[Tuple[int,int,int]]:
    """
    Implementación exacta DP O(n*k) con reconstrucción. Devuelve k periodos no solapados óptimos.
    ADVERTENCIA: usa arrays Python de tamaño n por cada layer -> uso de memoria grande.
    Recomendado solo para n moderados (por ejemplo <= EXACT_DP_LIMIT).
    """
    n = len(P)
    # precompute best subarray ending at i
    best_end_sum = [0]*n
    best_end_start = [0]*n
    cur_sum = P[0]
    cur_start = 0
    best_end_sum[0] = cur_sum
    best_end_start[0] = 0
    for i in range(1, n):
        if cur_sum + P[i] < P[i]:
            cur_sum = P[i]
            cur_start = i
        else:
            cur_sum += P[i]
        best_end_sum[i] = cur_sum
        best_end_start[i] = cur_start

    NEG_INF = -10**18
    # dp_prev[i] = best sum with j-1 segments in prefix up to i
    dp_prev = [NEG_INF] * n
    back_choice = [[-1]*n for _ in range(k+1)]  # back_choice[j][i] = end index chosen for dp[j][i]
    # base j=1
    max_so_far = NEG_INF
    for i in range(n):
        candidate = best_end_sum[i]
        if i > 0:
            if candidate < max_so_far:
                dp_prev[i] = max_so_far
                back_choice[1][i] = back_choice[1][i-1]
            else:
                dp_prev[i] = candidate
                back_choice[1][i] = i
            if dp_prev[i] > max_so_far:
                max_so_far = dp_prev[i]
        else:
            dp_prev[i] = candidate
            back_choice[1][i] = i
            max_so_far = dp_prev[i]

    # layers j=2..k
    for j in range(2, k+1):
        dp_curr = [NEG_INF]*n
        dp_curr[0] = dp_prev[0]
        back_choice[j][0] = -1
        for i in range(1, n):
            # option1: inherit
            opt1 = dp_curr[i-1]
            # option2: segment ending at i with start s
            s = best_end_start[i]
            prev_val = dp_prev[s-1] if s-1 >= 0 else 0
            opt2 = prev_val + best_end_sum[i]
            if opt2 >= opt1:
                dp_curr[i] = opt2
                back_choice[j][i] = i
            else:
                dp_curr[i] = opt1
                back_choice[j][i] = back_choice[j][i-1]
        dp_prev = dp_curr

    # reconstrucción
    res = []
    i = n-1
    j = k
    while j >= 1 and i >= 0:
        end_idx = back_choice[j][i]
        if end_idx == -1:
            j -= 1
            continue
        s = best_end_start[end_idx]
        sum_seg = best_end_sum[end_idx]
        res.append((sum_seg, s, end_idx))
        i = s - 1
        j -= 1
    res.reverse()
    return res

# -----------------------
# PARTE C: PERCENTILES Y MEDIANA
# -----------------------
class Counter:
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0

def quickselect_lomuto(arr: List[int], k: int, counter: Counter) -> int:
    """QuickSelect Lomuto (k is 0-based index)."""
    def partition(lo, hi):
        pivot = arr[hi]
        i = lo
        for j in range(lo, hi):
            counter.comparisons += 1
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                counter.swaps += 1
                i += 1
        arr[i], arr[hi] = arr[hi], arr[i]
        counter.swaps += 1
        return i

    lo, hi = 0, len(arr)-1
    while lo <= hi:
        p = partition(lo, hi)
        if k == p:
            return arr[p]
        elif k < p:
            hi = p - 1
        else:
            lo = p + 1
    return arr[k]

def quickselect_hoare(arr: List[int], k: int, counter: Counter) -> int:
    """QuickSelect Hoare (k is 0-based index)."""
    def partition(lo, hi):
        pivot = arr[(lo+hi)//2]
        i = lo - 1
        j = hi + 1
        while True:
            i += 1
            while True:
                counter.comparisons += 1
                if arr[i] >= pivot:
                    break
                i += 1
            j -= 1
            while True:
                counter.comparisons += 1
                if arr[j] <= pivot:
                    break
                j -= 1
            if i >= j:
                return j
            arr[i], arr[j] = arr[j], arr[i]
            counter.swaps += 1

    lo, hi = 0, len(arr)-1
    while lo < hi:
        p = partition(lo, hi)
        if k <= p:
            hi = p
        else:
            lo = p + 1
    return arr[lo]

def merge_sort_count(arr: List[int], counter: Counter) -> List[int]:
    """MergeSort con contador de comparaciones. Devuelve nuevo arreglo ordenado."""
    n = len(arr)
    if n <= 1:
        return arr[:]
    mid = n // 2
    left = merge_sort_count(arr[:mid], counter)
    right = merge_sort_count(arr[mid:], counter)
    # merge
    i = j = 0
    res = []
    while i < len(left) and j < len(right):
        counter.comparisons += 1
        if left[i] <= right[j]:
            res.append(left[i]); i += 1
        else:
            res.append(right[j]); j += 1
    if i < len(left):
        res.extend(left[i:])
    if j < len(right):
        res.extend(right[j:])
    return res

def validar_percentil(sorted_arr: List[int], value: int, idx: int) -> bool:
    """Verifica por acceso y conteo que value ocupa idx en sorted_arr (0-based)."""
    if sorted_arr[idx] != value:
        return False
    count_le = sum(1 for x in sorted_arr if x <= value)
    return count_le >= (idx+1)

# -----------------------
# FUNCIONES EXPERIMENTOS
# -----------------------
def experimento_parte_a(P_full: List[int], n_list: List[int]):
    print("=== PARTE A: Escalado Kadane (tiempos en ms) ===")
    results = []
    for n in n_list:
        print(f"\n[Escalado] n = {n}")
        P = P_full[:n]
        res_k, t_k, mem_k = medir(kadane, P)
        max_sum, s, e = res_k
        print(f" Kadane: max_sum={max_sum}, start={s}, end={e}, tiempo={t_k:.2f} ms, memPeak={mem_k:.1f} KB")
        results.append((n, t_k, mem_k, max_sum, s, e))
    # También probar Divide & Conquer en n=200k y n=maybe larger (si tu máquina lo soporta)
    for n in [200_000, len(P_full)]:
        if n > len(P_full):
            continue
        print(f"\n[D&C] n = {n}")
        P = P_full[:n]
        res_dc, t_dc, mem_dc = medir(max_subarray_divide_conquer, P)
        max_sum, s, e = res_dc
        print(f" D&C: max_sum={max_sum}, start={s}, end={e}, tiempo={t_dc:.2f} ms, memPeak={mem_dc:.1f} KB")
    return results

def experimento_parte_b(P_full: List[int], k: int):
    print("\n=== PARTE B: Top-k periodos no solapados ===")
    n = len(P_full)
    # elegir tamaño razonable para la demo: por defecto 600k si disponible
    n_for_B = min(600_000, n)
    P = P_full[:n_for_B]
    print(f" n_for_B = {n_for_B}")
    if n_for_B <= EXACT_DP_LIMIT:
        print(" Ejecutando DP exacto O(n*k) (memoria grande pero exacto)...")
        res_dp, t_dp, mem_dp = medir(top_k_exact_dp, P, k)
        print(f" DP exacto: tiempo={t_dp:.2f} ms, memPeak={mem_dp:.1f} KB")
        for i, (suma, si, ei) in enumerate(res_dp, start=1):
            print(f"  {i}. sum={suma}, start={si}, end={ei}")
    else:
        print(" Ejecutando modo práctico: extraer k veces Kadane (O(k*n), memoria baja).")
        res_greedy, t_g, mem_g = medir(top_k_greedy_kadane, P, k)
        print(f" Greedy Kadane x{k}: tiempo={t_g:.2f} ms, memPeak={mem_g:.1f} KB")
        for i, (suma, si, ei) in enumerate(res_greedy, start=1):
            print(f"  {i}. sum={suma}, start={si}, end={ei}")
        print("\nNota: este método es práctico y eficiente para n grande y k pequeño, "
              "pero no siempre coincide con la solución DP óptima en casos especiales.")
    return

def experimento_parte_c(P_full: List[int]):
    print("\n=== PARTE C: Percentiles y Mediana ===")
    # Usar n razonable para ordenar completamente
    n_for_C = min(MERGESORT_LIMIT, len(P_full))
    P = P_full[:n_for_C]
    n = len(P)
    print(f" n_for_C (ordenamiento completo) = {n}")
    # índices percentil (p50 p90 p95), con definicion idx = ceil(p/100*n)-1
    def idx_for(p):
        return max(0, int(math.ceil(p/100.0 * n) - 1))
    percentiles = [50, 90, 95]
    idxs = {p: idx_for(p) for p in percentiles}
    print(" Percentil indices:", idxs)

    # MergeSort baseline
    arr_list = P[:]  # copy
    counter_merge = Counter()
    print("\n Ejecutando MergeSort (baseline)...")
    tracemalloc.start()
    t0 = now_ms()
    sorted_arr = merge_sort_count(arr_list, counter_merge)
    t1 = now_ms()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t_merge = t1 - t0
    mem_merge = peak / 1024.0
    print(f" MergeSort: tiempo={t_merge:.2f} ms, memPeak={mem_merge:.1f} KB, comparisons={counter_merge.comparisons}")

    # Ground truth values
    gt = {}
    for p, idx in idxs.items():
        val = sorted_arr[idx]
        gt[p] = (val, idx)
        print(f"  Ground truth p{p}: value={val}, idx={idx}")

    # QuickSelect Lomuto
    for p, idx in idxs.items():
        print(f"\n QuickSelect Lomuto para p{p} (idx {idx}) ...")
        arr_copy = P[:]  # mutable list
        counter = Counter()
        tracemalloc.start()
        t0 = now_ms()
        val = quickselect_lomuto(arr_copy, idx, counter)
        t1 = now_ms()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  valor={val}, tiempo={t1-t0:.2f} ms, memPeak={peak/1024.0:.1f} KB, comparisons={counter.comparisons}, swaps={counter.swaps}")
        print(f"  Coincide con ground truth? {val == gt[p][0]}")

    # QuickSelect Hoare
    for p, idx in idxs.items():
        print(f"\n QuickSelect Hoare para p{p} (idx {idx}) ...")
        arr_copy = P[:]
        counter = Counter()
        tracemalloc.start()
        t0 = now_ms()
        val = quickselect_hoare(arr_copy, idx, counter)
        t1 = now_ms()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  valor={val}, tiempo={t1-t0:.2f} ms, memPeak={peak/1024.0:.1f} KB, comparisons={counter.comparisons}, swaps={counter.swaps}")
        print(f"  Coincide con ground truth? {val == gt[p][0]}")

    # Validación secuencial en sorted array
    for p, (val, idx) in gt.items():
        ok = validar_percentil(sorted_arr, val, idx)
        print(f"\n Validación secuencial p{p}: value={val}, idx={idx}, OK={ok}")

# -----------------------
# MAIN
# -----------------------
def main():
    print("=== Ejercicio 4 (sin numpy, sin visualización) ===")
    print(f"Generando P con n = {N_MAX} (seed={SEED}). Esto puede tardar unos segundos...")
    P_full = generar_P(N_MAX, low=-1000, high=1000, seed=SEED)
    print("Generado. Estadísticas rápidas:")
    print(f" n={len(P_full)}, min={min(P_full)}, max={max(P_full)}, mean={sum(P_full)/len(P_full):.2f}")

    # Parte A: escalado Kadane
    print("\n--- PARTE A: Sub-arreglo máximo (Kadane y D&C) ---")
    experimento_parte_a(P_full, N_LIST)

    # Parte B: top-k no solapados
    print("\n--- PARTE B: Top-k periodos no solapados ---")
    experimento_parte_b(P_full, K_TOP)

    # Parte C: percentiles y mediana
    print("\n--- PARTE C: Percentiles y Mediana ---")
    experimento_parte_c(P_full)

    print("\n=== FIN ===")

if __name__ == '__main__':
    main()
