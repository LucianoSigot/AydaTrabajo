import random
import itertools
import time
import tracemalloc
import math
import copy

# --------------------- ASIGNACION ---------------------

def generar_matriz_distancias(n=9, min_dist=1, max_dist=100):
    """Genera una matriz de distancias n x n aleatoria"""
    return [[0 if i==j else random.randint(min_dist, max_dist) for j in range(n)] for i in range(n)]

# Fuerza Bruta Total para asignación
def asignacion_fuerza_bruta(matriz):
    n = len(matriz)
    min_costo = float('inf')
    mejor_perm = None
    comparaciones = 0

    tracemalloc.start()
    inicio = time.perf_counter()

    for perm in itertools.permutations(range(n)):
        costo = sum(matriz[i][perm[i]] for i in range(n))
        comparaciones += 1
        if costo < min_costo:
            min_costo = costo
            mejor_perm = perm

    fin = time.perf_counter()
    mem_actual, mem_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return mejor_perm, min_costo, comparaciones, fin - inicio, mem_pico / 1024

# Reducción y vencerás usando mediana
def mediana_tupla(lista):
    valores = sorted([v for _, v in lista])
    n = len(valores)
    if n % 2 == 1:
        return valores[n // 2]
    else:
        return (valores[n//2 - 1] + valores[n//2]) / 2

def reduccion_venceras_mediana(matriz):
    n = len(matriz)
    matriz_copia = copy.deepcopy(matriz)

    tracemalloc.start()
    inicio = time.perf_counter()

    # Reducir filas
    for i in range(n):
        min_fila = min(matriz_copia[i])
        for j in range(n):
            matriz_copia[i][j] -= min_fila

    # Reducir columnas
    for j in range(n):
        min_col = min(matriz_copia[i][j] for i in range(n))
        for i in range(n):
            matriz_copia[i][j] -= min_col

    # Asignación usando mediana por fila
    asignacion = [-1]*n
    usados_col = set()
    comparaciones = 0

    for i in range(n):
        fila = [(j, matriz_copia[i][j]) for j in range(n) if j not in usados_col]
        med = mediana_tupla(fila)
        dif_min = float('inf')
        elegido = fila[0][0]
        for j, val in fila:
            comparaciones += 1
            if abs(val - med) < dif_min:
                dif_min = abs(val - med)
                elegido = j
        asignacion[i] = elegido
        usados_col.add(elegido)

    costo = sum(matriz[i][asignacion[i]] for i in range(n))

    fin = time.perf_counter()
    mem_actual, mem_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return asignacion, costo, comparaciones, fin - inicio, mem_pico / 1024

# --------------------- TSP ---------------------

def generar_puntos(n=9, max_coord=100):
    return [(random.randint(0, max_coord), random.randint(0, max_coord)) for _ in range(n)]

def distancia(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# TSP Fuerza Bruta
def tsp_fuerza_bruta(puntos):
    n = len(puntos)
    min_costo = float('inf')
    mejor_ruta = None
    comparaciones = 0

    tracemalloc.start()
    inicio = time.perf_counter()

    for perm in itertools.permutations(range(n)):
        costo = sum(distancia(puntos[perm[i]], puntos[perm[(i+1)%n]]) for i in range(n))
        comparaciones += 1
        if costo < min_costo:
            min_costo = costo
            mejor_ruta = perm

    fin = time.perf_counter()
    mem_actual, mem_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return mejor_ruta, min_costo, comparaciones, fin - inicio, mem_pico / 1024

# TSP Heurística Insertion Sort
def tsp_insertion_heuristica(puntos):
    n = len(puntos)
    ruta = [0, 1]
    swaps = 0
    comps = 0

    tracemalloc.start()
    inicio = time.perf_counter()

    for i in range(2, n):
        mejor_pos = 0
        min_aum = float('inf')
        for j in range(len(ruta)):
            comps += 1
            costo_ant = distancia(puntos[ruta[j]], puntos[ruta[(j+1)%len(ruta)]])
            costo_nuevo = distancia(puntos[ruta[j]], puntos[i]) + distancia(puntos[i], puntos[ruta[(j+1)%len(ruta)]])
            aumento = costo_nuevo - costo_ant
            if aumento < min_aum:
                min_aum = aumento
                mejor_pos = j + 1
        ruta.insert(mejor_pos, i)
        swaps += 1

    costo_total = sum(distancia(puntos[ruta[i]], puntos[ruta[(i+1)%n]]) for i in range(n))

    fin = time.perf_counter()
    mem_actual, mem_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return ruta, costo_total, comps, swaps, fin - inicio, mem_pico / 1024

# TSP Branch and Bound (simplificado)
def tsp_branch_and_bound(puntos):
    n = len(puntos)
    mejor_ruta = None
    min_costo = float('inf')
    comparaciones = 0

    def bnb(ruta, visitados, costo_actual):
        nonlocal mejor_ruta, min_costo, comparaciones
        if len(ruta) == n:
            costo_total = costo_actual + distancia(puntos[ruta[-1]], puntos[ruta[0]])
            comparaciones +=1
            if costo_total < min_costo:
                min_costo = costo_total
                mejor_ruta = ruta.copy()
            return
        for i in range(n):
            if not visitados[i]:
                visitados[i] = True
                costo_nuevo = costo_actual
                if ruta:
                    costo_nuevo += distancia(puntos[ruta[-1]], puntos[i])
                if costo_nuevo < min_costo:
                    ruta.append(i)
                    bnb(ruta, visitados, costo_nuevo)
                    ruta.pop()
                visitados[i] = False

    tracemalloc.start()
    inicio = time.perf_counter()
    bnb([], [False]*n, 0)
    fin = time.perf_counter()
    mem_actual, mem_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return mejor_ruta, min_costo, comparaciones, fin - inicio, mem_pico / 1024

# --------------------- EJERCICIO 3 ---------------------

def ejercicio3():
    resultados = []

    # Fuerza Bruta n=9 (Asignación)
    matriz_fb = generar_matriz_distancias(9)
    asign_fb, costo_fb, comps_fb, tiempo_fb, mem_fb = asignacion_fuerza_bruta(matriz_fb)
    resultados.append("EJERCICIO 3 - Asignación de camiones (Fuerza Bruta n=9):")
    resultados.append(f"Asignación: {asign_fb}")
    resultados.append(f"Costo total: {costo_fb}")
    resultados.append(f"Comparaciones: {comps_fb}")
    resultados.append(f"Tiempo de ejecución: {tiempo_fb:.6f}s")
    resultados.append(f"Memoria pico: {mem_fb:.2f} KB\n")

    # Reducción y Vencerás usando Mediana n=9 (Asignación)
    matriz_rv = generar_matriz_distancias(9)
    asign_rv, costo_rv, comps_rv, tiempo_rv, mem_rv = reduccion_venceras_mediana(matriz_rv)
    resultados.append("EJERCICIO 3 - Asignación de camiones (Reducción y Vencerás Mediana n=9):")
    resultados.append(f"Asignación: {asign_rv}")
    resultados.append(f"Costo total: {costo_rv}")
    resultados.append(f"Comparaciones: {comps_rv}")
    resultados.append(f"Tiempo de ejecución: {tiempo_rv:.6f}s")
    resultados.append(f"Memoria pico: {mem_rv:.2f} KB\n")

    # TSP Fuerza Bruta n=9
    puntos_fb = generar_puntos(9)
    ruta_fb, costo_fb, comps_fb, tiempo_fb, mem_fb = tsp_fuerza_bruta(puntos_fb)
    resultados.append("EJERCICIO 3 - TSP Fuerza Bruta (n=9):")
    resultados.append(f"Ruta: {ruta_fb}")
    resultados.append(f"Costo total: {costo_fb:.2f}")
    resultados.append(f"Comparaciones: {comps_fb}")
    resultados.append(f"Tiempo de ejecución: {tiempo_fb:.6f}s")
    resultados.append(f"Memoria pico: {mem_fb:.2f} KB\n")

    # TSP Heurística Insertion Sort n=9
    puntos_ins = generar_puntos(9)
    ruta_ins, costo_ins, comps_ins, swaps_ins, tiempo_ins, mem_ins = tsp_insertion_heuristica(puntos_ins)
    resultados.append("EJERCICIO 3 - TSP Heurística Insertion Sort (n=9):")
    resultados.append(f"Ruta: {ruta_ins}")
    resultados.append(f"Costo total: {costo_ins:.2f}")
    resultados.append(f"Comparaciones: {comps_ins}")
    resultados.append(f"Swaps: {swaps_ins}")
    resultados.append(f"Tiempo de ejecución: {tiempo_ins:.6f}s")
    resultados.append(f"Memoria pico: {mem_ins:.2f} KB\n")

    # TSP Branch and Bound n=9
    puntos_bb = generar_puntos(9)
    ruta_bb, costo_bb, comps_bb, tiempo_bb, mem_bb = tsp_branch_and_bound(puntos_bb)
    resultados.append("EJERCICIO 3 - TSP Branch and Bound (n=9):")
    resultados.append(f"Ruta: {ruta_bb}")
    resultados.append(f"Costo total: {costo_bb:.2f}")
    resultados.append(f"Comparaciones: {comps_bb}")
    resultados.append(f"Tiempo de ejecución: {tiempo_bb:.6f}s")
    resultados.append(f"Memoria pico: {mem_bb:.2f} KB")

    return "\n".join(resultados)


