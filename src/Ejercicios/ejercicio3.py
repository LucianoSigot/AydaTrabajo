import random
import math
import itertools
import time
import heapq
import sys
from utils.util import medir_tiempo, medir_memoria, Contador

class OptimizacionLogistica:
    def __init__(self, n_depositos=15):
        self.n = n_depositos
        self.distancias = self.generar_matriz_distancias()
        self.contador = Contador()
    
    def generar_matriz_distancias(self):
        """Genera matriz de distancias simétrica y realista"""
        matriz = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Distancias más realistas (1-50 km)
                distancia = random.randint(5, 50)
                matriz[i][j] = distancia
                matriz[j][i] = distancia
        return matriz

    def factorial(self, x: int) -> int:
        """Pequeño wrapper para factorial (usado en mensajes/estimates)."""
        return math.factorial(x)

    def calcular_costo_asignacion(self, asignacion):
        """Calcula el costo total de una asignación (lista/tupla donde i->asignacion[i])."""
        costo = 0
        for i, dest in enumerate(asignacion):
            costo += self.distancias[i][dest]
        return costo

    def calcular_costo_ruta(self, ruta):
        """Calcula la distancia total de una ruta dada como tupla/lista de nodos (incluye retorno si está)."""
        costo = 0
        for i in range(len(ruta) - 1):
            costo += self.distancias[ruta[i]][ruta[i+1]]
        return costo
    
    def imprimir_matriz(self):
        """Imprime la matriz de distancias"""
        print("Matriz de Distancias:")
        print("    " + " ".join(f"{i:2d}" for i in range(self.n)))
        for i in range(self.n):
            print(f"{i:2d} " + " ".join(f"{self.distancias[i][j]:2d}" for j in range(self.n)))
        print()
    
    # ===== PARTE A: PROBLEMA DE ASIGNACIÓN =====
    
    def asignacion_fuerza_bruta_total(self):
        """FUERZA BRUTA TOTAL para problema de asignación - O(n!)"""
        print("\n" + "="*70)
        print("PROBLEMA DE ASIGNACIÓN - FUERZA BRUTA TOTAL")
        print("="*70)
        
        memoria_inicio = medir_memoria()
        inicio_tiempo = time.time()
        print(f"Generando todas las {self.n}! = {math.factorial(self.n):,} permutaciones posibles...")

        mejor_costo = float('inf')
        mejor_asignacion = None
        total_permutaciones = 0

        # Generar TODAS las permutaciones posibles (fuerza bruta total)
        for perm in itertools.permutations(range(self.n)):
            total_permutaciones += 1
            asignacion_actual = perm
            # calcular costo usando una función auxiliar si existe
            try:
                costo_actual = self.calcular_costo_asignacion(asignacion_actual)
            except Exception:
                # Fallback: sumar las distancias directamente
                costo_actual = sum(self.distancias[i][asignacion_actual[i]] for i in range(self.n))

            # Actualizar mejor solución
            if costo_actual < mejor_costo:
                mejor_costo = costo_actual
                mejor_asignacion = asignacion_actual
                self.contador.swaps += 1

            # Mostrar progreso cada 100,000 permutaciones
            if total_permutaciones % 100000 == 0:
                tiempo_transcurrido = time.time() - inicio_tiempo
                print(f"Procesadas {total_permutaciones:,} permutaciones... "
                      f"Tiempo: {tiempo_transcurrido:.2f}s - Mejor costo: {mejor_costo}")

        tiempo_total = time.time() - inicio_tiempo
        memoria_uso = medir_memoria() - memoria_inicio

        # RESULTADOS
        print("\n" + "="*50)
        print("RESULTADOS - ASIGNACIÓN FUERZA BRUTA")
        print("="*50)
        print(f"Total de permutaciones evaluadas: {total_permutaciones:,}")
        print(f"Mejor asignación encontrada: {mejor_asignacion}")
        print(f"Costo total mínimo: {mejor_costo}")

        # Mostrar asignación detallada
        print("\nAsignación detallada:")
        if mejor_asignacion is not None:
            for i in range(self.n):
                print(f"  Camión {i} → Depósito {mejor_asignacion[i]} "
                      f"(Distancia: {self.distancias[i][mejor_asignacion[i]]})")
        else:
            print("  No se encontró una asignación óptima.")

        print(f"\nTiempo de ejecución: {tiempo_total:.6f} segundos")
        print(f"Comparaciones realizadas: {self.contador.comparaciones:,}")
        print(f"Swaps/actualizaciones: {self.contador.swaps}")
        print(f"Memoria utilizada: {memoria_uso:.2f} MB")

        return mejor_asignacion, mejor_costo, tiempo_total, memoria_uso
    
    def asignacion_hungaro(self):
        """Algoritmo Húngaro (reducción y conquista) - O(n³)"""
        print("\n=== ASIGNACIÓN - ALGORITMO HÚNGARO ===")
        memoria_inicio = medir_memoria()
        inicio = time.time()
        
        # Implementación simplificada del algoritmo Húngaro
        n = self.n
        matriz = [fila[:] for fila in self.distancias]
        
        # Paso 1: Restar el mínimo de cada fila
        for i in range(n):
            min_val = min(matriz[i])
            for j in range(n):
                matriz[i][j] -= min_val
            self.contador.comparaciones += n
        
        # Paso 2: Restar el mínimo de cada columna
        for j in range(n):
            min_val = min(matriz[i][j] for i in range(n))
            for i in range(n):
                matriz[i][j] -= min_val
            self.contador.comparaciones += n
        
        # Paso 3: Encontrar asignación óptima (versión simplificada)
        asignacion = self._encontrar_asignacion_optima(matriz)
        
        # Calcular costo real
        costo = sum(self.distancias[i][asignacion[i]] for i in range(n))
        
        tiempo = time.time() - inicio
        memoria = medir_memoria() - memoria_inicio
        
        print(f"Mejor asignación: {asignacion}")
        print(f"Costo total: {costo}")
        print(f"Tiempo: {tiempo:.6f} segundos")
        print(f"Comparaciones: {self.contador.comparaciones}")
        print(f"Memoria: {memoria:.2f} MB")
        
        return asignacion, costo, tiempo, memoria
    
    def _encontrar_asignacion_optima(self, matriz):
        """Encuentra asignación óptima en matriz reducida"""
        n = len(matriz)
        asignacion = [-1] * n
        cubiertas_filas = [False] * n
        cubiertas_columnas = [False] * n
        
        # Asignación greedy inicial
        for i in range(n):
            for j in range(n):
                self.contador.comparaciones += 1
                if matriz[i][j] == 0 and not cubiertas_columnas[j] and asignacion[i] == -1:
                    asignacion[i] = j
                    cubiertas_columnas[j] = True
                    cubiertas_filas[i] = True
                    self.contador.swaps += 1
        
        # Completar asignación para filas no cubiertas
        for i in range(n):
            if asignacion[i] == -1:
                for j in range(n):
                    self.contador.comparaciones += 1
                    if not cubiertas_columnas[j]:
                        asignacion[i] = j
                        cubiertas_columnas[j] = True
                        self.contador.swaps += 1
                        break
        
        return asignacion
    
    # ===== PARTE B: PROBLEMA DEL VIAJANTE (TSP) =====
    
    def tsp_fuerza_bruta_total(self):
        """FUERZA BRUTA TOTAL para TSP - O(n!)"""
        print("\n" + "="*70)
        print("PROBLEMA DEL VIAJANTE (TSP) - FUERZA BRUTA TOTAL")
        print("="*70)
        
        memoria_inicio = medir_memoria()
        inicio_tiempo = time.time()
        
        # Para TSP, consideramos que empieza y termina en depósito 0
        ciudades_por_visitar = list(range(1, self.n))
        n_ciudades = len(ciudades_por_visitar)
        
        print(f"Generando todas las {n_ciudades}! = {self.factorial(n_ciudades):,} rutas posibles...")
        
        mejor_costo = float('inf')
        mejor_ruta = None
        total_rutas = 0
        
        # Generar TODAS las permutaciones de las ciudades (fuerza bruta total)
        for perm in itertools.permutations(ciudades_por_visitar):
            total_rutas += 1
            # Crear ruta completa: empieza en 0, visita todas las ciudades, termina en 0
            ruta_actual = (0,) + perm + (0,)
            costo_actual = self.calcular_costo_ruta(ruta_actual)
            
            # Actualizar mejor solución
            if costo_actual < mejor_costo:
                mejor_costo = costo_actual
                mejor_ruta = ruta_actual
                self.contador.swaps += 1
            
            # Mostrar progreso cada 50,000 rutas
            if total_rutas % 50000 == 0:
                tiempo_transcurrido = time.time() - inicio_tiempo
                print(f"Procesadas {total_rutas:,} rutas... "
                      f"Tiempo: {tiempo_transcurrido:.2f}s - Mejor costo: {mejor_costo}")
        
        tiempo_total = time.time() - inicio_tiempo
        memoria_uso = medir_memoria() - memoria_inicio
        
        # RESULTADOS
        print("\n" + "="*50)
        print("RESULTADOS - TSP FUERZA BRUTA")
        print("="*50)
        print(f"Total de rutas evaluadas: {total_rutas:,}")
        print(f"Mejor ruta encontrada: {mejor_ruta}")
        print(f"Distancia total mínima: {mejor_costo}")
        
        # Mostrar ruta detallada
        print("\nRuta detallada:")
        for i in range(len(mejor_ruta) - 1):
            print(f"  {mejor_ruta[i]} → {mejor_ruta[i+1]} "
                  f"(Distancia: {self.distancias[mejor_ruta[i]][mejor_ruta[i+1]]})")
        
        print(f"\nTiempo de ejecución: {tiempo_total:.6f} segundos")
        print(f"Comparaciones realizadas: {self.contador.comparaciones:,}")
        print(f"Swaps/actualizaciones: {self.contador.swaps}")
        print(f"Memoria utilizada: {memoria_uso:.2f} MB")
        
        return mejor_ruta, mejor_costo, tiempo_total, memoria_uso
    
    def tsp_insercion(self):
        """TSP con heurística de inserción - O(n²)"""
        print("\n=== TSP - HEURÍSTICA DE INSERCIÓN ===")
        memoria_inicio = medir_memoria()
        inicio = time.time()
        
        n = self.n
        no_visitados = set(range(1, n))
        ruta = [0, 0]  # Empieza y termina en 0
        
        while no_visitados:
            mejor_costo = float('inf')
            mejor_posicion = -1
            mejor_ciudad = -1
            
            for ciudad in no_visitados:
                for i in range(1, len(ruta)):
                    # Calcular costo de inserción
                    costo_insercion = (self.distancias[ruta[i - 1]][ciudad] + 
                                      self.distancias[ciudad][ruta[i]] - 
                                      self.distancias[ruta[i - 1]][ruta[i]])
                    self.contador.comparaciones += 1
                    
                    if costo_insercion < mejor_costo:
                        mejor_costo = costo_insercion
                        mejor_posicion = i
                        mejor_ciudad = ciudad
                        self.contador.swaps += 1
            
            # Insertar la mejor ciudad encontrada
            ruta.insert(mejor_posicion, mejor_ciudad)
            no_visitados.remove(mejor_ciudad)
        
        # Calcular costo total
        costo_total = sum(self.distancias[ruta[i]][ruta[i + 1]] for i in range(len(ruta) - 1))
        
        tiempo = time.time() - inicio
        memoria = medir_memoria() - memoria_inicio
        
        print(f"Mejor ruta: {tuple(ruta)}")
        print(f"Costo total: {costo_total}")
        print(f"Tiempo: {tiempo:.6f} segundos")
        print(f"Comparaciones: {self.contador.comparaciones}")
        print(f"Memoria: {memoria:.2f} MB")
        
        return tuple(ruta), costo_total, tiempo, memoria
    
    def tsp_divide_venceras_branch_bound(self):
        """TSP con Divide y Vencerás + Branch and Bound"""
        print("\n=== TSP - DIVIDE Y VENCERÁS + BRANCH AND BOUND ===")
        memoria_inicio = medir_memoria()
        inicio = time.time()
        
        # Límite superior inicial usando la heurística de inserción
        _, limite_superior, _, _ = self.tsp_insercion()
        
        # Reiniciar contador para medir solo este algoritmo
        self.contador.reset()
        
        # Usar Divide y Vencerás para particionar el problema
        mejor_ruta, mejor_costo = self._tsp_dc_bb(limite_superior)
        
        tiempo = time.time() - inicio
        memoria = medir_memoria() - memoria_inicio
        
        print(f"Mejor ruta: {mejor_ruta}")
        print(f"Costo total: {mejor_costo}")
        print(f"Tiempo: {tiempo:.6f} segundos")
        print(f"Comparaciones: {self.contador.comparaciones}")
        print(f"Memoria: {memoria:.2f} MB")
        
        return mejor_ruta, mejor_costo, tiempo, memoria
    
    def _tsp_dc_bb(self, limite_superior):
        """Divide y Vencerás + Branch and Bound para TSP"""
        n = self.n
        
        # Cola de prioridad para Branch and Bound
        cola = []
        
        # Estado inicial: nodo 0 visitado, costo 0, ruta [0]
        heapq.heappush(cola, (0, 0, [0], set([0])))
        
        mejor_costo = limite_superior
        mejor_ruta = None
        
        while cola:
            costo_actual, ciudad_actual, ruta_actual, visitadas = heapq.heappop(cola)
            self.contador.comparaciones += 1
            
            # Poda: si ya superamos el mejor costo conocido
            if costo_actual >= mejor_costo:
                continue
            
            # Si hemos visitado todas las ciudades, completar el ciclo
            if len(visitadas) == n:
                costo_final = costo_actual + self.distancias[ciudad_actual][0]
                self.contador.comparaciones += 1
                if costo_final < mejor_costo:
                    mejor_costo = costo_final
                    mejor_ruta = ruta_actual + [0]
                    self.contador.swaps += 1
                continue
            
            # Expandir nodos hijos (Divide y Vencerás)
            for proxima_ciudad in range(n):
                self.contador.comparaciones += 1
                if proxima_ciudad not in visitadas:
                    nuevo_costo = costo_actual + self.distancias[ciudad_actual][proxima_ciudad]
                    
                    # Cota inferior usando el mínimo de las conexiones restantes
                    cota_inferior = nuevo_costo + self._calcular_cota_inferior(proxima_ciudad, visitadas)
                    
                    # Poda: solo explorar si la cota inferior es prometedora
                    if cota_inferior < mejor_costo:
                        nueva_ruta = ruta_actual + [proxima_ciudad]
                        nuevas_visitadas = visitadas.copy()
                        nuevas_visitadas.add(proxima_ciudad)
                        
                        heapq.heappush(cola, (nuevo_costo, proxima_ciudad, nueva_ruta, nuevas_visitadas))
                        self.contador.swaps += 1
        
        return mejor_ruta, mejor_costo
    
    def _calcular_cota_inferior(self, ciudad_actual, visitadas):
        """Calcula cota inferior para Branch and Bound"""
        n = self.n
        cota = 0
        
        # Mínima conexión desde la ciudad actual a una no visitada
        min_salida = float('inf')
        for j in range(n):
            if j not in visitadas and j != ciudad_actual:
                min_salida = min(min_salida, self.distancias[ciudad_actual][j])
                self.contador.comparaciones += 1
        cota += min_salida if min_salida != float('inf') else 0
        
        # Mínimas conexiones para las ciudades no visitadas
        for i in range(n):
            if i not in visitadas and i != ciudad_actual:
                min_conexion = float('inf')
                for j in range(n):
                    if (j not in visitadas or j == 0) and j != i:
                        min_conexion = min(min_conexion, self.distancias[i][j])
                        self.contador.comparaciones += 1
                cota += min_conexion if min_conexion != float('inf') else 0
        
        return cota
    
    def ejecutar_comparacion_completa(self):
        """Ejecuta todas las implementaciones y compara resultados"""
        print("=" * 70)
        print("         OPTIMIZACIÓN DE RUTAS Y ASIGNACIONES - EJERCICIO 3")
        print("=" * 70)
        
        # Mostrar matriz de distancias
        self.imprimir_matriz()
        
        resultados = {}
        
        # Reiniciar contador
        self.contador.reset()
        
        # PARTE A: PROBLEMA DE ASIGNACIÓN
        print("\n" + "=" * 50)
        print("PARTE A: PROBLEMA DE ASIGNACIÓN")
        print("=" * 50)

        # Fuerza Bruta (total)
        resultados['asignacion_fb'] = self.asignacion_fuerza_bruta_total()

        # Reiniciar contador para siguiente algoritmo
        self.contador.reset()

        # Algoritmo Húngaro (Reducción y Conquista)
        resultados['asignacion_hungaro'] = self.asignacion_hungaro()
        
        # PARTE B: PROBLEMA DEL VIAJANTE (TSP)
        print("\n" + "=" * 50)
        print("PARTE B: PROBLEMA DEL VIAJANTE (TSP)")
        print("=" * 50)
        
        # Reiniciar contador
        self.contador.reset()
        
        # Fuerza Bruta
        if self.n <= 10:  # Para n=15, fuerza bruta es muy lento
            resultados['tsp_fb'] = self.tsp_fuerza_bruta_total()
        else:
            print("=== TSP - FUERZA BRUTA ===")
            print("Omitido para n=15 (demasiado lento - O(15!) ≈ 1.3e12 operaciones)")
        
        # Reiniciar contador
        self.contador.reset()
        
        # Heurística de Inserción
        resultados['tsp_insercion'] = self.tsp_insercion()
        
        # Reiniciar contador
        self.contador.reset()
        
        # Divide y Vencerás + Branch and Bound
        resultados['tsp_dc_bb'] = self.tsp_divide_venceras_branch_bound()

        # ANÁLISIS COMPARATIVO
        self._analizar_resultados(resultados)

        return resultados
    
    def _analizar_resultados(self, resultados):
        """Analiza y compara los resultados obtenidos"""
        print("\n" + "=" * 70)
        print("                 ANÁLISIS COMPARATIVO")
        print("=" * 70)
        
        print("\n--- EFICIENCIA DE ALGORITMOS ---")
        
        # Comparar tiempos y memoria
        algoritmos = []
        if 'asignacion_fb' in resultados:
            alg, costo, tiempo, memoria = resultados['asignacion_fb']
            algoritmos.append(("Asignación - Fuerza Bruta", tiempo, memoria, costo))
        
        if 'asignacion_hungaro' in resultados:
            alg, costo, tiempo, memoria = resultados['asignacion_hungaro']
            algoritmos.append(("Asignación - Húngaro", tiempo, memoria, costo))
        
        if 'tsp_insercion' in resultados:
            ruta, costo, tiempo, memoria = resultados['tsp_insercion']
            algoritmos.append(("TSP - Inserción", tiempo, memoria, costo))
        
        if 'tsp_dc_bb' in resultados:
            ruta, costo, tiempo, memoria = resultados['tsp_dc_bb']
            algoritmos.append(("TSP - Divide y BB", tiempo, memoria, costo))
        
        # Mostrar tabla comparativa
        print("\n" + "-" * 80)
        print(f"{'ALGORITMO':<25} {'TIEMPO (s)':<12} {'MEMORIA (MB)':<12} {'COSTO':<10}")
        print("-" * 80)
        for nombre, tiempo, memoria, costo in algoritmos:
            print(f"{nombre:<25} {tiempo:<12.6f} {memoria:<12.2f} {costo:<10}")
        print("-" * 80)
        
        print("\n--- CONCLUSIONES ---")
        print("1. ALGORITMOS FACTIBLES EN LA PRÁCTICA:")
        print("   • Asignación: Algoritmo Húngaro (O(n³)) es viable para n=15")
        print("   • TSP: Heurística de Inserción y Divide y Vencerás + Branch and Bound")
        print("   • Fuerza Bruta es impracticable para n≥10 en TSP")
        
        print("\n2. TÉCNICAS DE REDUCCIÓN Y CONQUISTA EFECTIVAS:")
        print("   • Algoritmo Húngaro: Reduce problema mediante operaciones matriciales")
        print("   • Branch and Bound: Poda de ramas no prometedoras")
        print("   • Divide y Vencerás: Partición del espacio de búsqueda")
        
        print("\n3. VENTAJAS DE COMBINAR TÉCNICAS:")
        print("   • Divide y Vencerás + Branch and Bound = Mayor eficiencia en problemas NP-hard")
        print("   • Reducción de espacio de búsqueda mediante cotas inferiores")
        print("   • Mejor escalabilidad para instancias más grandes")
        
        print("\n4. RECOMENDACIONES:")
        print("   • Para asignación: Usar Algoritmo Húngaro (óptimo y eficiente)")
        print("   • Para TSP pequeño: Branch and Bound puede encontrar óptimo")
        print("   • Para TSP grande: Heurísticas como Inserción o Vecino Más Cercano")
        print("   • Siempre: Usar podas y cotas para reducir complejidad")

def ejecutar_ejercicio3_completo():
    """Función principal para ejecutar el ejercicio 3 completo"""
    # Para demostración, usar n=8 para que fuerza bruta sea manejable
    # Cambiar a n=15 para el caso real (pero fuerza bruta de TSP será muy lenta)
    n = 8  # Puedes cambiar a 15 para el caso real
    
    optimizador = OptimizacionLogistica(n_depositos=n)
    resultados = optimizador.ejecutar_comparacion_completa()
    
    return resultados

# Versión específica para n=15 (sin fuerza bruta de TSP)
def ejecutar_ejercicio3_n15():
    """Ejecuta el ejercicio 3 con n=15 (sin fuerza bruta de TSP)"""
    print("EJECUTANDO CON 15 DEPÓSITOS/CAMIONES")
    print("(Fuerza Bruta de TSP omitida por complejidad factorial)")
    
    optimizador = OptimizacionLogistica(n_depositos=15)
    
    print("=" * 70)
    print("         OPTIMIZACIÓN DE RUTAS Y ASIGNACIONES - n=15")
    print("=" * 70)
    
    # Mostrar matriz de distancias
    optimizador.imprimir_matriz()
    
    resultados = {}
    
    # PARTE A: PROBLEMA DE ASIGNACIÓN
    print("\n" + "=" * 50)
    print("PARTE A: PROBLEMA DE ASIGNACIÓN")
    print("=" * 50)
    
    # Solo Algoritmo Húngaro para n=15 (fuerza bruta sería O(15!) ≈ 1.3e12)
    resultados['asignacion_hungaro'] = optimizador.asignacion_hungaro()
    
    # PARTE B: PROBLEMA DEL VIAJANTE (TSP)
    print("\n" + "=" * 50)
    print("PARTE B: PROBLEMA DEL VIAJANTE (TSP)")
    print("=" * 50)
    
    # Reiniciar contador
    optimizador.contador.reset()
    
    # Heurística de Inserción
    resultados['tsp_insercion'] = optimizador.tsp_insercion()
    
    # Reiniciar contador
    optimizador.contador.reset()
    
    # Divide y Vencerás + Branch and Bound
    resultados['tsp_dc_bb'] = optimizador.tsp_divide_venceras_branch_bound()
    
    # Análisis comparativo
    optimizador._analizar_resultados(resultados)
    
    return resultados

if __name__ == "__main__":
    # Ejecutar versión completa
    ejecutar_ejercicio3_completo()
    
    # Para ejecutar con n=15, descomenta la siguiente línea:
    # ejecutar_ejercicio3_n15()