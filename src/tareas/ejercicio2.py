import time
import tracemalloc
import random

# Genera 10 millones de numeros enteros en [0, 10_000_000)
def generar_datos(n=10_000_000):
    return [random.randint(0, int(1e7)) for _ in range(n)]


#Algoritmo Merge sort, divide y luego ordena
def merge_sort (arr):
    comparaciones = 0
    swaps = 0
        
    #Funcion de merge para ordenar los datos
    def merge(izquierda, derecha):
        nonlocal comparaciones,swaps
        resultado = []
        i = j = 0
        while i < len(izquierda) and j < len(derecha):
            comparaciones +=1
            if izquierda[i] <= derecha[j]:
                resultado.append(izquierda[i])
                i+=1
            else:
                resultado.append(derecha[j])
                j+=1
                swaps += 1
        resultado.extend(izquierda[i:])
        resultado.extend(derecha[j:])
        return resultado
        
    #Funcion para dividir la lista en partes pequenas
    def dividir(lista):
        if len(lista)<= 1:
            return lista
            
        #Divido la lista en 2
        medio = len(lista)//2
        izquierda = dividir(lista[:medio])
        derecha = dividir(lista[medio:])
            
        #Combino las 2 mitades
        return merge(izquierda, derecha)
    ordenado = dividir(arr)
    return ordenado, comparaciones, swaps

#Funcion que encuentra al elemento en la posicion k buscando un pivote final y colocando todos los elementos menores a la izquierda y mayores a la derecha
def quick_select_lomuto(arr, k):
    comparaciones = 0
    swaps = 0
    #Ordeno alrededor del pivote 
    def particion(chico, grande):
        nonlocal comparaciones, swaps
        pivot = arr[grande]
        i = chico
        for j in range (chico, grande):
            comparaciones +=1
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                swaps +=1
                i +=1
        arr[i], arr[grande] = arr[grande], arr[i]
        return i
    #Devuelvo el elemento que me interesa 
    def select(chico, grande, k):
        if chico == grande:
            return arr[chico]
        p = particion(chico, grande)
        if k == p:
            return arr[k]
        elif k < p:
            return select(chico, p - 1, k)
        else:
            return select(p + 1, grande, k)
    resultado = select(0, len(arr) - 1, k)
    return resultado, comparaciones, swaps
#Funcion parecida a la anterior pero con la diferencia que se busca por 2 indices
def quick_select_hoare(arr, k):
    comparaciones = 0
    swaps = 0
    def particion (chico, grande):
        nonlocal comparaciones, swaps
        pivot = arr[(chico + grande) //2]
        i,j = chico, grande
        while True:
            while arr[i] < pivot:
                comparaciones += 1
                i += 1
            while arr[j] > pivot:
                comparaciones += 1
                j -= 1
            if i >= j:
                return j
            arr[i], arr[j] = arr[j], arr[i]
            swaps += 1
            i += 1
            j -= 1
            
    def select (chico,grande,k):
        if chico == grande:
            return arr[chico]
        p = particion(chico,grande)
        if k <= p:
            return select(chico,p,k)
        else:
            return select(p+1,grande,k)
    resultado = select(0, len(arr) - 1, k)
    return resultado, comparaciones, swaps

def bubble_sort(arr):
    comparaciones = 0
    swaps = 0
    n = len(arr)
    lista = arr.copy()
    for i in range(n):
        for j in range(0, n - i - 1):
            comparaciones += 1
            if lista[j] > lista[j + 1]:
                lista[j], lista[j + 1] = lista[j + 1], lista[j]
                swaps += 1
    return lista, comparaciones, swaps

def ejercicio2():
    datos= generar_datos(1_000_000)
    n = len(datos)
    k= n//2 #posicion de la mediana
    resultados = []
    def medir(func, *args):
        tracemalloc.start()
        inicio = time.perf_counter()
        resultado = func(*args)
        fin = time.perf_counter()
        mem_actual, mem_pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return resultado, fin - inicio, mem_pico / 1024
    
    # A) Merge Sort
    (ordenado, comps, swaps), tiempo, mem = medir(merge_sort, datos.copy())
    mediana_merge = ordenado[k]
    resultados.append(f"(A) Merge Sort → Mediana: {mediana_merge} \nTiempo: {tiempo:.4f}s\nComparaciones: {comps}\nSwaps: {swaps}\nMemoria pico: {mem:.2f} KB\n")
    # --- QuickSelect Lomuto ---
    (mediana_qs, comps, swaps), tiempo, mem = medir(quick_select_lomuto, datos.copy(), k)
    resultados.append(f"(B) QuickSelect Lomuto → Mediana: {mediana_qs}\nTiempo: {tiempo:.4f}s\nComparaciones: {comps}\nSwaps: {swaps}\nMemoria pico: {mem:.2f} KB\n")
     # QuickSelect Hoare
    (mediana_hoare, comps, swaps), tiempo, mem = medir(quick_select_hoare, datos.copy(), k)
    resultados.append(f"(C) QuickSelect Hoare → Mediana: {mediana_hoare}\nTiempo: {tiempo:.4f}s\nComparaciones: {comps}\nSwaps: {swaps}\nMemoria pico: {mem:.2f} KB\n")

    # Bubble Sort (solo con subarreglo pequeño para que no tarde)
    subdatos = datos[:1000]  # reducir tamaño
    (ordenado_bubble, comps, swaps), tiempo, mem = medir(bubble_sort, subdatos)
    mediana_bubble = ordenado_bubble[len(ordenado_bubble)//2]
    resultados.append(f"(D) Bubble Sort → Mediana: {mediana_bubble}\nTiempo: {tiempo:.4f}s\nComparaciones: {comps}\nSwaps: {swaps}\nMemoria pico: {mem:.2f} KB\n")
    return "\n".join(resultados)
