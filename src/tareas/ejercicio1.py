import time
import tracemalloc

def ejercicio1():
    archivo = "clavesb.txt"
    clave_buscada ="reinaldo"

    comparaciones = 0
    encontrada = False

    #Empieza a medir el tiempo y la memora
    tracemalloc.start()
    inicio = time.perf_counter()

    #Leo el archivo y hago la busqueda lineal hasta encontrar la clave, si la encuentro encontrada es true
    with open(archivo, "r", encoding="utf-8") as f:
        for linea in f:
            comparaciones +=1
            clave = linea.strip()
            if clave_buscada in clave:
                encontrada = True
                break

    #Termina el tiempo de ejecucion y se guardan la información necesaria

    fin = time.perf_counter()
    mem_actual, mem_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    #Si se encuentra la clave la muestra si no muestra un mensaje
    if encontrada is True:
        resultado = f"Clave encontrada: {clave}"
    else:
        resultado = "Clave no encontrada"

    return (
        f"{resultado}\n"
        f"Comparaciones realizadas: {comparaciones}\n"
        f"Tiempo de ejecución: {fin - inicio:.6f} segundos\n"
        f"Memoria pico usada: {mem_pico / 1024:.2f} KB" 
    )