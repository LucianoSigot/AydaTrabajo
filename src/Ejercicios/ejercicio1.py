import time
import os
import argparse

try:
    import psutil  
except Exception:
    psutil = None


def buscar_clave(clave_buscada: str):
    """Busca clave_buscada en el archivo fijo 'clavesb.txt' dentro de la carpeta del script.

    No se permite cambiar la ruta del archivo: siempre se usa 'clavesb.txt'.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(base, 'clavesb.txt')

    comparaciones = 0
    swaps = 0
    encontrada = None

    inicio_tiempo = time.perf_counter()

    proceso = None
    memoria_inicio = None
    if psutil:
        try:
            proceso = psutil.Process(os.getpid())
            memoria_inicio = proceso.memory_info().rss
        except Exception:
            proceso = None
            memoria_inicio = None
    try:
        with open(ruta, 'r', encoding='utf-8') as archivo:
            for linea in archivo:
                comparaciones += 1
                if clave_buscada in linea.strip():
                    encontrada = linea.strip()
                    break
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo 'clavesb.txt' en la carpeta del script ({ruta})")
        return

    fin_tiempo = time.perf_counter()
    memoria_fin = None
    if proceso:
        try:
            memoria_fin = proceso.memory_info().rss
        except Exception:
            memoria_fin = None

    print("\n🔍 Resultados de búsqueda")
    print(f"Clave buscada: '{clave_buscada}'")
    if encontrada:
        print(f"✅ Clave encontrada: {encontrada}")
    else:
        print("⚠️ Clave no encontrada.")

    print(f"\n⏱ Tiempo de ejecución: {fin_tiempo - inicio_tiempo:.6f} segundos")
    print(f"🔢 Comparaciones realizadas: {comparaciones}")
    print(f"🔁 Swaps (intercambios): {swaps}")
    if memoria_inicio is not None and memoria_fin is not None:
        print(f"💾 Memoria utilizada: {(memoria_fin - memoria_inicio) / 1024:.2f} KB")
    else:
        print("💾 Memoria utilizada: (psutil no disponible o no pudo leer memoria)")


def main():
    parser = argparse.ArgumentParser(description='Buscar una clave en el archivo fijo clavesb.txt')
    parser.add_argument('-k', '--key', default='reinaldo', help='clave a buscar (por defecto: reinaldo)')
    args = parser.parse_args()

    buscar_clave(args.key)


if __name__ == '__main__':
    main()
