import tkinter as tk
from tkinter import messagebox
import os
import sys
import traceback
import subprocess
import threading
import runpy

# --- CONFIGURACIÓN DE RUTAS ---
# Al ejecutar con PyInstaller en modo --onefile los datos añadidos se extraen
# a un directorio temporal accesible vía sys._MEIPASS. En ejecución normal usamos
# el directorio del script.
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# La carpeta con los ejercicios en el repo se llama 'Ejercicios'
EJERCICIOS_DIR = os.path.join(BASE_DIR, "Ejercicios")

# --- CREACIÓN DE LA VENTANA PRINCIPAL ---
ventana = tk.Tk()
ventana.title("Menú Principal - Ejercicios de Algoritmos")
ventana.geometry("500x500")
ventana.config(bg="#f4f4f4")

# --- FUNCIÓN PARA CAMBIAR DE PANTALLA ---
def mostrar_menu_principal():
    limpiar_ventana()
    titulo = tk.Label(ventana, text="Seleccione un ejercicio para ejecutar",
                      font=("Arial", 14, "bold"), bg="#f4f4f4")
    titulo.pack(pady=20)

    for i in range(1, 6):
        boton = tk.Button(ventana, text=f"Ejercicio {i}",
                          font=("Arial", 12), bg="#2196F3", fg="white",
                          width=20, height=2, command=lambda n=i: abrir_ejercicio(n))
        boton.pack(pady=10)

    # Ícono de puerta para salir
    icono_path = os.path.join(BASE_DIR, "icono_puerta.png")
    if os.path.exists(icono_path):
        puerta_img = tk.PhotoImage(file=icono_path)
        boton_salir = tk.Button(ventana, image=puerta_img, text="  Salir",
                                compound="left", font=("Arial", 12, "bold"),
                                bg="#E53935", fg="white", width=120, height=50,
                                command=ventana.destroy)
        boton_salir.image = puerta_img
    else:
        boton_salir = tk.Button(ventana, text="Salir", font=("Arial", 12, "bold"),
                                bg="#E53935", fg="white", width=15, height=2,
                                command=ventana.destroy)
    boton_salir.pack(pady=30)

# --- FUNCIÓN PARA LIMPIAR LA VENTANA ---
def limpiar_ventana():
    for widget in ventana.winfo_children():
        widget.destroy()

# --- FUNCIÓN PARA ABRIR UN EJERCICIO ---
def abrir_ejercicio(numero):
    limpiar_ventana()
    titulo = tk.Label(ventana, text=f"Ejercicio {numero}",
                      font=("Arial", 16, "bold"), bg="#f4f4f4")
    titulo.pack(pady=20)

    ruta = os.path.join(EJERCICIOS_DIR, f"ejercicio{numero}.py")
    if not os.path.exists(ruta):
        mensaje = tk.Label(ventana, text="No se encontró el archivo del ejercicio.",
                           font=("Arial", 12), bg="#f4f4f4", fg="red")
        mensaje.pack(pady=10)
    else:
        # Ejecutar el ejercicio en un proceso separado para no bloquear la GUI.
        # Usamos sys.executable y un argumento --run-exercise para que funcione
        # tanto en desarrollo (python) como en el bundle (exe).
        def lanzar_proceso(ruta_script):
            try:
                # Si estamos en bundle (frozen), sys.executable ya apunta al exe
                if getattr(sys, 'frozen', False):
                    cmd = [sys.executable, '--run-exercise', ruta_script]
                else:
                    # En desarrollo, invocamos el intérprete pasando el script main_menu.py
                    script_path = os.path.abspath(__file__)
                    cmd = [sys.executable, script_path, '--run-exercise', ruta_script]

                # Usar -u para modo sin buffer en Python y bufsize=1 para line-buffering
                if cmd and cmd[0].endswith('python.exe') and '-u' not in cmd:
                    # insertar -u tras el ejecutable
                    cmd.insert(1, '-u')
                proc = subprocess.Popen(cmd,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)

                # Crear ventana de salida
                out_win = tk.Toplevel(ventana)
                out_win.title(f"Salida - {os.path.basename(ruta_script)}")
                out_win.geometry("700x400")

                text_widget = tk.Text(out_win, wrap='none')
                text_widget.pack(fill='both', expand=True)

                # Botón para terminar proceso
                def terminar():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                btn_stop = tk.Button(out_win, text="Detener", command=terminar, bg="#E53935", fg="white")
                btn_stop.pack(pady=4)

                # Hilos para leer stdout/stderr
                def reader(stream, tag=''):
                    for line in iter(stream.readline, ''):
                        # insertar en el widget de forma segura
                        def ins(l=line):
                            text_widget.insert('end', l)
                            text_widget.see('end')
                        text_widget.after(0, ins)
                    stream.close()

                t_out = threading.Thread(target=reader, args=(proc.stdout,))
                t_err = threading.Thread(target=reader, args=(proc.stderr,))
                t_out.daemon = True
                t_err.daemon = True
                t_out.start()
                t_err.start()

            except Exception as e:
                messagebox.showerror("Error", f"No se pudo lanzar el proceso: {e}")

        lanzar_proceso(ruta)

    # Botón para volver al menú principal
    volver_btn = tk.Button(ventana, text="⏪ Volver al Menú",
                           font=("Arial", 12, "bold"), bg="#43A047", fg="white",
                           width=18, height=2, command=mostrar_menu_principal)
    volver_btn.pack(pady=20)

# --- INICIO DEL PROGRAMA ---
def _run_exercise_child_mode():
    # Modo invocado como: exe_or_python --run-exercise <ruta>
    try:
        idx = sys.argv.index('--run-exercise')
    except ValueError:
        return False
    if idx + 1 >= len(sys.argv):
        return False
    ruta = sys.argv[idx + 1]
    # Si la ruta es relativa, hacerla absoluta respecto a BASE_DIR
    if not os.path.isabs(ruta):
        ruta = os.path.join(BASE_DIR, ruta)

    # Asegurar que la carpeta Ejercicios esté en sys.path para imports como
    # `from utils.util import ...` y cambiar cwd para que los scripts que usan
    # archivos relativos funcionen.
    inserted = False
    prev_cwd = os.getcwd()
    try:
        if EJERCICIOS_DIR not in sys.path:
            sys.path.insert(0, EJERCICIOS_DIR)
            inserted = True
        os.chdir(EJERCICIOS_DIR)
        # Preparar sys.argv para el script hijo: [ruta, <args...>]
        child_args = sys.argv[idx+2:]
        sys_argv_backup = sys.argv
        sys.argv = [ruta] + child_args
        try:
            runpy.run_path(ruta, run_name='__main__')
        finally:
            sys.argv = sys_argv_backup
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # Restaurar cwd y sys.path
        try:
            os.chdir(prev_cwd)
        except Exception:
            pass
        if inserted and sys.path and sys.path[0] == EJERCICIOS_DIR:
            sys.path.pop(0)
    return True


if __name__ == '__main__' and _run_exercise_child_mode():
    # Si se invocó en modo hijo para ejecutar un ejercicio, terminamos aquí.
    sys.exit(0)

mostrar_menu_principal()
ventana.mainloop()