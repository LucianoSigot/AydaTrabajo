import tkinter as tk
from tkinter import messagebox
from tareas.ejercicio1 import ejercicio1
from tareas.ejercicio2 import ejercicio2
from tareas.ejercicio3 import ejercicio3
from tareas.ejercicio4 import ejercicio4
import threading

def ejecutar_tarea(tarea, ventana, label_estado):
    label_estado.config(text="⏳ Ejecutando, por favor espere...", fg="blue")
    ventana.update_idletasks()

    def tarea_en_hilo():
        try:
            if tarea == 1:
                resultado = ejercicio1()
            elif tarea == 2:
                resultado = ejercicio2()
            elif tarea == 3:
                resultado = ejercicio3()
            elif tarea == 4:
                resultado = ejercicio4()
            messagebox.showinfo(f"Resultado Tarea {tarea}", resultado)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            label_estado.config(text="✅ Tarea finalizada", fg="green")

    # se ejecuta en otro hilo para no congelar la interfaz
    threading.Thread(target=tarea_en_hilo).start()


def menu():
    ventana = tk.Tk()
    ventana.title("Menú de tareas")
    ventana.geometry("300x350")

    label_estado = tk.Label(ventana, text="", fg="blue")
    label_estado.pack(pady=10)

    btn1 = tk.Button(ventana, text="Ejecutar Tarea 1", command=lambda: ejecutar_tarea(1, ventana, label_estado))
    btn2 = tk.Button(ventana, text="Ejecutar Tarea 2", command=lambda: ejecutar_tarea(2, ventana, label_estado))
    btn3 = tk.Button(ventana, text="Ejecutar Tarea 3", command=lambda: ejecutar_tarea(3, ventana, label_estado))
    btn4 = tk.Button(ventana, text="Ejecutar Tarea 4", command=lambda: ejecutar_tarea(4, ventana, label_estado))

    btn1.pack(pady=10)
    btn2.pack(pady=10)
    btn3.pack(pady=10)
    btn4.pack(pady=10)

    ventana.mainloop()
