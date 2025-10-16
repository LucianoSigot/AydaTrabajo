import tkinter as tk
from tkinter import messagebox
from tareas.ejercicio1 import ejercicio1
from tareas.ejercicio2 import ejercicio2
from tareas.ejercicio3 import ejercicio3
from tareas.ejercicio4 import ejercicio4
def ejecutar_tarea(tarea):
    if tarea == 1:
        resultado = ejercicio1()
    elif tarea == 2:
        resultado = ejercicio2()
    elif tarea == 3:
        resultado = ejercicio3()
    elif tarea == 4:
        resultado = ejercicio4()
    messagebox.showinfo(f"Resultado Tarea {tarea}", resultado)

    
def menu():
    ventana = tk.Tk()
    ventana.title("Menú de tareas")
    ventana.geometry("300x300")    
    btn1 = tk.Button(ventana, text="Ejecutar Tarea 1", command=lambda:ejecutar_tarea(1))
    btn2 = tk.Button(ventana, text="Ejecutar Tarea 2", command=lambda:ejecutar_tarea(2))
    btn3 = tk.Button(ventana, text="Ejecutar Tarea 3", command=lambda:ejecutar_tarea(3))
    btn4 = tk.Button(ventana, text="Ejecutar Tarea 4", command=lambda:ejecutar_tarea(4))
    btn1.pack(pady=10)
    btn2.pack(pady=10)
    btn3.pack(pady=10)
    btn4.pack(pady=10)
    ventana.mainloop()
