from cx_Freeze import setup, Executable

setup(
    name="AYDA_Practico",
    version="1.0",
    description="TP AYDA - Practico 5",
    executables=[Executable("src/menu.py")],
)
