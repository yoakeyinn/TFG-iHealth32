# patch.py — Corrección automática para TensorFlowLite_ESP32
# ===========================================================
# Este script se ejecuta solo antes de cada compilación gracias a la línea
# extra_scripts = pre:scripts/patch.py en platformio.ini. No hace falta
# llamarlo manualmente.
#
# El problema que resuelve:
#   La librería TFLite_ESP32 tiene un fallo con compiladores GCC modernos (>= 8.x):
#   declara una función interna (operator delete) como privada, pero el compilador
#   necesita poder acceder a ella.
#
# Por qué se parchea en cada compilación y no una sola vez:
#   La librería se descarga en .pio/libdeps/ (carpeta excluida del repositorio).
#   Cada vez que alguien clona el proyecto y compila por primera vez, la librería
#   se descarga sin el parche. El script lo reaplica automáticamente. Si ya está
#   aplicado, detecta que no hay nada que cambiar y no modifica el archivo.

Import("env")
import os

p = os.path.join(
    env.subst("$PROJECT_DIR"),
    ".pio",
    "libdeps",
    env.subst("$PIOENV"),
    "TensorFlowLite_ESP32",
    "src",
    "tensorflow",
    "lite",
    "micro",
    "compatibility.h",
)

OLD = "void operator delete(void* p) {}"
NEW = "public: void operator delete(void* p) {}"

if os.path.isfile(p):
    with open(p, encoding="utf-8") as f:
        s = f.read()
    if NEW not in s and OLD in s:
        with open(p, "w", encoding="utf-8") as f:
            f.write(s.replace(OLD, NEW))
        print("[patch] compatibility.h parcheado: operator delete ahora public")
    elif NEW in s:
        print("[patch] compatibility.h ya estaba parcheado — sin cambios")
    else:
        print(
            "[patch] AVISO: patron no encontrado en compatibility.h — revision manual necesaria"
        )
else:
    print(
        "[patch] compatibility.h no encontrado — patch saltado (ejecuta 'pio run' primero para instalar libdeps)"
    )
