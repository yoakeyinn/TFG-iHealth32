"""
entrenar_rna.py
---------------
Entrena la red neuronal que clasifica el estado de salud a partir de los signos
vitales y los features del ECG, y genera los ficheros C++ que necesita el firmware
del ESP32 para hacer inferencia en tiempo real.

Arquitectura de la red: 8 entradas → 64 neuronas → 32 neuronas → 6 salidas
  Entradas (8): BPM, SpO2, Temp, ecg_range, ecg_skew, ecg_kurtosis,
                ecg_zero_crossings, ecg_max_slope
  Capa oculta 1: 64 neuronas con activación ReLU
                 + BatchNormalization (estabiliza el entrenamiento)
                 + Dropout 20% (desactiva neuronas al azar para evitar sobreajuste)
  Capa oculta 2: 32 neuronas ReLU + BatchNormalization + Dropout 20%
  Salida (6):    Softmax → 0=Sano, 1=FA, 2=Hipoxia, 3=Fiebre, 4=Taquicardia, 5=Hipotermia

NOTA sobre features ECG: ecg_std y ecg_rms se eliminaron del vector de entradas
  porque, tras normalizar la ventana ECG a media=0/std=1, estas dos métricas son
  siempre constantes (siempre valen ≈1.0). Al no variar entre muestras, su
  desviación típica es ~0 y provocan NaN al calcular el Z-score del entrenamiento.
  Las 5 features restantes sí varían entre condiciones y no dependen de la escala.

Archivos generados al terminar:
  include/normalizacion.h       — medias y desviaciones para normalizar en el ESP32
  include/modelo_vitales_data.h — modelo TFLite convertido a array C (se incluye en el firmware)
"""

# ── Suprimir mensajes de TensorFlow antes de importarlo ──────────────────────
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore")
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import time
import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dropout

LABEL_NOMBRES = {
    0: "Sano",
    1: "Fibrilacion Auricular",
    2: "Hipoxia/Apnea",
    3: "Fiebre/Infeccion",
    4: "Taquicardia",
    5: "Hipotermia",
}
NUM_CLASES = 6
# ecg_std y ecg_rms se excluyen: tras normalizar el ECG a media=0/std_pop=1
# son constantes en todo el dataset (std≈0), lo que causaría NaN en el Z-score.
FEATURES = [
    "BPM",
    "SpO2",
    "Temp",
    "ecg_range",
    "ecg_skew",
    "ecg_kurtosis",
    "ecg_zero_crossings",
    "ecg_max_slope",
]
carpeta = "datos_clinicos"

# ---------------------------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------------------------
todos = []
for archivo in os.listdir(carpeta):
    if not archivo.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(carpeta, archivo))
    if not df.empty and all(c in df.columns for c in FEATURES + ["Label"]):
        todos.append(df[FEATURES + ["Label"]])

if not todos:
    raise RuntimeError(
        "No se encontraron CSVs válidos. "
        "Asegúrate de haber ejecutado preparar_datos.py con la versión actual."
    )

df_total = pd.concat(todos).dropna().reset_index(drop=True)
df_total = df_total.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Banner de inicio ──────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  ENTRENAMIENTO DE RED NEURONAL — Clasificador Vital (mejorado)")
print("═" * 60)
print(
    f"  Arquitectura : {len(FEATURES)} entradas → 64 → 32 → {NUM_CLASES} salidas  (ecg_std y ecg_rms eliminados por ser constantes)"
)
print("  Parámetros   : ~1000 (cabe en ESP32 con arena ~10KB)")
print("  Optimizador  : Adam   |   Pérdida: Categorical Crossentropy")
print("  Max épocas   : 300    |   EarlyStopping: paciencia 5")
print("═" * 60)

print("\nDistribución de clases:")
for label, nombre in LABEL_NOMBRES.items():
    n = (df_total["Label"] == label).sum()
    barra = "█" * (n // 2000)
    print(f"  [{label}] {nombre:<25} {n:>6} muestras  {barra}")

X = df_total[FEATURES]
y = df_total["Label"].astype(int)

# ---------------------------------------------------------------------------
# NORMALIZACIÓN Z-SCORE
# ---------------------------------------------------------------------------
# La red neuronal aprende mejor cuando todos los datos de entrada están en la
# misma escala. Sin normalizar, el BPM (50–200) dominaría sobre la temperatura
# (36–42) y la red aprendería a ignorar los valores pequeños.
# Z-score: para cada feature, restamos su media y dividimos por su desviación típica.
# Resultado: todas las features quedan centradas en 0 con una variación de ±1.
# Las medias y desviaciones se guardan en normalizacion.h para aplicar exactamente
# la misma transformación en el ESP32 antes de cada inferencia.
medias = X.mean()
escalas = X.std()

print("\nNormalización Z-score:")
for i, feat in enumerate(FEATURES):
    print(f"  {feat:<18}  media={medias.iloc[i]:8.4f}   escala={escalas.iloc[i]:8.4f}")

X_norm = (X - medias) / escalas
y_cat = keras.utils.to_categorical(y, num_classes=NUM_CLASES)

# ---------------------------------------------------------------------------
# BALANCEO DE CLASES
# ---------------------------------------------------------------------------
# Si el dataset tiene muchas más muestras de "Sano" que de "Hipotermia", la red
# aprende a predecir siempre "Sano" y se equivoca con las condiciones raras.
# Para evitarlo, asignamos más peso a las clases con pocos ejemplos: cuando la
# red comete un error en una clase minoritaria, penaliza más durante el
# entrenamiento. Así presta igual atención a todas las condiciones clínicas.
# Fórmula: peso = total_muestras / (n_clases × muestras_de_esa_clase)
total = len(y)
class_weight = {c: total / (NUM_CLASES * (y == c).sum()) for c in range(NUM_CLASES)}
print("\nPesos de clase (balanceo):")
for k, v in class_weight.items():
    print(f"  [{k}] {LABEL_NOMBRES[k]:<25}  peso={v:.4f}")

# ---------------------------------------------------------------------------
# ARQUITECTURA DEL MODELO
# ---------------------------------------------------------------------------
# La red tiene 3 capas totalmente conectadas (Dense). Entre cada capa usamos:
#   BatchNormalization: normaliza los valores intermedios para que el entrenamiento
#     sea más estable (evita que los gradientes se disparen o se anulen).
#   Dropout(0.2): desactiva el 20% de las neuronas al azar en cada paso de
#     entrenamiento. Obliga a la red a aprender patrones robustos en lugar de
#     memorizar los datos de entrenamiento (sobreajuste / overfitting).
# NOTA: las BatchNorm se fusionan con las capas Dense antes de exportar al ESP32,
# porque TFLite_ESP32 v1.0.0 tiene un bug con BatchNorm en inferencia (ver abajo).
modelo = keras.Sequential(
    [
        keras.layers.Input(shape=(len(FEATURES),)),
        BatchNormalization(),
        keras.layers.Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        keras.layers.Dense(NUM_CLASES, activation="softmax"),
    ]
)

modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n" + "─" * 60)
modelo.summary()
print("─" * 60)


# ── Callback de progreso legible ───────────────────────────────────────────────
class ProgresoBonito(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self._t0 = time.time()
        self._mejor = float("inf")
        print("\nIniciando entrenamiento...\n")
        print(
            f"  {'Época':>6}  {'Acc train':>10}  {'Acc val':>10}  {'Loss train':>11}  {'Loss val':>10}  {'Estado':<20}"
        )
        print("  " + "─" * 78)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        elapsed = time.time() - self._t0
        total_ep = self.params["epochs"]
        secs_per_ep = elapsed / (epoch + 1)
        remaining = secs_per_ep * (total_ep - epoch - 1)
        eta_str = f"ETA ~{int(remaining//60)}m{int(remaining%60):02d}s"
        if val_loss < self._mejor - 1e-5:
            self._mejor = val_loss
            estado = "mejora"
        else:
            estado = "sin mejora"
        print(
            f"  {epoch+1:>6}  {acc:>10.4f}  {val_acc:>10.4f}  {loss:>11.4f}  {val_loss:>10.4f}  {estado:<14} {eta_str}"
        )

    def on_train_end(self, logs=None):
        elapsed = time.time() - self._t0
        print("\n" + "═" * 60)
        print(f"  Entrenamiento completado en {int(elapsed//60)}m {int(elapsed%60)}s")
        print(f"  Mejor val_loss: {self._mejor:.4f}")
        print("═" * 60 + "\n")


# EarlyStopping: detiene el entrenamiento automáticamente cuando el modelo lleva
# varias épocas sin mejorar en los datos de validación (los que no ve durante el
# entrenamiento). Con paciencia=5, espera 5 épocas sin mejora antes de parar.
# restore_best_weights=True recupera los pesos de la mejor época al terminar.
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=0,
)

# ReduceLROnPlateau: si el modelo lleva 3 épocas sin mejorar, reduce a la mitad
# la velocidad de aprendizaje (learning rate). Es como dar pasos más pequeños
# cuando te acercas al objetivo para no pasarte de largo.
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0
)

modelo.fit(
    X_norm,
    y_cat,
    epochs=300,  # máximo — early stopping cortará cuando corresponda
    batch_size=128,  # mayor batch para estabilidad
    validation_split=0.2,
    shuffle=False,  # ya mezclado manualmente arriba
    class_weight=class_weight,
    callbacks=[early_stop, ProgresoBonito(), reduce_lr],
    verbose=0,
)

# Guardamos el modelo entrenado completo (con BatchNorm, para referencia/reentrenamiento)
modelo.save("modelo_entrenado.keras")

# ---------------------------------------------------------------------------
# FUSIÓN DE BATCHNORM EN PESOS DENSE
# ---------------------------------------------------------------------------
# Problema: TFLite_ESP32 v1.0.0 produce valores NaN (inválidos) al hacer
# inferencia si el modelo tiene capas BatchNormalization. Para solucionarlo,
# "fusionamos" cada BatchNorm directamente dentro de los pesos de la capa Dense
# adyacente. El modelo resultante es matemáticamente equivalente (mismas
# predicciones) pero sin capas BatchNorm, y el ESP32 lo ejecuta sin errores.
#
# Arquitectura entrenada (con BatchNorm):
#   Input → BN[0] → Dense[0](64) → BN[1] → Dense[1](32) → BN[2] → Dense[2](6)
# Arquitectura exportada al ESP32 (BatchNorm absorbida en los pesos Dense):
#   Input → Dense[0]'(64) → Dense[1]'(32) → Dense[2]'(6)
#
# Por qué funciona: BatchNorm aplica una transformación lineal (escala + desplazamiento)
# y Dense también aplica una transformación lineal (multiplicación de matriz + sesgo).
# Dos transformaciones lineales seguidas siempre se pueden combinar en una sola,
# absorbiendo la BatchNorm en los pesos W y el sesgo b de la capa Dense.


def _bn_params(bn_layer):
    gamma, beta, mean, var = bn_layer.get_weights()
    eps = bn_layer.epsilon
    scale = (gamma / np.sqrt(var + eps)).astype(np.float32)
    shift = (beta - mean * scale).astype(np.float32)
    return scale, shift


def _fold_bn(dense_layer, scale, shift):
    W, b = [w.astype(np.float32) for w in dense_layer.get_weights()]
    return (W * scale[:, np.newaxis]), (shift @ W + b)


bn_layers = [l for l in modelo.layers if isinstance(l, keras.layers.BatchNormalization)]
dense_layers = [l for l in modelo.layers if isinstance(l, keras.layers.Dense)]
assert (
    len(bn_layers) == 3 and len(dense_layers) == 3
), f"Se esperaban 3 BN y 3 Dense, se encontraron {len(bn_layers)} BN y {len(dense_layers)} Dense"

s0, h0 = _bn_params(bn_layers[0])
W0_new, b0_new = _fold_bn(dense_layers[0], s0, h0)
s1, h1 = _bn_params(bn_layers[1])
W1_new, b1_new = _fold_bn(dense_layers[1], s1, h1)
s2, h2 = _bn_params(bn_layers[2])
W2_new, b2_new = _fold_bn(dense_layers[2], s2, h2)

modelo_fused = keras.Sequential(
    [
        keras.layers.Input(shape=(len(FEATURES),), name="input"),
        keras.layers.Dense(64, activation="relu", name="dense_0"),
        keras.layers.Dense(32, activation="relu", name="dense_1"),
        keras.layers.Dense(NUM_CLASES, activation="softmax", name="dense_out"),
    ]
)
modelo_fused.layers[0].set_weights([W0_new, b0_new])
modelo_fused.layers[1].set_weights([W1_new, b1_new])
modelo_fused.layers[2].set_weights([W2_new, b2_new])

print("\nValidando fusión BatchNorm (tolerancia 1e-3):")
x_val = X_norm.sample(min(300, len(X_norm)), random_state=0).values.astype(np.float32)
pred_orig = modelo.predict(x_val, verbose=0)
pred_fused = modelo_fused.predict(x_val, verbose=0)
max_diff = float(np.max(np.abs(pred_orig - pred_fused)))
estado_bn = "OK" if max_diff < 1e-3 else "ADVERTENCIA — revisar fusión"
print(f"  max_diff orig vs fusionado: {max_diff:.2e}  [{estado_bn}]")
if max_diff >= 1e-3:
    raise RuntimeError("La fusión BatchNorm produjo diferencias demasiado grandes.")

# ---------------------------------------------------------------------------
# GENERACIÓN DE ARCHIVOS C++ PARA EL FIRMWARE
# ---------------------------------------------------------------------------
# Convertimos el modelo entrenado a dos ficheros que el firmware del ESP32 incluye
# directamente al compilar:
#   normalizacion.h:       las medias y desviaciones del Z-score (8 números cada una)
#   modelo_vitales_data.h: el modelo TFLite entero convertido a un array de bytes en C
# Estos ficheros se escriben en ../include/ y PlatformIO los compila junto con el firmware.
# Después de regenerarlos, basta con hacer "pio run --target upload" para subir el
# nuevo modelo al ESP32 sin tocar nada más.
os.makedirs("../include", exist_ok=True)

with open("../include/normalizacion.h", "w", encoding="utf-8") as f:
    f.write("// Auto-generado por entrenar_rna.py -- no editar manualmente\n")
    f.write(f"// Orden ({len(FEATURES)} features): {', '.join(FEATURES)}\n")
    f.write("#pragma once\n\n")
    n = len(medias)
    vals_mean = ", ".join(f"{medias.iloc[i]:.4f}f" for i in range(n))
    vals_scale = ", ".join(f"{escalas.iloc[i]:.4f}f" for i in range(n))
    f.write(f"static const float v_mean[]  = {{{vals_mean}}};\n")
    f.write(f"static const float v_scale[] = {{{vals_scale}}};\n")
print("  OK: ../include/normalizacion.h generado.")

print("  Convirtiendo modelo fusionado a TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(modelo_fused)
tflite_bytes = converter.convert()
print(f"  Tamaño TFLite: {len(tflite_bytes)} bytes")

_C_HEADER = """\
// Auto-generado por ml/entrenar_rna.py — no editar manualmente
// Modelo sin BatchNormalization (fusionada en pesos Dense).
// Compatible con TensorFlowLite_ESP32 v1.0.0 — sin NaN en inferencia.
#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif

const unsigned char model_data[] DATA_ALIGN_ATTRIBUTE = {{
  {hex_vals}
}};
const int model_data_len = {model_len};
"""

hex_vals = ", ".join(f"0x{b:02x}" for b in tflite_bytes)
with open("../include/modelo_vitales_data.h", "w", encoding="utf-8") as f:
    f.write(_C_HEADER.format(hex_vals=hex_vals, model_len=len(tflite_bytes)))

print("  OK: ../include/modelo_vitales_data.h generado (TFLite sin BatchNorm).")
print("\n¡Todo listo! Ahora puedes compilar el firmware del ESP32.\n")
