# iHealth32 — Firmware ESP32

Firmware para monitorización de signos vitales (ECG, BPM, SpO₂, temperatura cutánea) con clasificación en tiempo real mediante TFLite Micro. Emite los resultados por WebSocket a la app móvil iHealth32.

---

## Hardware

| Componente | Función | Precio aprox. |
|---|---|---|
| **ESP32 DevKit V1** | MCU principal (WiFi, dual-core) | 8–12 € |
| **AD8232** | Adquisición ECG (incluye electrodos) | 6–10 € |
| **MAX30100** | BPM y SpO₂ por fotopletismografía | 3–7 € |
| **MAX30205** | Temperatura cutánea ±0,1 °C por I2C | 3–6 € |
| **Protoboard + cables** | Montaje sin soldadura | 4–6 € |

**Total: < 42 €**

### Conexiones

```
ESP32 GPIO21 (SDA) ──┬── MAX30100 SDA
                      └── MAX30205 SDA

ESP32 GPIO22 (SCL) ──┬── MAX30100 SCL
                      └── MAX30205 SCL

ESP32 GPIO34 (ADC1) ───── AD8232 OUTPUT
ESP32 GPIO26        ───── AD8232 LO+
ESP32 GPIO27        ───── AD8232 LO−

3.3V / GND ──────────── todos los sensores
```

> El ECG va al GPIO34 (ADC1) porque el ADC2 es incompatible con WiFi activo.

### Electrodos ECG

- **Rojo (RA):** muñeca/hombro derecho
- **Amarillo (LA):** muñeca/hombro izquierdo
- **Verde (RL):** cresta ilíaca derecha o tobillo (referencia)

---

## Arquitectura

### Ejecución dual-core

```
Core 0:  ECG ADC @ 200 Hz  ──►  buffer circular (200 muestras)
Core 1:  loop @ ~1 kHz
           │ Oxi::updateFast()       — vacía FIFO MAX30100 (máx. 160 ms entre llamadas)
           │ Temp::updateIfDue()     — lectura I2C cada 1 s
           │ Oxi::updateVitalsIfDue()— consolida BPM/SpO₂ cada 1 s
           └─► cada 1 s:
               linearizar buffer → calcular_features_ecg()
               RNA::analizar()   → clase 0–5 + confianza
               filtro persistencia → alerta si procede
               wifiWsBroadcast() → JSON al cliente
```

### Validación de sensores

La RNA solo se invoca si todos los sensores están en rango. Si alguno falla, `sensores = 0` en el JSON y no se emiten alertas.

| Sensor | Rango válido |
|---|---|
| BPM | 20–250 lpm |
| SpO₂ | 50–100 % |
| Temperatura | 28–42 °C |
| ECG | > 50 % de muestras ≠ 0 |

### Filtro de persistencia

Una predicción puntual no genera alerta. La misma clase debe mantenerse de forma continua y superar el 70 % de confianza durante:

| Condición | Mínimo |
|---|---|
| Fibrilación Auricular | 10 s |
| Hipoxia / Apnea | 10 s |
| Fiebre / Infección | 20 s |
| Taquicardia | 15 s |
| Hipotermia | 20 s |

---

## Red neuronal

### Entradas (8 variables, normalizadas con Z-score)

| # | Variable | Fuente |
|---|---|---|
| 1 | BPM | MAX30100 |
| 2 | SpO₂ | MAX30100 |
| 3 | Temperatura | MAX30205 |
| 4 | `ecg_range` | buffer ECG |
| 5 | `ecg_skew` | buffer ECG |
| 6 | `ecg_kurtosis` | buffer ECG |
| 7 | `ecg_zero_crossings` | buffer ECG |
| 8 | `ecg_max_slope` | buffer ECG |

### Arquitectura

```
8 entradas → Dense(64, ReLU) → Dense(32, ReLU) → Dense(6, Softmax) → clase + confianza
```

Huella en flash: **13,3 KB**. Tensor arena: **24 KB SRAM**.

### Clases de salida

| Clase | Condición |
|---|---|
| 0 | Sano |
| 1 | Fibrilación Auricular |
| 2 | Hipoxia / Apnea |
| 3 | Fiebre / Infección |
| 4 | Taquicardia |
| 5 | Hipotermia |

### Datos de entrenamiento (PhysioNet)

- **NSRDB** → clase 0
- **MIT-BIH AF Database** → clase 1
- **Apnea-ECG Database** → clase 2
- **CinC Challenge 2019** → clases 3, 4, 5

---

## Instalación

**Requisitos:** VS Code + extensión PlatformIO IDE + cable USB.

```bash
git clone https://github.com/tu-usuario/iHealth32.git
cd iHealth32/esp32

# Compilar y flashear (COM8 por defecto)
pio run --target upload

# Monitor serie (115200 baud)
pio device monitor
```

Si el puerto no se detecta, edita `upload_port` en `platformio.ini`.

### Entornos

| Entorno | Archivo fuente | Sensores |
|---|---|---|
| `produccion` (defecto) | `src/main.cpp` | Todos reales |
| `testing` | `src/main-test.cpp` | BPM real, SpO₂/Temp/ECG simulados |

En `testing`, edita las constantes `TEST_*_TARGET` al inicio de `src/main-test.cpp` para simular una condición concreta (valor negativo = usar sensor real).

```bash
pio run -e testing --target upload
```

---

## Comunicación WiFi / WebSocket

El ESP32 actúa como punto de acceso propio:

- **Red WiFi:** `iHealth32` (sin contraseña)
- **WebSocket:** `ws://192.168.4.1/ws`

Conecta el teléfono a esa red y abre la conexión WebSocket. El ESP32 emite un JSON cada segundo:

```json
{"ecg": 2048, "bpm": 72.5, "spo2": 98.0, "temp": 36.50, "alerta": 0, "conf": 0.87, "sensores": 1}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `ecg` | int 0–4095 | Valor crudo ADC (12 bits) |
| `bpm` | float | Pulsaciones por minuto |
| `spo2` | float | Saturación O₂ (%) |
| `temp` | float | Temperatura cutánea (°C) |
| `alerta` | int 0–5 | 0 = sano/sin alerta; 1–5 = clase detectada |
| `conf` | float 0–1 | Confianza de la predicción |
| `sensores` | 0 o 1 | 1 = todos los sensores válidos |

---

## Regenerar la RNA (pipeline ML)

`include/normalizacion.h` e `include/modelo_vitales_data.h` son **auto-generados**. No los edites manualmente.

```bash
cd esp32/ml

# Entorno virtual Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install tensorflow==2.15.0 scikit-learn pandas numpy wfdb scipy

# Descarga datasets de PhysioNet y construye el dataset (~30–60 min, varios GB)
python preparar_datos.py

# Entrena la RNA, aplica BN folding y regenera los headers C
python entrenar_rna.py
```

Tras regenerar, vuelve a flashear con `pio run --target upload`.

---

## Salida del monitor serie

```
[WiFi] AP 'iHealth32' activo — IP: 192.168.4.1
[OK] Sensores listos: MAX30100 | MAX30205 | AD8232
[OK] Red neuronal lista (8 entradas)

[00:01] Calentando BPM/SpO2... [1s/20s]
...
[00:21]  72.1 BPM  97.5%  25.81C  Sano (91%)
```

El MAX30100 necesita 10–30 s para estabilizarse. Durante ese tiempo, la RNA no se ejecuta. Si un sensor no está en rango:

```
[03:00]  --.- BPM  --.-% --.-C  [SIN: BPM SpO2]
```

---

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| Temperatura siempre 0,00 | MAX30205 no detectado | Verifica dirección I2C `0x4C` con escáner |
| BPM/SpO₂ no se actualizan | `delay()` en el loop bloquea el FIFO | Elimina delays; `Oxi::updateFast()` debe estar en cada iteración |
| ECG en 0 o muy ruidoso | ADC2/WiFi, electrodo suelto | Confirma GPIO34 (ADC1); comprueba electrodos |
| App no conecta | Teléfono no en red "iHealth32" | Conéctate al WiFi "iHealth32" antes de abrir la app |
| Valores NaN en inferencia | BatchNorm sin fusionar | Regenera `include/` con `python entrenar_rna.py` |

---

## Estructura del proyecto

```
esp32/
├── src/
│   ├── main.cpp             # Firmware producción
│   └── main-test.cpp        # Firmware testing (valores simulados)
├── lib/
│   ├── Clasificador/        # Motor TFLite: RNA::setup() / RNA::analizar()
│   ├── ECG_AD8232/          # Muestreo Core 0 @ 200 Hz (FreeRTOS)
│   ├── ECG_Features/        # 5 estadísticos del buffer ECG
│   ├── Oximetro_MAX30100/   # BPM y SpO₂
│   ├── Temp_MAX30205/        # Temperatura I2C
│   └── WiFi_WS/             # AP WiFi + servidor WebSocket
├── include/
│   ├── modelo_vitales_data.h  # Modelo TFLite como array C (AUTO-GENERADO)
│   └── normalizacion.h        # Parámetros Z-score (AUTO-GENERADO)
├── ml/
│   ├── preparar_datos.py    # Descarga PhysioNet, construye dataset
│   └── entrenar_rna.py      # Entrena RNA, BN folding, genera headers
├── scripts/
│   └── patch.py             # Parche pre-build: operator delete en TFLite
└── platformio.ini
```

---

## Aviso de seguridad

Prototipo experimental con fines educativos. **No certificado como dispositivo médico** (ni CE ni FDA). No usar para decisiones clínicas.

Durante las pruebas, alimentar únicamente desde USB de portátil con batería o batería LiPo. **Nunca** conectar a la red eléctrica directamente sin aislamiento certificado (los electrodos hacen contacto directo con la piel).

---

## Glosario

| Término | Definición breve |
|---|---|
| **ADC1 / ADC2** | Dos grupos de entradas analógicas del ESP32. El ADC2 es incompatible con WiFi activo; por eso el ECG va al ADC1 (GPIO34). |
| **BN folding** | Técnica que fusiona las capas BatchNorm en los pesos de la capa anterior antes de exportar a TFLite, evitando NaN en la inferencia embebida. |
| **Buffer circular** | Estructura de datos de tamaño fijo donde las nuevas muestras sobreescriben las más antiguas. El buffer ECG almacena los últimos 200 valores (1 s a 200 Hz). |
| **FreeRTOS** | Sistema operativo de tiempo real del ESP32. `vTaskDelayUntil` garantiza periodos exactos compensando el tiempo de ejecución. |
| **I2C** | Bus serie de dos hilos (SDA datos, SCL reloj) que conecta el ESP32 con MAX30100 y MAX30205 a 400 kHz. |
| **MLP** | Perceptrón Multicapa — arquitectura de red neuronal con capas totalmente conectadas. La usada aquí: 8 → 64 → 32 → 6 neuronas. |
| **Softmax** | Función de activación de la capa de salida que convierte los 6 valores finales en probabilidades que suman 1,0. |
| **TFLite Micro** | Versión de TensorFlow Lite para microcontroladores, sin dependencias de sistema operativo ni asignación dinámica de memoria. |
| **Z-score** | Normalización estadística: `(x − media) / desv_típica`. Se aplica a las 8 entradas para que coincidan con la escala del entrenamiento. |
