# iHealth32 — Monitor de Salud con RNA Embebida · ESP32

Monitor de constantes vitales en tiempo real. Ejecuta una red neuronal artificial directamente en un ESP32 para clasificar el estado clínico del paciente y transmitir los datos por WiFi.

---

## Sensores

| Sensor | Señal |
|--------|-------|
| MAX30100 | BPM y SpO2 (pulsioximetría) |
| MAX30205 | Temperatura corporal (±0,1 °C) |
| AD8232 | ECG a 200 Hz |

---

## Condiciones detectadas

| Clase | Condición |
|-------|-----------|
| 0 | Sano |
| 1 | Fibrilación Auricular |
| 2 | Hipoxia / Apnea |
| 3 | Fiebre / Infección |
| 4 | Taquicardia |
| 5 | Hipotermia |

---

## Red neuronal

```
8 entradas → Dense(64, ReLU) → Dense(32, ReLU) → Dense(6, Softmax)
```

Entradas: `BPM, SpO2, Temp` + 5 features estadísticas de la ventana ECG (1 s / 200 muestras).

Entrenada con datos clínicos reales de PhysioNet: NSRDB, afdb, apnea-ecg y CinC Challenge 2019.
El modelo se exporta a TensorFlow Lite y se embebe en el firmware como array C.

---

## Arquitectura del firmware

- **Core 0** — tarea FreeRTOS: muestrea el ECG a 200 Hz exactos.
- **Core 1** — loop principal: lee sensores I2C, calcula features ECG, ejecuta la RNA y publica resultados.

La alerta se activa solo si la RNA detecta la misma anomalía de forma sostenida con confianza ≥ 70 %.

---

## Instalación

**Requisito:** [PlatformIO](https://platformio.org/).

```bash
git clone https://github.com/yoakeyinn/iHealth32-esp32.git
cd esp32
pio run --target upload
pio device monitor
```


## Uso

El ESP32 crea la red WiFi **iHealth32**. Conecta un cliente WebSocket a `ws://192.168.4.1/ws`.

Paquete JSON emitido cada segundo:

```json
{"ecg": 2048, "bpm": 72.5, "spo2": 98.0, "temp": 36.50, "alerta": 0, "conf": 0.87, "sensores": 1}
```

| Campo | Descripción |
|-------|-------------|
| `alerta` | 0=Sano, 1=FA, 2=Hipoxia, 3=Fiebre, 4=Taquicardia, 5=Hipotermia |
| `conf` | Confianza del modelo (0,0–1,0) |
| `sensores` | 1=todos los sensores válidos, 0=algún sensor falla |

---

> **Aviso:** Prototipo experimental con fines educativos. No certificado como producto sanitario.
