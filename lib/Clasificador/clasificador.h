// Interfaz de la red neuronal artificial (RNA) que clasifica el estado de salud del paciente.
// Usa TensorFlow Lite Micro (TFLite), la versión de TensorFlow diseñada para microcontroladores.
// Recibe 8 parámetros vitales (BPM, SpO2, Temp y 5 features del ECG) y devuelve una de 6
// condiciones: Sano, Fibrilación Auricular, Hipoxia, Fiebre, Taquicardia o Hipotermia.
#pragma once

namespace RNA {
    void  setup();
    int   analizar(float bpm, float spo2, float temp,
                   float ecg_range, float ecg_skew, float ecg_kurtosis,
                   float ecg_zero_crossings, float ecg_max_slope);
    float confianza();
}
