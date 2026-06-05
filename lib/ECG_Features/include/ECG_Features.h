// Calcula 5 estadísticos que resumen la forma de una ventana ECG de 1 segundo.
// En lugar de enviar las 200 muestras brutas a la red neuronal, las comprimimos en 5 números:
// rango (amplitud total), asimetría, curtosis (forma de los picos), cruces por cero y pendiente máxima.
// Estos 5 valores son suficientes para que la RNA distinga condiciones como FA o taquicardia.
#pragma once

void calcular_features_ecg(int* buf, int n,
                            float& ecg_range, float& ecg_skew, float& ecg_kurtosis,
                            float& ecg_zero_crossings, float& ecg_max_slope);
