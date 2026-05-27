/*
 * normalizacion.h — Parámetros Z-score del modelo RNA
 * =====================================================
 * AUTO-GENERADO por entrenar_rna.py — NO EDITAR MANUALMENTE.
 * Regenerar ejecutando el paso 3 del pipeline ML (cd ml && python entrenar_rna.py).
 *
 * La red neuronal espera que sus 8 entradas estén normalizadas con Z-score antes
 * de la inferencia:
 *
 *     entrada_normalizada[i] = (valor[i] - v_mean[i]) / v_scale[i]
 *
 * Esta transformación centra cada variable en 0 y la escala a desviación típica 1,
 * lo que acelera la convergencia durante el entrenamiento y estabiliza la inferencia.
 * Los parámetros v_mean y v_scale se calculan sobre el dataset de entrenamiento
 * completo (≈6.000 muestras de cuatro bases de datos PhysioNet) y son los mismos
 * que aplica scikit-learn en el pipeline de entrenamiento.
 *
 * Orden de los 8 features (índices 0–7):
 *   [0] BPM                 — Frecuencia cardiaca en lpm
 *   [1] SpO2                — Saturación de oxígeno en sangre (%)
 *   [2] Temp                — Temperatura cutánea (°C)
 *   [3] ecg_range           — Rango de la señal ECG normalizada
 *   [4] ecg_skew            — Asimetría (momento de 3er orden)
 *   [5] ecg_kurtosis        — Curtosis en exceso (momento de 4o orden − 3)
 *   [6] ecg_zero_crossings  — Número de cruces por cero en la ventana
 *   [7] ecg_max_slope       — Máxima pendiente entre muestras consecutivas
 *
 * Este archivo es incluido por lib/Clasificador/clasificador.cpp a través
 * de la ruta de include configurada en platformio.ini (-I${PROJECT_INCLUDE_DIR}).
 */
#pragma once

// Medias del dataset de entrenamiento por feature (misma escala que los datos de entrada)
static const float v_mean[]  = {83.5005f, 93.9700f, 36.5646f, 7.7135f, 1.6458f, 13.2696f, 13.9017f, 2.1102f};

// Desviaciones típicas del dataset de entrenamiento por feature (denominador del Z-score)
static const float v_scale[] = {22.2375f, 4.7595f, 0.5537f, 1.6552f, 2.5642f, 10.4325f, 10.3731f, 0.7200f};
