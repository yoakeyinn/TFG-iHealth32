// Implementación de los features ECG. Antes de calcular los estadísticos, normaliza la
// ventana restando la media y dividiendo por la desviación típica (media=0, std=1).
// Esto hace que los resultados sean comparables entre pacientes sin importar la ganancia
// del amplificador ni el nivel de contacto del electrodo con la piel.
#include "ecg_features.h"
#include <math.h>

void calcular_features_ecg(int* buf, int n,
                            float& ecg_range, float& ecg_skew, float& ecg_kurtosis,
                            float& ecg_zero_crossings, float& ecg_max_slope) {

    ecg_range = ecg_skew = ecg_kurtosis = ecg_zero_crossings = ecg_max_slope = 0.0f;
    if (n == 0) return;

    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += buf[i];
    double mean = sum / n;

    double var_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)buf[i] - mean;
        var_sum += d * d;
    }
    double pop_std = sqrt(var_sum / n);

    if (pop_std < 1e-6) return;

    float s3 = 0.0f, s4 = 0.0f;
    float xn_prev   = (float)(((double)buf[0] - mean) / pop_std);
    float xn_min    = xn_prev;
    float xn_max    = xn_prev;
    int   zc        = 0;
    float max_slope = 0.0f;

    for (int i = 0; i < n; i++) {
        float xnf = (float)(((double)buf[i] - mean) / pop_std);

        if (xnf < xn_min) xn_min = xnf;
        if (xnf > xn_max) xn_max = xnf;

        s3 += xnf * xnf * xnf;
        s4 += xnf * xnf * xnf * xnf;

        if (i > 0) {
            if ((xn_prev >= 0.0f && xnf < 0.0f) || (xn_prev < 0.0f && xnf >= 0.0f))
                zc++;
            float slope = fabsf(xnf - xn_prev);
            if (slope > max_slope) max_slope = slope;
        }
        xn_prev = xnf;
    }

    ecg_range          = xn_max - xn_min;
    ecg_skew           = s3 / (float)n;
    ecg_kurtosis       = s4 / (float)n - 3.0f;
    ecg_zero_crossings = (float)zc;
    ecg_max_slope      = max_slope;
}
