// Motor de inferencia de la red neuronal en el ESP32. Antes de ejecutar el modelo, normaliza
// las 8 entradas con Z-score: resta la media y divide por la desviación típica de cada parámetro
// (valores guardados en normalizacion.h). Esto es imprescindible: la red fue entrenada con datos
// normalizados, y sin este paso las predicciones serían incorrectas.
#include "clasificador.h"
#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "modelo_vitales_data.h"
#include "normalizacion.h"

namespace RNA {

    static float confianza_ = 0.0f;
    float confianza() { return confianza_; }

    const int kArenaSize = 24 * 1024;
    uint8_t   tensor_arena[kArenaSize];

    tflite::ErrorReporter*    error_reporter = nullptr;
    const tflite::Model*      model          = nullptr;
    tflite::MicroInterpreter* interpreter    = nullptr;
    TfLiteTensor* input  = nullptr;
    TfLiteTensor* output = nullptr;

    void setup() {
        static tflite::MicroErrorReporter micro_error_reporter;
        error_reporter = &micro_error_reporter;

        model = tflite::GetModel(model_data);

        static tflite::MicroMutableOpResolver<3> resolver;
        resolver.AddFullyConnected();
        resolver.AddRelu();
        resolver.AddSoftmax();

        static tflite::MicroInterpreter static_interpreter(
            model, resolver, tensor_arena, kArenaSize, error_reporter
        );
        interpreter = &static_interpreter;

        if (interpreter->AllocateTensors() != kTfLiteOk) {
            error_reporter->Report("AllocateTensors() fallido - aumentar kArenaSize.");
            while (1) delay(1000);
        }

        input  = interpreter->input(0);
        output = interpreter->output(0);
    }

    int analizar(float bpm, float spo2, float temp,
                 float ecg_range, float ecg_skew, float ecg_kurtosis,
                 float ecg_zero_crossings, float ecg_max_slope) {

        input->data.f[0] = (bpm                - v_mean[0]) / v_scale[0];
        input->data.f[1] = (spo2               - v_mean[1]) / v_scale[1];
        input->data.f[2] = (temp               - v_mean[2]) / v_scale[2];
        input->data.f[3] = (ecg_range          - v_mean[3]) / v_scale[3];
        input->data.f[4] = (ecg_skew           - v_mean[4]) / v_scale[4];
        input->data.f[5] = (ecg_kurtosis       - v_mean[5]) / v_scale[5];
        input->data.f[6] = (ecg_zero_crossings - v_mean[6]) / v_scale[6];
        input->data.f[7] = (ecg_max_slope      - v_mean[7]) / v_scale[7];

        if (interpreter->Invoke() != kTfLiteOk) return -1;

        int   clase    = 0;
        float max_prob = output->data.f[0];
        int   n_clases = output->dims->data[1];
        for (int i = 1; i < n_clases; i++) {
            if (output->data.f[i] > max_prob) {
                max_prob = output->data.f[i];
                clase    = i;
            }
        }
        confianza_ = max_prob;
        return clase;
    }
}
