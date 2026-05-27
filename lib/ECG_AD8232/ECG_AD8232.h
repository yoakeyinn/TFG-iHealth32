// Driver para el sensor de ECG AD8232. El ECG (electrocardiograma) registra la actividad
// eléctrica del corazón mediante electrodos en la piel. Este módulo crea una tarea que
// corre en el Core 0 del ESP32 (el chip tiene dos núcleos) y toma 200 muestras por segundo.
// Si algún electrodo está despegado (lead-off), la señal se pone a 0 para indicar dato inválido.
#pragma once
#include <Arduino.h>
#include <freertos/task.h>

namespace ECG {
    void begin(int analogPin, int loPlusPin, int loMinusPin);
    void setSamplePeriodUs(uint32_t us);
    void updateIfDue();
    bool hasNewSample();
    int  lastValuePlot();
    TaskHandle_t taskHandle();
}
