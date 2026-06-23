// Implementación del driver AD8232. Usa FreeRTOS (el sistema operativo en tiempo real del ESP32)
// para crear una tarea independiente en el Core 0 que lee el ADC exactamente cada 5 ms.
// El ADC convierte la tensión analógica del electrodo (0–3.3 V) a un número entero (0–4095).
// Si LO+ o LO- están a HIGH, el electrodo está suelto y se guarda 0 como valor de señal.
#include "ECG_AD8232.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

static int aPin_ = 34;
static int loP_  = 26;
static int loM_  = 27;

static uint32_t      periodUs_  = 4000;
static volatile bool newSample_ = false;
static volatile int  lastPlot_  = 0;

TaskHandle_t ecgTaskHandle = NULL;

static void ecgTask(void* pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(periodUs_ / 1000);

    while (1) {
        vTaskDelayUntil(&xLastWakeTime, xFrequency);
        bool leadOff = (digitalRead(loP_) == HIGH) || (digitalRead(loM_) == HIGH);
        lastPlot_  = leadOff ? 0 : analogRead(aPin_);
        newSample_ = true;
    }
}

namespace ECG {

void begin(int analogPin, int loPlusPin, int loMinusPin) {
    aPin_ = analogPin;
    loP_  = loPlusPin;
    loM_  = loMinusPin;

    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);

    pinMode(loP_, INPUT);
    pinMode(loM_, INPUT);

    xTaskCreatePinnedToCore(
        ecgTask, "Tarea_ECG", 2048, NULL,
        configMAX_PRIORITIES - 1, &ecgTaskHandle, 0
    );
}

void setSamplePeriodUs(uint32_t us) { periodUs_ = us; }
TaskHandle_t taskHandle()           { return ecgTaskHandle; }
void updateIfDue()                  {}

bool hasNewSample() {
    if (newSample_) { newSample_ = false; return true; }
    return false;
}

int lastValuePlot() { return lastPlot_; }

} // namespace ECG
