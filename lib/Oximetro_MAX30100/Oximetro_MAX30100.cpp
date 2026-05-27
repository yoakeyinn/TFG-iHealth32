// Implementación del driver MAX30100. Envuelve la librería Arduino-MAX30100 con una interfaz
// más sencilla: updateFast() drena el buffer interno del sensor en cada ciclo del loop, y
// updateVitalsIfDue() lee BPM y SpO2 cuando ha pasado el intervalo configurado.
// Los valores se guardan como enteros ×10 (ej: 753 = 75.3 lpm) para evitar decimales en el driver.
#include "Oximetro_MAX30100.h"
#include "MAX30100_PulseOximeter.h"

static PulseOximeter pox;
static uint32_t tNext_    = 0;
static int      bpm_x10_  = 0;
static int      spo2_x10_ = 0;

static void onBeatDetected() {}

namespace Oxi {

bool begin() {
    if (!pox.begin()) return false;
    pox.setIRLedCurrent(MAX30100_LED_CURR_50MA);
    pox.setOnBeatDetectedCallback(onBeatDetected);
    tNext_ = millis();
    return true;
}

void updateFast() {
    pox.update();
}

void updateVitalsIfDue(uint32_t ms) {
    uint32_t now = millis();
    if ((int32_t)(now - tNext_) < 0) return;
    tNext_ = now + ms;

    float bpm = pox.getHeartRate();
    float sp  = pox.getSpO2();

    if (bpm > 0 && bpm < 240)  bpm_x10_  = (int)(bpm * 10.0f) + BPM_OFFSET_X10;
    if (sp  > 0 && sp  <= 100) spo2_x10_ = min((int)((sp + SPO2_OFFSET_PCT) * 10.0f), 1000);
}

int bpm_x10()  { return bpm_x10_;  }
int spo2_x10() { return spo2_x10_; }

} // namespace Oxi
