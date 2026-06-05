// Driver para el pulsioxímetro MAX30100. Mide la frecuencia cardíaca (BPM, latidos por minuto)
// y la saturación de oxígeno en sangre (SpO2, %) mediante pulsioximetría: ilumina el dedo con
// LEDs y mide cuánta luz absorbe la sangre en cada pulso (técnica PPG).
// IMPORTANTE: hay que llamar a updateFast() en cada ciclo del loop o el sensor perderá datos.
#pragma once
#include <Arduino.h>

#define SPO2_OFFSET_PCT   3
#define BPM_OFFSET_X10    0

namespace Oxi {
    bool begin();
    void updateFast();
    void updateVitalsIfDue(uint32_t ms);
    int  bpm_x10();
    int  spo2_x10();
}
