// Driver para el sensor de temperatura corporal MAX30205. Se comunica por I2C, el protocolo
// estándar para conectar chips con solo dos cables (SDA y SCL). Tiene una precisión de ±0.1 °C
// y puede distinguir diferencias de 1/256 °C (~0.004 °C), lo que lo hace adecuado para
// detectar fiebre (≥38 °C) o hipotermia (≤35.5 °C) con fiabilidad clínica.
#pragma once
#include <Arduino.h>

#define TEMP_OFFSET_X100  0

namespace Temp {
    bool begin(uint8_t addr = 0x4C);
    void updateIfDue();
    int  tempC_x100();
}
