// Implementación del driver MAX30205. Lee el registro de temperatura por I2C una vez por
// segundo para no saturar el bus de comunicación. El sensor devuelve 2 bytes en complemento
// a dos con resolución de 1/256 °C, que se convierten a entero en centésimas de grado
// (ej: 3650 = 36.50 °C) para trabajar sin decimales en la capa de driver.
#include "Temp_MAX30205.h"
#include <Wire.h>

static uint8_t  addr_     = 0x4C;
static uint32_t tNext_    = 0;
static int      lastTemp_ = 0;

namespace Temp {

bool begin(uint8_t addr) {
    addr_ = addr;
    Wire.beginTransmission(addr_);
    if (Wire.endTransmission() != 0) return false;

    Wire.beginTransmission(addr_);
    Wire.write(0x01);
    Wire.write(0x00);
    Wire.endTransmission();

    return true;
}

void updateIfDue() {
    if ((int32_t)(millis() - tNext_) < 0) return;
    tNext_ = millis() + 1000;

    Wire.beginTransmission(addr_);
    Wire.write(0x00);
    Wire.endTransmission(false);
    Wire.requestFrom(addr_, (uint8_t)2);

    if (Wire.available() == 2) {
        uint8_t msb = Wire.read();
        uint8_t lsb = Wire.read();
        int16_t raw = (int16_t)((msb << 8) | lsb);
        if (raw != 0) {
            float celsius = raw * 0.00390625f;
            lastTemp_ = (int)(celsius * 100.0f) + TEMP_OFFSET_X100;
        }
    }
}

int tempC_x100() { return lastTemp_; }

} // namespace Temp
