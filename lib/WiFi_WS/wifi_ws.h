// Módulo de comunicación WiFi y WebSocket. El ESP32 actúa como punto de acceso (AP):
// crea su propia red WiFi "iHealth32" a la que se conecta la app móvil. WebSocket mantiene
// una conexión abierta para enviar datos en tiempo real, a diferencia de HTTP donde habría
// que pedir los datos repetidamente. Los datos viajan en formato JSON (texto estructurado).
#pragma once
#include <Arduino.h>

void wifiWsSetup();
void wifiWsBroadcast(const String& json);
void wifiWsCleanup();
