// Implementación del módulo WiFi y WebSocket. wifiWsSetup() levanta el punto de acceso WiFi
// y arranca el servidor WebSocket en ws://192.168.4.1/ws. wifiWsBroadcast() envía el JSON
// de vitales a todos los clientes conectados. wifiWsCleanup() libera las conexiones cerradas
// para evitar que se acumulen y agoten la memoria del ESP32 tras sesiones largas.
#include "wifi_ws.h"
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

static const char* AP_SSID     = "iHealth32";
static const char* AP_PASSWORD = "F7%qp54ñn91jIHEALTH32éW3&mK8@tLx#R9";

static AsyncWebServer wsServer(80);
static AsyncWebSocket ws("/ws");

static void onWsEvent(AsyncWebSocket*, AsyncWebSocketClient*,
                      AwsEventType, void*, uint8_t*, size_t) {}

void wifiWsSetup() {
    WiFi.softAP(AP_SSID, AP_PASSWORD);
    Serial.printf("  [WiFi] AP '%s' activo - IP: %s\n",
                  AP_SSID, WiFi.softAPIP().toString().c_str());
    ws.onEvent(onWsEvent);
    wsServer.addHandler(&ws);
    wsServer.begin();
    Serial.println("  [WiFi] WebSocket en ws://192.168.4.1/ws");
}

void wifiWsBroadcast(const String& json) {
    if (ws.count() == 0) return;
    ws.textAll(json);
}

void wifiWsCleanup() {
    ws.cleanupClients();
}
