/*
 * main-test.cpp — Firmware iHealth32 en modo test (forzado de alertas)
 * =====================================================================
 * Variante del firmware de producción para verificar que el clasificador
 * y las alertas funcionan correctamente. Cada señal (BPM, SpO₂, Temp, ECG)
 * puede ser real o forzada a un valor objetivo según las constantes
 * TEST_*_TARGET al principio del archivo (valor negativo = usar el sensor real).
 * Los valores forzados oscilan levemente para que la RNA reciba datos que
 * parecen reales y no un vector estático repetido segundo a segundo.
 *
 * Para qué sirve:
 *   - Probar que la RNA clasifica bien cada condición sin necesitar todos
 *     los sensores conectados (AD8232 y MAX30205 pueden estar ausentes).
 *   - Forzar una condición clínica específica (fiebre, hipoxia…) cambiando
 *     los TEST_*_TARGET y reflasheando, para comprobar que la alerta se activa.
 *   - Verificar la comunicación WebSocket y la visualización en la app con
 *     datos de ECG predecibles con forma PQRST paramétrica.
 *
 * Selección en PlatformIO:
 *   El entorno "testing" en platformio.ini usa build_src_filter para
 *   compilar este archivo en lugar de main.cpp. Nunca se compilan los dos
 *   a la vez (produciría símbolos duplicados: dos definiciones de setup/loop).
 *
 * Clasificador RNA: 8 entradas → 64 → 32 → 6 salidas
 *   0=Sano | 1=FA | 2=Hipoxia | 3=Fiebre | 4=Taquicardia | 5=Hipotermia
 */

#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include "Oximetro_MAX30100.h"
#include "ECG_AD8232.h"
#include "Temp_MAX30205.h"
#include "WiFi_WS.h"
#include "Clasificador.h"
#include "ECG_Features.h"

#define TEST_TEMP_ECG

// ── Valores objetivo para forzar cada condición de prueba ──────────────────
// Cada condición clínica tiene un conjunto de valores típicos que permiten
// al clasificador reconocerla. Los valores de abajo son los centroides
// aproximados de cada clase en el dataset de entrenamiento.
//
// Cambia los TEST_*_TARGET para forzar la condición que quieres probar:
//
//              BPM    SPO2   TEMP   RANGE  SKEW   KURT   ZC     SLOPE
//   Sano    →  -1     99.5   36.0   7.8    1.0    3.6    6.0    1.9
//   Fiebre  →  110    99.0   38.5   7.8    1.0    3.6    6.0    1.9
//   Hipoxia →  -1     88.0   36.5   7.8    1.0    3.6    6.0    1.9
//   Taqui   →  130    97.0   36.8   8.5    1.2    4.0    8.0    2.2
//   Hipo    →  -1     96.0   34.5   7.5    0.9    3.4    5.5    1.8
//
// TEST_*_TARGET < 0 → se usa el valor real del sensor (no se fuerza ese canal).
// Aplica a BPM, SpO2 y Temp. El ECG siempre usa la forma de prueba PQRST.

#define TEST_BPM_TARGET     -1.0f  // lpm  — < 0 = BPM real del MAX30100
#define TEST_SPO2_TARGET    -1.0f  // %    — < 0 = SpO2 real del MAX30100
#define TEST_TEMP_TARGET    36.0f  // °C   — < 0 = Temp real del MAX30205
#define TEST_ECG_RANGE       7.8f  // ecg_range
#define TEST_ECG_SKEW        1.0f  // ecg_skew
#define TEST_ECG_KURT        3.6f  // ecg_kurtosis
#define TEST_ECG_ZC          6.0f  // ecg_zero_crossings
#define TEST_ECG_SLOPE       1.9f  // ecg_max_slope

// Amplitudes de oscilación alrededor de los valores objetivo.
// Los valores forzados no son estáticos: oscilan levemente con funciones
// sinusoidales de distintos períodos para que no estén correlados entre sí.
// Esto evita que la RNA reciba siempre el mismo vector exacto, que sería
// demasiado artificial; con la oscilación se comporta más como datos reales.
#define TEST_BPM_AMP         3.0f
#define TEST_SPO2_AMP        0.20f
#define TEST_TEMP_AMP        0.10f
#define TEST_ECG_RANGE_AMP   0.30f
#define TEST_ECG_SKEW_AMP    0.10f
#define TEST_ECG_KURT_AMP    0.20f
#define TEST_ECG_ZC_AMP      0.60f
#define TEST_ECG_SLOPE_AMP   0.08f

// ---------------------------------------------------------------------------
// Pines hardware
// ---------------------------------------------------------------------------
static const int SDA_PIN  = 21;
static const int SCL_PIN  = 22;
static const int ECG_PIN  = 34;
static const int LO_PLUS  = 26;
static const int LO_MINUS = 27;

// ---------------------------------------------------------------------------
// Buffer circular ECG — almacena 1 segundo de muestras a 200 Hz
// ---------------------------------------------------------------------------
static const int ECG_BUF_SIZE = 200;
static int  ecg_buf[ECG_BUF_SIZE];
static int  ecg_buf_idx   = 0;
static bool ecg_buf_ready = false;
static int  ecg_lin[ECG_BUF_SIZE];

// ---------------------------------------------------------------------------
// Cabecera de columnas del monitor (se repite cada 25 filas)
// ---------------------------------------------------------------------------
static void imprimirCabecera() {
    Serial.println();
    Serial.println(" ────────────────────────────────────────────────────────────────────────────");
    Serial.println("    Tiempo      BPM (lpm)   SpO2 (%)    Temp (C)    ECG (ADC)   Estado");
    Serial.println(" ────────────────────────────────────────────────────────────────────────────");
}

// ---------------------------------------------------------------------------
// Funciones que inyectan valores forzados para el test (bloque TEST_TEMP_ECG)
// ---------------------------------------------------------------------------
// Cada función devuelve el valor de prueba correspondiente oscilando alrededor
// del objetivo. Los períodos son números "raros" (5100 ms, 3183 ms…) para que
// BPM, SpO2 y Temp no oscilen sincronizados entre sí, lo que haría evidente
// que los datos son forzados y no reales.
#ifdef TEST_TEMP_ECG

static float testGetBpm(float bpm_real) {
    // Si TEST_BPM_TARGET < 0, se usa el BPM real del sensor sin forzar nada.
    if (TEST_BPM_TARGET < 0.0f) return bpm_real;
    return TEST_BPM_TARGET + TEST_BPM_AMP * sinf(millis() / 5100.0f);
}

static float testGetSpO2(float spo2_real) {
    if (TEST_SPO2_TARGET < 0.0f) return spo2_real;
    return TEST_SPO2_TARGET + TEST_SPO2_AMP * sinf(millis() / 3183.0f);
}

static float testGetTemp(float temp_real) {
    if (TEST_TEMP_TARGET < 0.0f) return temp_real;
    return TEST_TEMP_TARGET + TEST_TEMP_AMP * sinf(millis() / 4775.0f);
}

// Genera las 5 features ECG de prueba oscilando alrededor de los valores
// objetivo. Van directamente a la RNA sin pasar por el buffer circular ni
// por calcular_features_ecg(), ya que no hay muestras ADC reales que procesar.
static void testGetEcgFeatures(float& range, float& skew, float& kurt,
                               float& zc, float& slope) {
    float t = millis() / 1000.0f;  // tiempo en segundos (float suficiente aquí)
    range = TEST_ECG_RANGE + TEST_ECG_RANGE_AMP * sinf(t /  7.3f);
    skew  = TEST_ECG_SKEW  + TEST_ECG_SKEW_AMP  * sinf(t / 11.1f);
    kurt  = TEST_ECG_KURT  + TEST_ECG_KURT_AMP  * sinf(t /  9.7f);
    zc    = TEST_ECG_ZC    + TEST_ECG_ZC_AMP    * sinf(t / 13.3f);
    slope = TEST_ECG_SLOPE + TEST_ECG_SLOPE_AMP * sinf(t /  6.1f);
}

// Genera una muestra ADC de prueba con la morfología PQRST del ECG real
// a ≈75 BPM (período de 800 ms). Se construye con dos gaussianas superpuestas:
//   - Pico R (complejo QRS): amplitud 900, centrado en t=0.35, σ=0.02 s
//     → pico estrecho y pronunciado, característico del QRS en un ECG normal
//   - Onda T: amplitud 200, centrada en t=0.60, σ=0.06 s
//     → pico más ancho y bajo, onda de repolarización ventricular
// El valor base 2048 corresponde a la mitad del rango ADC de 12 bits (0–4095),
// que es la tensión de referencia del AD8232 (VCC/2 ≈ 1.65 V → ≈ 2048 ADC).
static int testGetEcgRaw() {
    float t   = fmodf(millis() / 800.0f, 1.0f);  // fase normalizada 0..1 dentro del ciclo
    float val = 900.0f * expf(-(t - 0.35f) * (t - 0.35f) / (2.0f * 0.02f * 0.02f));
    val      += 200.0f * expf(-(t - 0.60f) * (t - 0.60f) / (2.0f * 0.06f * 0.06f));
    return 2048 + (int)val;
}

#endif  // TEST_TEMP_ECG

// ---------------------------------------------------------------------------
// setup()
// ---------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);

    Serial.println();
    Serial.println("  ═══════════════════════════════════");
    Serial.println("       iHEALTH32 - ESP32");
    Serial.println("  ═══════════════════════════════════");
    Serial.println();

    wifiWsSetup();

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);

    if (!Oxi::begin()) {
        Serial.println("  [ERROR] MAX30100 no responde en I2C (BPM/SpO2).");
        Serial.println("  Revisa el cableado I2C (SDA=GPIO21, SCL=GPIO22) y reinicia.");
        while (1) delay(1000);
    }

#ifdef TEST_TEMP_ECG
    Temp::begin(0x4C);
    Serial.println("  [OK]  MAX30100 listo.");
    // Muestra qué canal es real y cuál está forzado para el test.
    Serial.printf("  [TEST] BPM:%s  |  SpO2:%s  |  Temp:%s  |  ECG de prueba.\n",
                  TEST_BPM_TARGET  < 0 ? "real" : "forzado",
                  TEST_SPO2_TARGET < 0 ? "real" : "forzado",
                  TEST_TEMP_TARGET < 0 ? "real" : "forzado");
#else
    if (!Temp::begin(0x4C)) {
        Serial.println("  [ERROR] MAX30205 no responde en I2C (Temp).");
        Serial.println("  Revisa el cableado I2C (SDA=GPIO21, SCL=GPIO22) y reinicia.");
        while (1) delay(1000);
    }
    Serial.println("  [OK] Sensores listos: MAX30100 (BPM/SpO2)  |  MAX30205 (Temp)  |  AD8232 (ECG)");
#endif

    ECG::begin(ECG_PIN, LO_PLUS, LO_MINUS);

    RNA::setup();
    Serial.println("  [OK] Red neuronal lista  (8 entradas: BPM + SpO2 + Temp + 5x ECG).");
    Serial.println();
    Serial.println("  Iniciando monitoreo cada 1 segundo...");
    imprimirCabecera();
}

// ---------------------------------------------------------------------------
// loop() — Core 1
// ---------------------------------------------------------------------------
void loop() {
    wifiWsCleanup();

    Oxi::updateFast();
    Temp::updateIfDue();
    Oxi::updateVitalsIfDue(1000);

    if (ECG::hasNewSample()) {
        ecg_buf[ecg_buf_idx] = ECG::lastValuePlot();
        ecg_buf_idx = (ecg_buf_idx + 1) % ECG_BUF_SIZE;
        if (ecg_buf_idx == 0) ecg_buf_ready = true;
    }

    // Nombres de clase y umbrales de persistencia del filtro de estabilidad.
    // La alerta se activa solo si la RNA predice la misma clase durante N segundos
    // seguidos con confianza ≥70 %. Ver main.cpp para la justificación clínica
    // de cada umbral.
    static const char* CONDICION[] = {
        "Sano", "Fibrilacion Auricular", "Hipoxia/Apnea",
        "Fiebre/Infeccion", "Taquicardia", "Hipotermia"
    };
    static const int UMBRAL_S[] = { 3, 10, 10, 20, 15, 20 };

    static unsigned long ultimoAnalisis     = 0;
    static float         ultimoBPM_leido    = 0;
    static unsigned long tiempoUltimoLatido = 0;
    static int           persistencia       = 0;
    static int           clase_previa       = -1;
    static int           lineCount          = 0;

    unsigned long ahora = millis();
    if (ahora - ultimoAnalisis < 1000) return;
    ultimoAnalisis += 1000;

    unsigned long t_s = ahora / 1000;
    unsigned long mm  = t_s / 60;
    unsigned long ss  = t_s % 60;

    // ─── Lectura de sensores ────────────────────────────────────────────
    float bpm    = Oxi::bpm_x10()    / 10.0f;
    float spo2   = Oxi::spo2_x10()   / 10.0f;
    float temp   = Temp::tempC_x100() / 100.0f;
    int   ecgRaw = ECG::lastValuePlot();

    // Watchdog: si BPM no cambia en 4 s, el sensor está desconectado
    if (bpm != ultimoBPM_leido) {
        ultimoBPM_leido    = bpm;
        tiempoUltimoLatido = ahora;
    }
    if (ahora - tiempoUltimoLatido >= 4000) { bpm = 0.0f; spo2 = 0.0f; }

    // ─── Inyección de valores forzados para el test ────────────────────
    // Sustituye los valores del hardware por los valores de prueba según TEST_*_TARGET.
    // Si un TEST_*_TARGET es negativo, testGetXxx() devuelve el valor real del sensor
    // tal cual. Así puedes, por ejemplo, mantener el BPM real y forzar SpO2, Temp y
    // ECG para verificar que la alerta de hipoxia o fiebre se activa correctamente.
#ifdef TEST_TEMP_ECG
    bpm    = testGetBpm(bpm);
    spo2   = testGetSpO2(spo2);
    temp   = testGetTemp(temp);
    ecgRaw = testGetEcgRaw();
#endif

    // ─── Validación de sensores ─────────────────────────────────────────
    int ecg_n       = ecg_buf_ready ? ECG_BUF_SIZE : ecg_buf_idx;
    int ecg_nonzero = 0;
    for (int i = 0; i < ecg_n; i++) { if (ecg_buf[i] != 0) ecg_nonzero++; }

    bool bpm_ok  = (bpm  >  20.0f && bpm  < 250.0f);
    bool spo2_ok = (spo2 >  50.0f && spo2 <= 100.0f);
    bool temp_ok = (temp >  28.0f && temp <   42.0f);
    bool ecg_ok  = (ecg_n > 10) && (ecg_nonzero > ecg_n / 2);

    // En modo test, SpO₂, Temp y ECG se consideran siempre válidos porque los
    // valores forzados están dentro del rango fisiológico por diseño.
    // Solo BPM mantiene la validación real porque el MAX30100 sigue activo.
#ifdef TEST_TEMP_ECG
    spo2_ok = true;
    temp_ok = true;
    ecg_ok  = true;
#endif

    bool sensores_ok = bpm_ok && spo2_ok && temp_ok && ecg_ok;

    int   alertaVal = 0;
    float conf      = 0.0f;
    int   clase     = -1;
    int   umbral    = 3;

    // ─── Inferencia (requiere todos los sensores válidos) ───────────────
    if (sensores_ok) {
        float ecg_range, ecg_skew, ecg_kurtosis, ecg_zc, ecg_slope;

#ifdef TEST_TEMP_ECG
        testGetEcgFeatures(ecg_range, ecg_skew, ecg_kurtosis, ecg_zc, ecg_slope);
#else
        if (ecg_buf_ready) {
            for (int i = 0; i < ECG_BUF_SIZE; i++)
                ecg_lin[i] = ecg_buf[(ecg_buf_idx + i) % ECG_BUF_SIZE];
            calcular_features_ecg(ecg_lin, ECG_BUF_SIZE, ecg_range, ecg_skew, ecg_kurtosis, ecg_zc, ecg_slope);
        } else {
            calcular_features_ecg(ecg_buf, ecg_n, ecg_range, ecg_skew, ecg_kurtosis, ecg_zc, ecg_slope);
        }
#endif

        clase = RNA::analizar(bpm, spo2, temp,
                              ecg_range, ecg_skew, ecg_kurtosis, ecg_zc, ecg_slope);
        conf = RNA::confianza();

        // Filtro de persistencia: la alerta solo se activa si la RNA predice la
        // misma condición anómala N segundos seguidos con confianza ≥70 %.
        // Si la clase predicha cambia en algún segundo, el contador vuelve a 0.
        if (clase != clase_previa) { persistencia = 0; clase_previa = clase; }
        persistencia = (clase > 0) ? persistencia + 1 : 0;
        umbral = (clase > 0 && clase < 6) ? UMBRAL_S[clase] : 3;
        alertaVal = (persistencia >= umbral && clase > 0 && conf >= 0.70f) ? clase : 0;
    } else {
        persistencia = 0;
        clase_previa = -1;
    }

    // ─── Salida serial (siempre, con "--.-" para sensores ausentes) ─────
    if (lineCount > 0 && lineCount % 25 == 0) imprimirCabecera();
    lineCount++;

    if (ahora < 20000UL && !bpm_ok) {
        Serial.printf("  [%02lu:%02lu] Calentando BPM/SpO2... [%2lus/20s]%s%s\n",
            mm, ss, ahora / 1000,
            temp_ok ? "" : " | SIN: Temp",
            ecg_ok  ? "" : " | SIN: ECG");
    } else {
        char bpm_s[8], spo2_s[7], temp_s[8], ecg_s[6], sin_s[40];
        if (bpm_ok)  snprintf(bpm_s,  sizeof(bpm_s),  "%5.1f", bpm);
        else         strcpy(bpm_s,  " --.-");
        if (spo2_ok) snprintf(spo2_s, sizeof(spo2_s), "%4.1f", spo2);
        else         strcpy(spo2_s, "--.-");
        if (temp_ok) snprintf(temp_s, sizeof(temp_s), "%5.2f", temp);
        else         strcpy(temp_s, " --.-");
        snprintf(ecg_s, sizeof(ecg_s), "%4d", ecgRaw);

        sin_s[0] = '\0';
        if (!sensores_ok) {
            strcpy(sin_s, " [SIN:");
            if (!bpm_ok)  strcat(sin_s, " BPM");
            if (!spo2_ok) strcat(sin_s, " SpO2");
            if (!temp_ok) strcat(sin_s, " Temp");
            if (!ecg_ok)  strcat(sin_s, " ECG");
            strcat(sin_s, "]");
        }

        int pct = (int)(conf * 100.0f + 0.5f);
        if (alertaVal > 0) {
            Serial.printf("  [%02lu:%02lu]  %s BPM  %s%%  %sC  ECG:%s | ALERTA: %s (%d%%)\n",
                          mm, ss, bpm_s, spo2_s, temp_s, ecg_s, CONDICION[alertaVal], pct);
        } else if (clase > 0) {
            Serial.printf("  [%02lu:%02lu]  %s BPM  %s%%  %sC  ECG:%s | %s (%d%%) [%d/%ds]\n",
                          mm, ss, bpm_s, spo2_s, temp_s, ecg_s, CONDICION[clase], pct, persistencia, umbral);
        } else {
            Serial.printf("  [%02lu:%02lu]  %s BPM  %s%%  %sC  ECG:%s%s\n",
                          mm, ss, bpm_s, spo2_s, temp_s, ecg_s, sin_s);
        }
    }

    // ─── WebSocket JSON ─────────────────────────────────────────────────
    // Buffer estático con snprintf: sin allocaciones de heap por iteración.
    static char wsJson[96];
    snprintf(wsJson, sizeof(wsJson),
             "{\"ecg\":%d,\"bpm\":%.1f,\"spo2\":%.1f,\"temp\":%.2f"
             ",\"alerta\":%d,\"conf\":%.2f,\"sensores\":%d}",
             ecgRaw, bpm, spo2, temp, alertaVal, conf, sensores_ok ? 1 : 0);
    wifiWsBroadcast(String(wsJson));
}
