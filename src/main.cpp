/* Firmware iHealth32: monitoreo de signos vitales con clasificación por red neuronal.
   El ESP32 tiene dos núcleos (cores) que usamos para separar tareas críticas:
     Core 0 — captura el ECG a 250 muestras/segundo de forma precisa con FreeRTOS.
     Core 1 — lee los sensores I2C (MAX30100, MAX30205), calcula las features del ECG,
              ejecuta la red neuronal cada segundo y envía los datos por WiFi/WebSocket. */

#include <Arduino.h>
#include <Wire.h>
#include "Oximetro_MAX30100.h"
#include "ECG_AD8232.h"
#include "Temp_MAX30205.h"
#include "WiFi_WS.h"
#include "Clasificador.h"
#include "ECG_Features.h"

// Pines hardware: SDA=21, SCL=22 (I2C), ECG=34 (ADC1), LO+=26, LO-=27 (lead-off)
static const int SDA_PIN  = 21;
static const int SCL_PIN  = 22;
static const int ECG_PIN  = 34;
static const int LO_PLUS  = 26;
static const int LO_MINUS = 27;

/* Buffer circular del ECG: almacena las últimas 250 muestras (= 1 segundo a 250 Hz).
   Funciona como un anillo: cuando llega al final vuelve al principio sobreescribiendo
   la muestra más antigua. Core 0 escribe muestras; Core 1 las lee cada segundo.
   Antes de calcular los features, el buffer se "lineariza" (reordena en orden
   cronológico) para evitar que el salto entre el final y el principio del anillo
   parezca un cambio brusco de señal y distorsione los estadísticos ECG. */
static const int ECG_BUF_SIZE = 250;
static int  ecg_buf[ECG_BUF_SIZE];
static int  ecg_buf_idx   = 0;
static bool ecg_buf_ready = false;  // true tras la primera vuelta completa del anillo
static int  ecg_lin[ECG_BUF_SIZE];  // copia linearizada (orden cronologico) para features

// Cabecera de columnas del monitor (se repite cada 25 filas)
static void imprimirCabecera() {
    Serial.println();
    Serial.println(" ────────────────────────────────────────────────────────────────────────────");
    Serial.println("    Tiempo      BPM (lpm)   SpO2 (%)    Temp (C)    ECG (ADC)   Estado");
    Serial.println(" ────────────────────────────────────────────────────────────────────────────");
}

void setup() {
    Serial.begin(115200);

    Serial.println();
    Serial.println("  ═══════════════════════════════════");
    Serial.println("       iHEALTH32 - ESP32");
    Serial.println("  ═══════════════════════════════════");
    Serial.println();

    // WiFi AP + WebSocket antes de iniciar sensores: la conexion de clientes
    // puede empezar mientras los sensores calientan (warmup ~20s del MAX30100)
    wifiWsSetup();

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);  // Fast Mode I2C (400 kHz): cuatro veces más rápido que el modo estándar
                            // (100 kHz). Libera el bus I2C antes, dejando más tiempo disponible
                            // para que updateFast() drene el MAX30100 sin demoras.

    // MAX30100 y MAX30205 comparten el bus I2C; se inicializan juntos
    bool error = false;
    if (!Oxi::begin())      { Serial.println("  [ERROR] MAX30100 no responde en I2C (BPM/SpO2)."); error = true; }
    if (!Temp::begin(0x4C)) { Serial.println("  [ERROR] MAX30205 no responde en I2C (Temp)."); error = true; }
    if (error) {
        // Sin sensores vitales el sistema no tiene sentido: halt para forzar revision
        Serial.println("  Revisa el cableado I2C (SDA=GPIO21, SCL=GPIO22) y reinicia.");
        while (1) delay(1000);
    }
    Serial.println("  [OK] Sensores listos: MAX30100 (BPM/SpO2)  |  MAX30205 (Temp)  |  AD8232 (ECG)");

    // ECG se inicia despues de I2C: begin() lanza la tarea FreeRTOS en Core 0,
    // que empieza a muestrear inmediatamente a 200 Hz con vTaskDelayUntil
    ECG::begin(ECG_PIN, LO_PLUS, LO_MINUS);

    // Carga el modelo TFLite en el tensor_arena estatico (24 KB RAM)
    RNA::setup();
    Serial.println("  [OK] Red neuronal lista  (8 entradas: BPM + SpO2 + Temp + 5x ECG).");
    Serial.println();
    Serial.println("  Iniciando monitoreo cada 1 segundo...");
    imprimirCabecera();
}

// loop() corre en Core 1 sin delay() propio; el timer de 1s es software (ultimoAnalisis)
void loop() {
    // Libera descriptores WebSocket cerrados; se llama antes de todo para no acumular
    wifiWsCleanup();

    /* updateFast() DEBE llamarse en CADA iteración del loop, sin ningún delay previo.
       El MAX30100 tiene un buffer interno (FIFO) que acumula muestras a 100 Hz.
       Si tardamos más de ~160 ms en vaciarlo, el FIFO se desborda, pierde muestras
       y el algoritmo interno de detección de latidos falla hasta reiniciar el sensor. */
    Oxi::updateFast();
    Temp::updateIfDue();           // lee MAX30205 si ha pasado >= 1s desde la ultima lectura
    Oxi::updateVitalsIfDue(1000);  // consolida BPM y SpO2 a partir del stream PPG cada 1s

    /* Patrón productor-consumidor entre los dos núcleos:
       Core 0 (el productor) lee el ADC cada 5 ms y activa la bandera newSample_.
       Core 1 (el consumidor) comprueba aquí esa bandera, coge la muestra y la
       escribe en el buffer circular. Así los dos núcleos trabajan en paralelo
       sin bloquearse el uno al otro. */
    if (ECG::hasNewSample()) {
        ecg_buf[ecg_buf_idx] = ECG::lastValuePlot();
        ecg_buf_idx = (ecg_buf_idx + 1) % ECG_BUF_SIZE;
        // Primera vuelta completa: a partir de aqui el buffer contiene siempre 200 muestras
        if (ecg_buf_idx == 0) ecg_buf_ready = true;
    }

    /* Filtro de persistencia: evita falsas alarmas. La alerta solo se activa si
       la RNA predice la misma condición anómala durante N segundos seguidos Y
       con al menos 70% de confianza. Cada clase tiene su propio umbral según
       la urgencia clínica: FA necesita 10 s, fiebre 20 s, taquicardia 15 s. */
    static const char* CONDICION[] = {
        "Sano", "Fibrilacion Auricular", "Hipoxia/Apnea",
        "Fiebre/Infeccion", "Taquicardia", "Hipotermia"
    };
    // Umbral de segundos consecutivos requeridos por clase para activar alerta
    static const int UMBRAL_S[] = { 3, 10, 10, 20, 15, 20 };

    // Variables static: se inicializan una sola vez y persisten entre iteraciones del loop
    static unsigned long ultimoAnalisis     = 0;
    static float         ultimoBPM_leido    = 0;
    static unsigned long tiempoUltimoLatido = 0;
    static int           persistencia       = 0;
    static int           clase_previa       = -1;
    static int           lineCount          = 0;

    unsigned long ahora = millis();
    // Bloque de analisis a exactamente 1 Hz: se salta el resto del loop hasta que pasen 1000 ms
    if (ahora - ultimoAnalisis < 1000) return;
    ultimoAnalisis += 1000;  // += en vez de = ahora para no acumular deriva en el timer

    // Calculo de minutos:segundos para la columna Tiempo del monitor serial
    unsigned long t_s = ahora / 1000;
    unsigned long mm  = t_s / 60;
    unsigned long ss  = t_s % 60;

    /* Los drivers guardan los valores como enteros escalados (×10 o ×100)
       en lugar de números decimales. Es más eficiente en el ESP32.
       Aquí los convertimos a float real para usarlos en los cálculos:
       753 / 10.0 = 75.3 lpm  |  985 / 10.0 = 98.5 %  |  3650 / 100.0 = 36.50 °C */
    float bpm    = Oxi::bpm_x10()    / 10.0f;   // ej: 753 -> 75.3 lpm
    float spo2   = Oxi::spo2_x10()   / 10.0f;   // ej: 985 -> 98.5 %
    float temp   = Temp::tempC_x100() / 100.0f;  // ej: 3650 -> 36.50 C
    int   ecgRaw = ECG::lastValuePlot();          // ultimo valor ADC 0-4095 para el JSON

    /* Watchdog del MAX30100: cuando retiras el dedo, el sensor deja de actualizar
       el BPM pero mantiene el último valor válido congelado en lugar de volver a 0.
       Detectamos esto comprobando si el BPM ha cambiado en los últimos 4 segundos.
       Si no ha cambiado, asumimos que el sensor está desconectado y forzamos bpm=spo2=0. */
    if (bpm != ultimoBPM_leido) {
        ultimoBPM_leido    = bpm;
        tiempoUltimoLatido = ahora;
    }
    if (ahora - tiempoUltimoLatido >= 4000) { bpm = 0.0f; spo2 = 0.0f; }

    /* Validación de sensores: antes de ejecutar la RNA comprobamos que los cuatro
       sensores dan valores dentro de rangos fisiológicos razonables.
       BPM: 20–250 lpm | SpO2: 50–100 % | Temp: 28–45 °C | ECG: >50 % de muestras no cero.
       Si alguno falla (electrodo suelto, sensor desconectado), la RNA no se ejecuta
       porque sus predicciones no serían fiables con datos inválidos. */
    int ecg_n       = ecg_buf_ready ? ECG_BUF_SIZE : ecg_buf_idx;
    int ecg_nonzero = 0;
    for (int i = 0; i < ecg_n; i++) { if (ecg_buf[i] != 0) ecg_nonzero++; }
    // El driver ECG pone 0 cuando detecta lead-off (electrodo suelto);
    // si mas de la mitad de las muestras son 0, el electrodo no esta bien colocado
    bool bpm_ok  = (bpm  >  20.0f && bpm  < 250.0f);
    bool spo2_ok = (spo2 >  50.0f && spo2 <= 100.0f);
    bool temp_ok = (temp >  28.0f && temp <   45.0f);  // <28C = sensor en el aire, no en piel
    bool ecg_ok  = (ecg_n > 10) && (ecg_nonzero > ecg_n / 2);
    bool sensores_ok = bpm_ok && spo2_ok && temp_ok && ecg_ok;

    int   alertaVal = 0;  // 0 = sin alerta activa; 1-5 = clase en alerta
    float conf      = 0.0f;
    int   clase     = -1;
    int   umbral    = 3;

    // Inferencia (requiere todos los sensores validos)
    if (sensores_ok) {
        float ecg_range, ecg_skew, ecg_kurtosis, ecg_zc, ecg_slope;

        /* Linearizar el buffer circular antes de calcular features:
           reorganiza las 200 muestras en orden cronológico (de la más antigua
           a la más reciente). Sin este paso, el último elemento del anillo y el
           primero no son consecutivos en el tiempo, y ecg_max_slope calcularía
           una pendiente enorme que no corresponde a ningún latido real. */
        if (ecg_buf_ready) {
            for (int i = 0; i < ECG_BUF_SIZE; i++)
                ecg_lin[i] = ecg_buf[(ecg_buf_idx + i) % ECG_BUF_SIZE];
            calcular_features_ecg(ecg_lin, ECG_BUF_SIZE, ecg_range, ecg_skew, ecg_kurtosis, ecg_zc, ecg_slope);
        } else {
            // Antes de la primera vuelta completa se usan las muestras disponibles en orden
            calcular_features_ecg(ecg_buf, ecg_n, ecg_range, ecg_skew, ecg_kurtosis, ecg_zc, ecg_slope);
        }

        // RNA::analizar normaliza internamente los 8 valores con Z-score (usando los
        // parámetros de normalizacion.h) antes de pasarlos al modelo TFLite.
        clase = RNA::analizar(bpm, spo2, temp,
                              ecg_range, ecg_skew, ecg_kurtosis, ecg_zc, ecg_slope);
        conf = RNA::confianza();  // probabilidad softmax de la clase ganadora (0.0-1.0)

        /* Filtro de persistencia: si la clase predicha cambia, el contador vuelve a 0.
           Solo cuenta segundos para clases anómalas (>0); si la RNA predice "Sano",
           el contador se resetea. Así una detección aislada no dispara la alerta. */
        if (clase != clase_previa) { persistencia = 0; clase_previa = clase; }
        persistencia = (clase > 0) ? persistencia + 1 : 0;
        umbral = (clase > 0 && clase < 6) ? UMBRAL_S[clase] : 3;
        // Alerta activa: segundos >= umbral AND confianza >= 70%
        alertaVal = (persistencia >= umbral && clase > 0 && conf >= 0.70f) ? clase : 0;
    } else {
        // Sin sensores validos se reinicia el filtro para no arrastrar estado previo
        persistencia = 0;
        clase_previa = -1;
    }

    // Salida serial (siempre, con "--.-" para sensores ausentes)
    if (lineCount > 0 && lineCount % 25 == 0) imprimirCabecera();
    lineCount++;

    /* Durante los primeros 20s, si BPM no es valido aun, se muestra progreso de calentamiento.
       El MAX30100 necesita ~10-20s de senial PPG continua para estabilizar el algoritmo de BPM */
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

        // Construir sufijo "[SIN: BPM SpO2 ...]" solo si hay algun sensor fuera de rango
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
            // Alerta activa: clase sostenida >= umbral con confianza >= 70%
            Serial.printf("  [%02lu:%02lu]  %5.1f BPM  %4.1f%%  %5.2fC  ECG:%4d | ALERTA: %s (%d%%)\n",
                          mm, ss, bpm, spo2, temp, ecgRaw, CONDICION[alertaVal], pct);
        } else if (clase > 0) {
            // Clase anomala detectada pero aun acumulando segundos hacia el umbral
            Serial.printf("  [%02lu:%02lu]  %5.1f BPM  %4.1f%%  %5.2fC  ECG:%4d | %s (%d%%) [%d/%ds]\n",
                          mm, ss, bpm, spo2, temp, ecgRaw, CONDICION[clase], pct, persistencia, umbral);
        } else {
            // Estado normal o sensores ausentes
            Serial.printf("  [%02lu:%02lu]  %5.1f BPM  %4.1f%%  %5.2fC  ECG:%4d%s\n",
                          mm, ss, bpm, spo2, temp, ecgRaw, sin_s);
        }
    }

    // Enviar JSON por WebSocket cada 1 segundo a app movil
    // Buffer estatico (96 bytes) para evitar fragmentar la RAM en sesiones largas
    // Campo "sensores": 1 si todos OK, 0 si alguno falla (app muestra "--.-")
    static char wsJson[96];
    snprintf(wsJson, sizeof(wsJson),
             "{\"ecg\":%d,\"bpm\":%.1f,\"spo2\":%.1f,\"temp\":%.2f"
             ",\"alerta\":%d,\"conf\":%.2f,\"sensores\":%d}",
             ecgRaw, bpm, spo2, temp, alertaVal, conf, sensores_ok ? 1 : 0);
    wifiWsBroadcast(String(wsJson));
}
