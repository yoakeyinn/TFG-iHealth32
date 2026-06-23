"""
preparar_datos.py — Descarga y prepara el dataset de entrenamiento para iHealth32
==================================================================================
Descarga ECGs reales de PhysioNet (repositorio médico público con miles de registros
clínicos anotados por cardiólogos) y los convierte en el formato que necesita
entrenar_rna.py: un CSV por cada clase con columnas BPM, SpO2, Temp y 5 features ECG.

Fuentes utilizadas (todas gratuitas, sin contraseña):
  1. NSRDB — adultos sanos a 128 Hz                       →  Clase 0: Sano
  2. MIT-BIH AF Database — FA anotada por cardiólogos      →  Clase 1: Fibrilación Auricular
  3. Apnea-ECG Database — apnea del sueño a 100 Hz        →  Clase 2: Hipoxia/Apnea
  4. CinC Challenge 2019 — vitales reales de UCI           →  Clases 3, 4, 5: Fiebre, Taquicardia, Hipotermia

  Nota: la bradicardia (BPM < 50) se detecta en el firmware por umbral simple y no
  tiene clase propia en la RNA.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PARÁMETROS GLOBALES
# ---------------------------------------------------------------------------
FS_COMMON = 250  # Hz de referencia para features ECG
VENTANA = FS_COMMON  # 250 muestras = 1 s
MUESTRAS_MIN = 5 * 60  # 300 filas mínimas para aceptar un registro
CARPETA = "datos_clinicos"
MAX_FILAS_PHYSIONET = 10_000  # máximo de filas por registro

LABEL_NOMBRES = {
    0: "Sano",
    1: "Fibrilacion Auricular",
    2: "Hipoxia/Apnea",
    3: "Fiebre/Infeccion",
    4: "Taquicardia",
    5: "Hipotermia",
}
NUM_CLASES = 6

# ---------------------------------------------------------------------------
# DISTRIBUCIONES DE SpO2 Y TEMPERATURA POR CLASE
# (BPM se calcula siempre del ECG real)
# ---------------------------------------------------------------------------

# Sano — adultos sanos en reposo/actividad leve
_SANO_BPM_RANGO = (50, 95)
_SANO_SPO2 = (99.0, 0.7, 97.0, 100.0)  # μ, σ, clip_min, clip_max
_SANO_TEMP = (36.6, 0.3, 36.0, 37.2)

# Fibrilación Auricular
_FA_SPO2 = (97.5, 0.8, 96.0, 100.0)
_FA_TEMP = (36.5, 0.5, 35.5, 37.5)

# Hipoxia / Apnea obstructiva
_APNEA_SPO2 = (89.0, 2.5, 82.0, 94.0)
_APNEA_TEMP = (36.5, 0.3, 35.5, 37.5)

# Challenge 2019 — filtros de selección (valores reales del dataset)
_C19_FIEBRE_HR_MIN = 90.0
_C19_FIEBRE_TEMP_MIN = 38.0
_C19_TAQUI_HR_MIN = 100.0
_C19_TAQUI_HR_MAX = 150.0
_C19_TAQUI_TEMP_MIN = 36.0
_C19_TAQUI_TEMP_MAX = 37.9
_C19_HIPO_TEMP_MAX = 35.5
_C19_SPO2_MIN = 88.0
_C19_BASE = (
    "https://physionet.org/files/challenge-2019/1.0.0/" "training/training_setA/"
)


def _sample(rng, mu, sigma, lo, hi):
    """Muestra un valor de N(mu, sigma) recortado a [lo, hi]."""
    return float(np.clip(rng.normal(mu, sigma), lo, hi))


# ---------------------------------------------------------------------------
# FEATURES ECG — idénticas al firmware ESP32
# ---------------------------------------------------------------------------
# Estas funciones calculan exactamente los mismos estadísticos que calcula
# ecg_features.cpp en el ESP32. Es importante que sean idénticas: si el
# firmware calcula los features de una forma y el entrenamiento los calcula
# de otra, la red aprende patrones que luego no verá en producción.


def _norm_ecg(x: np.ndarray) -> np.ndarray:
    std = x.std(ddof=0)
    if std < 1e-10:
        return x - x.mean()
    return (x - x.mean()) / std


def _skew(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    mean, std = x.mean(), x.std()
    if std == 0:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    mean, std = x.mean(), x.std()
    if std == 0:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 4) - 3.0)


def _zero_crossings(x: np.ndarray) -> float:
    return float(np.sum(np.diff(np.sign(x)) != 0))


def _max_slope(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    if len(x) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(x))))


def _features_ecg_window(window_250hz: np.ndarray) -> dict:
    # Extrae las 5 features de una ventana de 1 segundo (250 muestras a 250 Hz).
    # Primero normaliza la ventana (media=0, std=1) igual que hace el firmware,
    # y luego calcula los 5 estadísticos sobre esa señal normalizada.
    xn = _norm_ecg(window_250hz.astype(np.float64))
    return {
        "ecg_range": float(xn.max() - xn.min()),
        "ecg_skew": _skew(xn),
        "ecg_kurtosis": _kurtosis(xn),
        "ecg_zero_crossings": _zero_crossings(xn),
        "ecg_max_slope": _max_slope(xn),
    }


def _resample_to_250(ecg: np.ndarray, fs_orig: int) -> np.ndarray:
    if fs_orig == FS_COMMON:
        return ecg
    from scipy.signal import resample

    n_target = int(len(ecg) * FS_COMMON / fs_orig)
    return resample(ecg, n_target)


def _bpm_from_ecg(ecg_250: np.ndarray, fallback: float = 0.0) -> float:
    # Estima el BPM de un segmento ECG detectando los picos R (el pico más alto
    # de cada latido, el complejo QRS). La librería WFDB incluye gqrs_detect,
    # un detector de picos validado clínicamente. El BPM se calcula como el
    # inverso del intervalo medio entre picos R consecutivos (intervalo RR).
    # Si el segmento es demasiado corto o la detección falla, devuelve fallback.
    try:
        from wfdb.processing import gqrs_detect

        if len(ecg_250) < 5 * FS_COMMON:
            return fallback
        peaks = gqrs_detect(ecg_250.astype(float), fs=FS_COMMON)
        if len(peaks) < 2:
            return fallback
        rr = np.diff(peaks) / FS_COMMON
        rr_ok = rr[(rr > 0.3) & (rr < 2.0)]
        if len(rr_ok) < 2:
            return fallback
        return float(60.0 / np.median(rr_ok))
    except Exception:
        return fallback


def _ecg_to_rows(
    ecg_250: np.ndarray, bpm: float, spo2: float, temp: float, label: int
) -> list:
    # Divide un ECG largo en ventanas de 1 segundo (250 muestras cada una) y
    # convierte cada ventana en una fila del dataset con los 8 features más la
    # etiqueta de clase. Las ventanas con señal prácticamente plana (std muy
    # pequeña, electrodo suelto o artefacto) se descartan.
    rows = []
    n_windows = len(ecg_250) // VENTANA
    for i in range(n_windows):
        win = ecg_250[i * VENTANA : (i + 1) * VENTANA]
        if np.isnan(win).any() or win.std() < 1e-10:
            continue
        row = _features_ecg_window(win)
        row["BPM"] = float(bpm)
        row["SpO2"] = float(spo2)
        row["Temp"] = float(temp)
        row["Label"] = int(label)
        rows.append(row)
    return rows


CSV_COLS = [
    "BPM",
    "SpO2",
    "Temp",
    "ecg_range",
    "ecg_skew",
    "ecg_kurtosis",
    "ecg_zero_crossings",
    "ecg_max_slope",
    "Label",
]


def _save_csv(rows: list, filename: str) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=CSV_COLS)
    for col in df.columns:
        df[col] = df[col].astype("float32")
    df["Label"] = df["Label"].astype(int)
    df.to_csv(os.path.join(CARPETA, filename), index=False, float_format="%.4f")
    return len(df)


# ===========================================================================
# FUENTE 1 — PhysioNet NSRDB (Normal Sinus Rhythm Database)
#            18 adultos sanos, ECG a 128 Hz
# ===========================================================================

_NSRDB_RECORDS = [
    "16265",
    "16272",
    "16273",
    "16420",
    "16483",
    "16539",
    "16773",
    "16786",
    "16795",
    "17052",
    "17453",
    "18177",
    "18184",
    "19088",
    "19090",
    "19093",
    "19140",
    "19830",
]

_NSRDB_FS = 128
_NSRDB_BLOQUE = 30  # segundos por bloque para calcular BPM y muestrear SpO2/Temp


def _nsrdb_worker(args: tuple) -> tuple:
    """Descarga y procesa un registro de adultos sanos de la base NSRDB.
    Divide el ECG en bloques de 30 s, calcula el BPM de cada bloque y solo
    guarda los que tienen BPM en rango sano (50–95 lpm) como Clase 0=Sano."""
    rec_id, seed = args
    import wfdb

    rng = np.random.default_rng(seed)
    try:
        rec = wfdb.rdrecord(rec_id, pn_dir="nsrdb", sampto=_NSRDB_FS * 60 * 30)
    except Exception as e:
        return rec_id, [], [], str(e)

    ecg_raw = rec.p_signal[:, 0].astype(np.float64)
    fs = rec.fs
    bloque = _NSRDB_BLOQUE * fs
    rows_s = []

    for start in range(0, len(ecg_raw) - bloque, bloque):
        seg = ecg_raw[start : start + bloque]
        seg250 = _resample_to_250(seg, fs)
        bpm = _bpm_from_ecg(seg250)

        if _SANO_BPM_RANGO[0] <= bpm <= _SANO_BPM_RANGO[1]:
            rows_s.extend(
                _ecg_to_rows(
                    seg250,
                    bpm,
                    _sample(rng, *_SANO_SPO2),
                    _sample(rng, *_SANO_TEMP),
                    label=0,
                )
            )
        if len(rows_s) >= MAX_FILAS_PHYSIONET:
            break

    return rec_id, rows_s[:MAX_FILAS_PHYSIONET], None


def download_nsrdb(n_records: int = 15, rng_seed: int = 42, workers: int = 3):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("\n" + "═" * 60)
    print("  FUENTE 1 — PhysioNet NSRDB (adultos sanos, 128 Hz)")
    print(f"  Sano BPM {_SANO_BPM_RANGO[0]}–{_SANO_BPM_RANGO[1]} lpm")
    print(f"  Paralelo: {workers} descargas simultáneas")
    print("═" * 60)

    os.makedirs(CARPETA, exist_ok=True)
    args = [
        (rec_id, rng_seed + i) for i, rec_id in enumerate(_NSRDB_RECORDS[:n_records])
    ]
    tot_sano = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_nsrdb_worker, a): a[0] for a in args}
        for fut in as_completed(futs):
            rec_id, rows_s, err = fut.result()
            if err:
                print(f"  nsrdb/{rec_id} → (error) {err}", flush=True)
                continue
            if len(rows_s) >= MUESTRAS_MIN:
                ns = _save_csv(rows_s, f"nsrdb_sano_{rec_id}.csv")
                tot_sano += 1
                print(f"  nsrdb/{rec_id} → sano:{ns:>5}", flush=True)
            else:
                print(f"  nsrdb/{rec_id} → (salta)", flush=True)

    print(f"\n  >> {tot_sano} registros Sano")


# ===========================================================================
# FUENTE 2 — MIT-BIH Atrial Fibrillation Database (afdb, 250 Hz)
# ===========================================================================

_AFDB_RECORDS = [
    "04015",
    "04043",
    "04048",
    "04126",
    "04746",
    "04908",
    "05091",
    "05121",
    "05261",
    "06426",
    "06453",
    "06995",
    "07162",
    "07859",
    "07879",
    "07910",
    "08215",
    "08219",
    "08378",
    "08405",
    "08434",
    "08455",
    "08701",
    "08762",
    "08867",
]
_AFDB_BLOQUE = 30  # segundos por bloque


def _afdb_worker(args: tuple) -> tuple:
    """Descarga y procesa un registro de la base de fibrilación auricular (afdb).
    Lee las anotaciones del cardiólogo para identificar los segmentos etiquetados
    como AFIB y solo convierte esos tramos en filas del dataset (Clase 1=FA)."""
    rec_id, seed = args
    import wfdb

    rng = np.random.default_rng(seed)
    try:
        rec = wfdb.rdrecord(rec_id, pn_dir="afdb")
        ann = wfdb.rdann(rec_id, "atr", pn_dir="afdb")
    except Exception as e:
        return rec_id, [], str(e)

    ecg_raw = rec.p_signal[:, 0].astype(np.float64)
    fs = rec.fs

    afib_segs, cur_start = [], None
    for sample, note in zip(ann.sample, ann.aux_note):
        if note and "(AFIB" in note:
            cur_start = sample
        elif cur_start is not None:
            afib_segs.append((cur_start, sample))
            cur_start = None
    if cur_start is not None:
        afib_segs.append((cur_start, len(ecg_raw)))

    if not afib_segs:
        return rec_id, [], "sin AFIB"

    bloque = _AFDB_BLOQUE * fs
    rows = []
    for seg_start, seg_end in afib_segs:
        for start in range(seg_start, seg_end - bloque, bloque):
            seg = ecg_raw[start : start + bloque]
            seg250 = _resample_to_250(seg, fs)
            rows.extend(
                _ecg_to_rows(
                    seg250,
                    _bpm_from_ecg(seg250, fallback=85.0),
                    _sample(rng, *_FA_SPO2),
                    _sample(rng, *_FA_TEMP),
                    label=1,
                )
            )
            if len(rows) >= MAX_FILAS_PHYSIONET:
                break
        if len(rows) >= MAX_FILAS_PHYSIONET:
            break

    return rec_id, rows[:MAX_FILAS_PHYSIONET], None


def download_afdb_fa(n_records: int = 15, rng_seed: int = 43, workers: int = 3):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("\n" + "═" * 60)
    print("  FUENTE 2 — MIT-BIH AF Database (FA)")
    print(f"  Paralelo: {workers} descargas simultáneas")
    print("═" * 60)

    os.makedirs(CARPETA, exist_ok=True)
    args = [
        (rec_id, rng_seed + i) for i, rec_id in enumerate(_AFDB_RECORDS[:n_records])
    ]
    total = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_afdb_worker, a): a[0] for a in args}
        for fut in as_completed(futs):
            rec_id, rows, err = fut.result()
            if err:
                print(f"  afdb/{rec_id} → ({err})")
                continue
            if len(rows) < MUESTRAS_MIN:
                print(f"  afdb/{rec_id} → (salta) {len(rows)} ventanas")
                continue
            n = _save_csv(rows, f"afdb_fibrilacion_auricular_{rec_id}.csv")
            print(f"  afdb/{rec_id} → {n:>6} filas AFIB")
            total += 1

    print(f"\n  >> {total} registros FA guardados")


# ===========================================================================
# FUENTE 3 — PhysioNet Apnea-ECG Database (100 Hz)
# ===========================================================================

_APNEA_RECORDS = [f"a{i:02d}" for i in range(1, 21)]  # a01–a20


def _apnea_worker(args: tuple) -> tuple:
    """Descarga y procesa un registro de la base Apnea-ECG.
    Las anotaciones minuto a minuto marcan con 'A' los episodios de apnea.
    Solo se procesan esos minutos como Clase 2=Hipoxia (en apnea el paciente
    deja de respirar y el oxígeno en sangre cae de forma característica)."""
    rec_id, seed = args
    import wfdb

    rng = np.random.default_rng(seed)
    try:
        rec = wfdb.rdrecord(rec_id, pn_dir="apnea-ecg/1.0.0")
        ann = wfdb.rdann(rec_id, "apn", pn_dir="apnea-ecg/1.0.0")
    except Exception as e:
        return rec_id, [], 0, str(e)

    ecg_raw = rec.p_signal[:, 0].astype(np.float64)
    fs = rec.fs
    apnea_samples = ann.sample[np.array(ann.symbol) == "A"]
    if len(apnea_samples) == 0:
        return rec_id, [], 0, "sin apnea"

    min_s = 60 * fs
    rows = []
    for start in apnea_samples:
        end = start + min_s
        if end > len(ecg_raw):
            continue
        seg = ecg_raw[start:end]
        seg250 = _resample_to_250(seg, fs)
        rows.extend(
            _ecg_to_rows(
                seg250,
                _bpm_from_ecg(seg250, fallback=60.0),
                _sample(rng, *_APNEA_SPO2),
                _sample(rng, *_APNEA_TEMP),
                label=2,
            )
        )
        if len(rows) >= MAX_FILAS_PHYSIONET:
            break

    return rec_id, rows[:MAX_FILAS_PHYSIONET], len(apnea_samples), None


def download_apnea_ecg_hipoxia(
    n_records: int = 15, rng_seed: int = 44, workers: int = 3
):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("\n" + "═" * 60)
    print("  FUENTE 3 — PhysioNet Apnea-ECG (Hipoxia/Apnea)")
    print(f"  Paralelo: {workers} descargas simultáneas")
    print("═" * 60)

    os.makedirs(CARPETA, exist_ok=True)
    args = [
        (rec_id, rng_seed + i) for i, rec_id in enumerate(_APNEA_RECORDS[:n_records])
    ]
    total = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_apnea_worker, a): a[0] for a in args}
        for fut in as_completed(futs):
            rec_id, rows, n_apnea, err = fut.result()
            if err:
                print(f"  apnea-ecg/{rec_id} → ({err})")
                continue
            if len(rows) < MUESTRAS_MIN:
                print(f"  apnea-ecg/{rec_id} → (salta) {len(rows)} ventanas")
                continue
            n = _save_csv(rows, f"apneaecg_hipoxia_apnea_{rec_id}.csv")
            print(f"  apnea-ecg/{rec_id} → {n:>6} filas  ({n_apnea} min apnea)")
            total += 1

    print(f"\n  >> {total} registros Hipoxia/Apnea guardados")


# ===========================================================================
# FUENTE 4 — PhysioNet CinC Challenge 2019 (vitales UCI)
#            Una sola pasada captura tres clases:
#              label 3 — Fiebre      : HR ≥ 90, Temp ≥ 38.0
#              label 5 — Taquicardia : HR ∈ [100,150], Temp ∈ [36.0, 37.9]
#              label 6 — Hipotermia  : Temp ≤ 35.5
#            CSVs guardados con vitales reales (HR, SpO2, Temp)
# ===========================================================================

_C19_WORKERS = 8
_C19_MAXPID = 10_000


def _fetch_c19(pid: int) -> str | None:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{_C19_BASE}p{pid:06d}.psv", timeout=10) as r:
            return r.read().decode("utf-8")
    except Exception:
        return None


def _c19_save(rows: list, label: int, nombre: str, target: int) -> int:
    if len(rows) < MUESTRAS_MIN:
        print(f"  >> {nombre}: insuficiente ({len(rows)} filas)")
        return 0
    sub = rows[:target]
    df = pd.DataFrame(sub, columns=["BPM", "SpO2", "Temp", "Label"])
    df = df.astype(
        {"BPM": "float32", "SpO2": "float32", "Temp": "float32", "Label": int}
    )
    df.to_csv(
        os.path.join(CARPETA, f"challenge2019_{nombre}.csv"),
        index=False,
        float_format="%.4f",
    )
    print(f"  >> {nombre}: {len(df)} filas")
    return len(df)


def download_challenge2019(target_por_clase: int = 6000):
    """
    Descarga datos de vitales reales del CinC Challenge 2019 (pacientes de UCI).
    Una sola pasada con 8 hilos paralelos recolecta las tres clases a la vez
    (Fiebre, Taquicardia e Hipotermia), filtrando cada fila según umbrales
    clínicos de HR y Temp. Los CSVs resultantes incluyen BPM, SpO2, Temp y Label.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("\n" + "═" * 60)
    print("  FUENTE 4 — PhysioNet CinC Challenge 2019")
    print(f"  Fiebre      : HR ≥ {_C19_FIEBRE_HR_MIN}, Temp ≥ {_C19_FIEBRE_TEMP_MIN}")
    print(
        f"  Taquicardia : HR {_C19_TAQUI_HR_MIN}–{_C19_TAQUI_HR_MAX},"
        f" Temp {_C19_TAQUI_TEMP_MIN}–{_C19_TAQUI_TEMP_MAX}"
    )
    print(f"  Hipotermia  : Temp ≤ {_C19_HIPO_TEMP_MAX}")
    print(f"  Revisando primeros {_C19_MAXPID} pacientes ({_C19_WORKERS} hilos)")
    print("═" * 60)

    os.makedirs(CARPETA, exist_ok=True)
    fiebre, taqui, hipo = [], [], []
    done = 0

    with ThreadPoolExecutor(max_workers=_C19_WORKERS) as ex:
        futs = {ex.submit(_fetch_c19, pid): pid for pid in range(1, _C19_MAXPID + 1)}

        for fut in as_completed(futs):
            done += 1
            if done % 500 == 0:
                print(
                    f"   {done}/{_C19_MAXPID}  F:{len(fiebre)}"
                    f"  T:{len(taqui)}  H:{len(hipo)}",
                    flush=True,
                )

            content = fut.result()
            if not content:
                continue

            lines = content.strip().split("\n")
            if len(lines) < 2:
                continue
            header = lines[0].split("|")
            try:
                hi = header.index("HR")
                oi = header.index("O2Sat")
                ti = header.index("Temp")
            except ValueError:
                continue

            for line in lines[1:]:
                p = line.split("|")
                try:
                    hr = float(p[hi])
                    o2 = float(p[oi])
                    tmp = float(p[ti])
                except (ValueError, IndexError):
                    continue
                if any(np.isnan(v) for v in (hr, o2, tmp)):
                    continue
                if o2 < _C19_SPO2_MIN or o2 > 100.0:
                    continue

                row = {"BPM": hr, "SpO2": o2, "Temp": tmp}
                # Hipotermia tiene prioridad (temperatura es la señal más distintiva)
                if tmp <= _C19_HIPO_TEMP_MAX:
                    hipo.append({**row, "Label": 5})
                elif tmp >= _C19_FIEBRE_TEMP_MIN and hr >= _C19_FIEBRE_HR_MIN:
                    fiebre.append({**row, "Label": 3})
                elif (
                    _C19_TAQUI_HR_MIN <= hr <= _C19_TAQUI_HR_MAX
                    and _C19_TAQUI_TEMP_MIN <= tmp <= _C19_TAQUI_TEMP_MAX
                ):
                    taqui.append({**row, "Label": 4})

            if (
                len(fiebre) >= target_por_clase
                and len(taqui) >= target_por_clase
                and len(hipo) >= target_por_clase
            ):
                break

    print(f"\n  Revisados {done} pacientes")
    _c19_save(fiebre, 3, "fiebre", target_por_clase)
    _c19_save(taqui, 4, "taquicardia", target_por_clase)
    _c19_save(hipo, 5, "hipotermia", target_por_clase)


# ===========================================================================
# ORQUESTADOR
# ===========================================================================


def download_data():
    print("\n" + "█" * 60)
    print("  PREPARAR DATOS — iHealth32 (6 clases)")
    print("█" * 60)
    print("  Clases (6):")
    for k, v in LABEL_NOMBRES.items():
        print(f"    [{k}] {v}")
    print()

    os.makedirs(CARPETA, exist_ok=True)

    old = [f for f in os.listdir(CARPETA) if f.endswith(".csv")]
    if old:
        print(f"  Borrando {len(old)} CSVs anteriores...")
        for f in old:
            os.remove(os.path.join(CARPETA, f))

    download_nsrdb(n_records=15)
    download_afdb_fa(n_records=15)
    download_apnea_ecg_hipoxia(n_records=15)
    download_challenge2019(target_por_clase=6000)

    csvs = [f for f in os.listdir(CARPETA) if f.endswith(".csv")]
    total_filas = 0
    conteo = {i: 0 for i in range(NUM_CLASES)}

    for f in csvs:
        df = pd.read_csv(os.path.join(CARPETA, f))
        if "Label" in df.columns and len(df):
            lbl = int(df["Label"].iloc[0])
            if lbl in conteo:
                conteo[lbl] += len(df)
                total_filas += len(df)

    print("\n" + "═" * 60)
    print("  RESUMEN FINAL")
    print("═" * 60)
    for k, v in LABEL_NOMBRES.items():
        bar = "█" * (conteo[k] // 5000)
        print(f"  [{k}] {v:<25} {conteo[k]:>8} filas  {bar}")
    print(f"\n  Total: {total_filas:,} filas en {len(csvs)} ficheros CSV")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    download_data()
