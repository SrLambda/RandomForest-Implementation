# RCA de microservicios con Random Forest

**Integrantes:**

- **Benjamin Arancibia**
- **Camilo Fuentes**

## Objetivo y alcance

Identificar y ordenar los servicios candidatos a causa raíz de un incidente,
distinguiendo el origen del fallo de sus efectos en cascada mediante métricas
locales y señales relativas al resto de servicios.

- Dataset previsto: [RCAEval](https://github.com/phamquiluan/RCAEval).
- Modalidad: exclusivamente métricas numéricas de CPU, memoria, latencia,
  red/sockets y errores.
- Herramientas a utilizar: Python, NumPy, Pandas, scikit-learn y time.
- Entorno: Linux local; Docker opcional para el empaquetado final.

La formulación elegida es clasificación binaria por `(incident_id, service)`:
`y=1` para el servicio causa raíz y `y=0` para los demás. Las puntuaciones
permiten construir un ranking completo por incidente. Los identificadores se
conservan para agrupar y evaluar.

La configuración del sistema se centraliza en un archivo JSON. El notebook sirve como apoyo exploratorio y complementa los módulos del pipeline.

## Instalación local

Usar Python 3.11 o superior:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Para ejecución sin herramientas de desarrollo, instalar `requirements.txt`.

## Estructura

```text
config/topology.json           Topología y parámetros de ejemplo
src/ingestion.py               Adaptación de datos y timestamps
src/window_detector.py         Ventanas NORMAL/ANÓMALA y baseline sano
src/feature_builder.py         Features locales y relativas por servicio
src/model.py                   Entrenamiento y puntuaciones del Random Forest
src/evaluation.py              GroupKFold y submuestreo exclusivo de train
src/metrics.py                 Top-1, Top-3, MRR y tiempos
scripts/run_pipeline.py        Punto de entrada provisional
notebooks/Untitled.ipynb       Exploración inicial
tests/                         Contratos de pruebas pendientes
```

## Configuración y datos

`config/topology.json` es un ejemplo parcial. Incluye familias de CPU, memoria,
latencia, errores y red/sockets. `allow_external_service_references` expresa
que se permiten referencias a servicios fuera de la lista; su validación está
pendiente. `upstream` representa llamadores y `downstream` los servicios llamados.

El detector contempla resolución objetivo de 1 segundo, baseline de 60 segundos,
umbral Z-score de 3, recuperación de 4 segundos consecutivos y máximo de ventana
anómala de 120 segundos.
Los campos `alert_on_spike` y `weight` son propuestas sin lógica ejecutable.
