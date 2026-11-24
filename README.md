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

Usar Python 3.11 o superior. Desde la raíz del proyecto, crear y activar
un entorno virtual local:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip check
```

Para ejecución sin herramientas de desarrollo, instalar `requirements.txt`.
Las dependencias de desarrollo incluyen las de ejecución, Jupyter, pytest y
`huggingface_hub` para descargar RCAEval. `pyarrow` permite leer Parquet.
Las versiones todavía no están fijadas; se fijarán tras validar el entorno.

Comprobar el intérprete y las dependencias:

```bash
python -c "import sys; print(sys.executable)"
python -c "import numpy, pandas, sklearn, pyarrow, huggingface_hub; print('Dependencias disponibles')"
```

El intérprete debe estar dentro de `.venv/`. En nuevas terminales, ejecutar
`source .venv/bin/activate`; para salir del entorno, ejecutar `deactivate`.
En VS Code, seleccionar `.venv/bin/python` como intérprete y como entorno
del kernel del notebook.

Si en Ubuntu/Debian falla la creación del entorno por falta de `venv`, instalar
`sudo apt install python3-venv` y repetir la creación. Para una versión de Python
instalada por separado, usar su paquete `venv` correspondiente.

`.venv/`, los datos y los archivos locales `.env` están excluidos de Git;
los archivos `requirements*.txt` sí deben versionarse. No se necesita un archivo
`.env` para crear el entorno o descargar este dataset público.

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
