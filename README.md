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
src/configuration.py           Lectura y validación de la configuración JSON
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

`config/topology.json` contiene el inventario de columnas observado en los 90
casos locales de RE2-SS (Sock Shop). Incluye CPU, memoria, latencia, errores,
sockets, disco y carga. Conserva los nombres originales sin convertir unidades,
que todavía están pendientes de verificar. Las métricas no presentes en todos
los casos figuran en `optional_metrics`; las comunes, en `required_metrics`.
Las columnas se identifican con el patrón `{service}_{metric}` y su ausencia
no significa valor cero. `metric_families` agrupa los sufijos por familia.

`upstream` y `downstream` son `null`: las dependencias no están verificadas,
lo que no equivale a afirmar que no existen. El inventario incluye todos los
componentes observados y no restringe candidatos a los servicios con fallos
inyectados.

El detector contempla resolución objetivo de 1 segundo, baseline de 60 segundos,
umbral Z-score de 3, recuperación de 5 segundos consecutivos y máximo de ventana
anómala de 120 segundos.
Estos parámetros son iniciales y todavía no están calibrados. `TopologyConfig`
valida el JSON y permite conectar su configuración con `RCAEvalLoader`:

```python
from src.configuration import TopologyConfig
from src.ingestion import RCAEvalLoader

config = TopologyConfig("config/topology.json")
loader = RCAEvalLoader(configuration=config)
incident_id = loader.list_incidents()[0]
metadata = loader.load_metadata(incident_id)
metrics = loader.load_metrics(incident_id)
```

Las rutas relativas de datos se resuelven desde el directorio padre de `config/`.
La carga configurada rechaza columnas desconocidas y requeridas ausentes;
conserva las opcionales ausentes sin crearlas ni rellenarlas. El punto de
entrada `run_pipeline.py` sigue siendo provisional: no ejecuta detección ni ML.

## Detector de ventanas

`WindowDetector` procesa cada caso de forma independiente: calentamiento de
60 segundos, estados NORMAL/OBSERVATION/ALERT, confirmación tras 5 segundos
irregulares y recuperación tras 5 segundos sanos. El baseline se congela ante
sospechas. Se utiliza Z-score bilateral; con desviación cero, un cambio del valor
constante se considera anómalo. Son reglas iniciales pendientes de calibración.

Una anomalía observable puede confirmar alerta aunque falten otras métricas.
Para acreditar salud se exige que todas las métricas presentes sean evaluables;
si no, la observación es UNKNOWN e interrumpe las seguidillas. Las columnas
opcionales ausentes no se agregan. Los faltantes persistentes pueden impedir
la recuperación automática y deben revisarse antes de calibrar el detector.

Ejecutar desde la raíz del repositorio, con el entorno activado:

```bash
python -m scripts.run_window_detector --incident re2ss_carts_cpu_1
```

La consola muestra ALERT_CONFIRMED inmediatamente y DIAGNOSIS_DUE 10 segundos
después, aunque el incidente ya se haya recuperado. Este último evento contiene
baseline y observaciones disponibles hasta ese instante para el futuro modelo;
todavía no produce un ranking. No consulta causas raíz ni tiempos de inyección.
En OBSERVATION se conservan como máximo las últimas 120 muestras sospechosas;
el evento también conserva el timestamp inicial de la sospecha.

El máximo de ventana segmenta la alerta sin declarar recuperación ni generar
otro diagnóstico. Un archivo que acaba antes del diagnóstico produce
DIAGNOSIS_INCOMPLETE. El calentamiento, la observación y el último segmento
también pueden quedar sin completar; no se infiere recuperación al terminar
los datos. El script valida la partición y usa desarrollo por defecto.
La opción `--partition test` se reserva para la evaluación final; no usarla
para calibrar umbrales o elegir features.

## Partición de incidentes

`config/incident_split.json` fija 72 incidentes de desarrollo y 18 de test con
semilla 42. Test contiene 3 casos por cada tipo de fallo y 3 o 4 por servicio;
cada combinación servicio/fallo conserva al menos dos casos en desarrollo.
La selección utiliza metadatos, sin consultar métricas ni resultados del modelo.
Los datos no se mueven: el manifiesto solo almacena identificadores y debe
versionarse para mantener la misma evaluación.

```bash
python -m scripts.create_incident_split
```

El comando reutiliza y valida el manifiesto si existe, sin sobrescribirlo ni
aplicar otra semilla. La huella de metadatos detecta cambios de identidad o
etiquetas; no verifica el contenido de los archivos de métricas.
`IncidentSplit.load()` expone `development` y `test` para la futura orquestación.
GroupKFold se aplicará dentro de desarrollo. Validación/test conservan todos los
servicios y nunca reciben submuestreo. El caso `re2ss_carts_cpu_1` ya se había
inspeccionado antes de reservar test; no se utilizó para ajustar esta selección.
