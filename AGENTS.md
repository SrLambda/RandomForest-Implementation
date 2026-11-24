# Contexto e instrucciones del proyecto

Este archivo conserva el contexto técnico y los acuerdos de colaboración para
las siguientes sesiones. Las instrucciones explícitas más recientes del usuario
tienen prioridad. Comprobar el estado real de los archivos antes de trabajar:
la descripción del estado puede quedar desactualizada.

## Objetivo y alcance

- Proyecto de Root Cause Analysis (RCA) en microservicios: identificar y ordenar
  los servicios candidatos a causa raíz, distinguiendo origen y efectos en cascada.
- Integrantes: Benjamin Arancibia y Camilo Fuentes.
- Dataset previsto: RCAEval, https://github.com/phamquiluan/RCAEval.
- Selección acordada dentro de RCAEval: `RE2-SS`. Centrar la inspección y el
  desarrollo en esta selección. Verificar su organización real antes de asumir
  que corresponde a un único incidente o de definir grupos `incident_id`.
- Repositorio base: https://github.com/SrLambda/RandomForest-Implementation.
- Usar exclusivamente métricas numéricas: CPU, memoria, latencia, red/sockets
  y errores. No incorporar logs ni trazas como entradas del modelo en esta etapa.
- Herramientas: Python, NumPy, Pandas, scikit-learn y biblioteca estándar `time`.
- Entorno Linux local, Python 3.11 o superior según el README. Docker es opcional
  para empaquetar al final, no un requisito para iterar.

## Decisiones arquitectónicas acordadas

### Tiempo y ventanas dinámicas

- Resolución objetivo de un segundo. Verificar la frecuencia nativa del dataset;
  remuestrear no crea observaciones reales. Documentar agregación y faltantes.
- Mantener una ventana NORMAL de histórico sano reciente, sin contaminación
  por muestras anómalas.
- Un umbral estadístico histórico, por ejemplo Z-score, abre automáticamente
  una ventana ANÓMALA. Cerrarla tras N segundos consecutivos de recuperación
  o al alcanzar un máximo de duración.
- Alcanzar el máximo no significa recuperación ni autoriza incorporar datos
  todavía anómalos al baseline.
- Contemplar calentamiento, desviación estándar cero, huecos temporales,
  faltantes, recuperación y anomalías persistentes.
- Usar timestamps del dataset para ventanas y tiempo de detección;
  `time.perf_counter()` para medir duraciones de ejecución y procesamiento.
  La reproducción con pausas de un segundo es opcional, separada del entrenamiento.

### Formulación del problema: enfoque B

- Una muestra tabular por `(incident_id, service)`.
- Etiqueta binaria: `y=1` si el servicio es causa raíz, `y=0` para los demás.
- Incluir features locales y señales relativas/de contexto respecto a los demás
  servicios del incidente para distinguir causa raíz y cascada.
- Conservar identificadores para agrupar y rankear; no utilizarlos como features.
- No utilizar etiquetas, metadatos de inyección ni información futura como
  variables predictoras. Definir el instante de diagnóstico antes de extraer features.
- Aclarar con los datos reales cómo tratar múltiples causas, servicios ausentes
  y la asociación entre ventanas detectadas e incidentes etiquetados.

### Validación, balanceo y ranking

- Usar `GroupKFold` agrupado por `incident_id`: un incidente no puede aparecer
  simultáneamente en train y validación.
- Dividir primero los folds y aplicar undersampling de la clase mayoritaria
  (`0`) exclusivamente dentro del train de cada fold.
- Nunca submuestrear validación/test: puntuar todos sus servicios por incidente.
- Ajustar transformaciones aprendidas, selección de features e hiperparámetros
  sin usar validación/test indebidamente. No ajustar decisiones con el test final.
- Rankear mediante puntuaciones de `predict_proba`; evaluar Top-1, Top-3 y MRR
  por incidente. El submuestreo puede afectar la calibración: no asumir que las
  puntuaciones son probabilidades calibradas.
- Respetar también la separación experimental al ajustar el detector.
  Reportar incidentes no detectados y falsas alarmas, además del ranking.

## Organización del código

Aplicar POO con clases modulares de responsabilidad clara, código tipado,
documentado y sin lógica principal trasladada al notebook.

| Archivo | Responsabilidad |
| --- | --- |
| `config/topology.json` | Topología, métricas y parámetros externos modificables |
| `src/ingestion.py` | Adaptación de datos, unidades y timestamps |
| `src/window_detector.py` | Baseline sano y estados de ventanas |
| `src/feature_builder.py` | Features locales y relativas; muestras y grupos |
| `src/model.py` | Wrapper de Random Forest, entrenamiento y puntuaciones |
| `src/evaluation.py` | División por incidentes y submuestreo exclusivo de train |
| `src/metrics.py` | Métricas de ranking y tiempos |
| `scripts/run_pipeline.py` | Orquestación y punto de entrada |
| `notebooks/Untitled.ipynb` | Apoyo exploratorio inicial |
| `tests/` | Verificaciones de los contratos y comportamiento |

`requirements.txt` contiene dependencias de ejecución y `requirements-dev.txt`
añade herramientas de desarrollo. `time` no requiere instalación.

## Estado al crear este contexto

Actualización: se inspeccionaron los 90 casos locales de RE2-SS en
`data/raw/RCAEval`. Cada uno contiene 1441 timestamps separados por 1 s y
720 muestras anteriores a la inyección. Hay faltantes y entre 74 y 82 métricas.
`src/ingestion.py` define `IncidentMetadata`, `MetricTimeSeries` y
`RCAEvalLoader`, con lectura separada de índice/metadatos y métricas Parquet.
El detector, features y modelo siguen pendientes. Los puntos del esqueleto
descritos debajo corresponden al estado inicial, anterior a esta actualización.

Actualización de configuración: `config/topology.json` usa schema_version 3
y el inventario real de columnas de los 90 casos RE2-SS. Incluye todos los
componentes observados, métricas opcionales según disponibilidad y nombres
originales. Unidades y dependencias aún no verificadas (`null`); no inferir
aristas a partir de las métricas. La versión simplificada usa `required_metrics`
y `optional_metrics` por servicio, columnas `{service}_{metric}` y un mapa
`metric_families`. Se eliminaron conteos y campos descriptivos redundantes.
Las unidades pendientes se documentan aquí y en README, sin convertir valores.
Los parámetros del detector siguen sin validar. Se retiraron
los pesos y alert_on_spike del ejemplo inicial. No existe aún un lector de
esta configuración conectado al pipeline en aquella etapa.

Actualización posterior: `src/configuration.py` define `TopologyConfig`, que
valida schema_version 3, parámetros, servicios, familias y políticas soportadas.
`RCAEvalLoader(configuration=config)` aplica rutas y archivos configurados y
comprueba columnas requeridas/desconocidas. Las opcionales ausentes no se crean
ni se rellenan. Las rutas relativas se resuelven desde el padre de `config/`.
La forma anterior `RCAEvalLoader()` sigue disponible sin validación de topología.
El script de entrada sigue provisional y no hay detector ni modelo implementados.

- Existe el esqueleto modular; los módulos contienen documentación de
  responsabilidades, sin implementación del pipeline ML.
- El script de entrada únicamente informa del estado provisional.
- Los archivos de pruebas describen casos pendientes; todavía no implementan
  pruebas de comportamiento. No presentar comprobaciones de sintaxis como
  evidencia de que el pipeline funciona.
- `config/topology.json` es un ejemplo parcial, no un esquema verificado de
  RCAEval. Sus nombres, unidades y relaciones requieren contrastarse con datos.
- Parámetros iniciales: intervalo de 1 s, baseline de 60 s, Z-score de 3,
  recuperación de 4 s y máximo de ventana anómala de 120 s. No están validados.
- `alert_on_spike`, `weight` y la política de referencias externas todavía
  no tienen lógica implementada.
- La siguiente dependencia técnica es inspeccionar un incidente real y sus
  metadatos antes de implementar ingesta y features.

## Reglas de colaboración y entrega

- Comunicarse en español y explicar brevemente decisiones y cómo verificarlas.
- Antes de código complejo o asumir columnas de RCAEval, inspeccionar una muestra
  real disponible; si falta, solicitarla o aclarar la información necesaria.
- No introducir nuevas decisiones de arquitectura como si ya estuvieran acordadas.
- Respetar las modificaciones del usuario y limitar cambios al alcance solicitado.
- Preferencia actual: no añadir pruebas automatizadas; comprobar los cambios
  directamente con datos locales. No hacer commits mientras se trabaja en
  metadatos; el usuario sigue gestionando Git personalmente.
- El usuario gestiona Git: no ejecutar operaciones de Git ni crear ramas,
  commits o publicaciones. Si lo solicita, proporcionar el nombre de rama y
  mensaje del commit para que los ejecute personalmente.
- El README es la carta de presentación del repositorio. Mantenerlo conciso;
  no agregar planes detallados de trabajo sin petición explícita. Los planes
  se presentan en la conversación.
- Mantener datasets, modelos y resultados pesados fuera de Git. Los directorios
  `data/`, `artifacts/`, `models/`, `results/` y `reports/` están excluidos.
- Mantener el notebook como exploración complementaria y revisar sus salidas
  antes de publicarlo.

## Comprobaciones disponibles

```bash
python3 scripts/run_pipeline.py
python3 -m compileall -q src scripts tests
python3 -m json.tool config/topology.json > /dev/null
```

Hay pruebas previas del contrato de ingesta en `tests/test_ingestion.py`.
No ampliarlas salvo nueva indicación del usuario. Para verificar la carga actual,
usar `RCAEvalLoader` con los archivos locales y comprobar que se conservan
los faltantes y la separación de etiquetas y métricas.
