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

### Reglas del detector y diagnóstico acordadas antes de implementarlo

Estas decisiones sustituyen los valores anteriores cuando haya discrepancias.
Son requisitos acordados, no funcionalidad ya implementada ni parámetros
calibrados experimentalmente. Actualizar JSON y validador al implementar.

- Calentamiento inicial de 60 segundos por caso, bajo la hipótesis explícita
  de arranque sano. No consultar el instante de inyección para decidir el
  calentamiento, disparar alertas o seleccionar observaciones predictoras.
- Después del calentamiento, flujo de estados NORMAL → OBSERVACIÓN → ALERTA,
  con retorno a NORMAL tras recuperación sostenida.
- La primera muestra fuera del umbral inicia OBSERVACIÓN y congela el baseline.
- Confirmar ALERTA con 5 segundos consecutivos de irregularidad. La seguidilla
  es temporal: una misma métrica puede sostenerla; no se exige que sean varias
  métricas distintas simultáneamente.
- Durante OBSERVACIÓN y ALERTA no incorporar muestras sospechosas al baseline.
  Descartarlas también si la observación no llega a convertirse en alerta.
- Volver a NORMAL tras 5 segundos consecutivos dentro del umbral seguro,
  tanto al descartar una sospecha como al recuperar una alerta.
- Una muestra normal interrumpe la seguidilla de anomalía; una anómala
  interrumpe la de recuperación. Un dato faltante interrumpe ambas cuentas:
  no acredita salud, anomalía ni recuperación. Señalar la falta de información.
  Aún debe concretarse cómo agregar métricas parcialmente faltantes para
  clasificar una observación global; no interpretar una opcional ausente como cero.
- Avisar al usuario inmediatamente al confirmar ALERTA.
- Programar el primer diagnóstico/ranking RCA para 10 segundos después de
  confirmar ALERTA, utilizando solo información disponible hasta ese instante.
- Si hay recuperación antes de esos 10 segundos, mantener el diagnóstico
  programado e indicar que el incidente ya se recuperó.
- Emitir un primer ranking; actualizaciones periódicas quedan para otra etapa.
- Si se alcanza el máximo de ventana (valor inicial de 120 s) y persiste la
  anomalía, continuar en ALERTA. Se puede segmentar la recogida sin declarar
  recuperación ni contaminar el baseline.
- El sistema identifica causas candidatas; proponer o ejecutar remedios no
  pertenece al alcance actual. La recuperación depende de las métricas,
  no de que alguien haya anunciado una solución.

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
- Reparto acordado: 72 incidentes de desarrollo y 18 de test final. Reservar
  test antes de calibrar detector, features o modelo; aplicar GroupKFold dentro
  de desarrollo. Mantener todos los servicios de validación/test sin submuestreo.
- La selección concreta sigue pendiente: usar semilla fija, persistir los
  identificadores y procurar equilibrio por servicio causa raíz y tipo de fallo.
  Hay 5 servicios × 6 fallos × 3 repeticiones: reservar una repetición por cada
  combinación produciría 30 casos, por lo que no corresponde al reparto 72/18.

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

Las actualizaciones siguientes están ordenadas de más reciente a más antigua;
ante discrepancias, prevalece la primera actualización aplicable.

Preparación de datos para entrenamiento implementada en `src/data_preparation.py`.
`python -m scripts.prepare_training_data` consume development_folds y crea
data/processed/prepared_development_folds (directorio nuevo obligatorio).
Opciones: --source, --output, --negative-ratio (1 por defecto) y --seed (42).
TrainMedianImputer aprende medianas de TODAS las filas asociadas de train ANTES
del submuestreo. Columnas completamente NaN en train usan constante 0, registrada
en imputer.json; este valor no acredita salud ni usa información de validación.
No cambia el esquema de 153 features ni los indicadores existentes. Transform
no reajusta valores; el estado JSON puede restaurarse con from_state.
El submuestreo conserva todos los positivos y elige negativos sin reemplazo,
reproducible con semilla fija. La proporción indica el máximo de negativos por
positivo, redondeado hacia arriba; no duplica filas si faltan negativos.
Validación conserva todas sus filas, servicios, etiquetas y orden, sin balanceo.
Cada fold exporta X/y/identifiers/groups, source_rows para rastrear posiciones
en origen, estado del imputador y manifiesto con procedencia/auditorías de origen.
Se comprueban IDs, partición, calibración train y coherencia con las auditorías.
Ejecución local de los cinco folds: train 841→112, 782→104, 796→106, 797→106,
797→106; proporción 1:1. Validación completa 211/195/196/210/195 = 1.007 filas.
Verificación directa: medianas coinciden exactamente con train original, sin
NaN/infinitos en salida, etiquetas/IDs alineados, sin repetición de filas,
positivos conservados y selección reproducible. Comprobados respaldo de columna
vacía, serialización y proporciones inválidas con fragmentos manuales, sin
añadir pruebas automatizadas ni tocar Git. No se consultaron métricas de test.
Siguiente tarea: Random Forest y ranking; la cobertura de validación sigue en
67/72 y las limitaciones de CV exploratoria descritas abajo permanecen.

Preparación por folds implementada: `src/cross_validation.py` y
`python -m scripts.prepare_folds [--n-splits 5] [--output <directorio-nuevo>]`.
GroupKFold determinista sobre los 72 IDs ANTES del detector, incluyendo omisiones;
cada caso aparece exactamente una vez en validación. No se exige equilibrio por
tipo de fallo o servicio en estos folds. `fit_deviation_floors` acepta
`fit_incidents` explícitos; sin ese argumento conserva el comportamiento 48/24.
Cada fold elimina la calibración global de su configuración inicial, ajusta
tolerancias exclusivamente con su train y regenera ventanas/features de train
y validación con esa misma calibración. TopologyConfig.validate_fold_calibration
exige coincidencia exacta de IDs y huella; la asociación acepta roles del fold
solo tras esta comprobación. Guardados folds.json, calibraciones, snapshots,
datasets y auditorías locales en data/processed/development_folds (sin sobrescribir).
Ejecución de cinco folds: train 57/57/58/58/58 y validación 15/15/14/14/14;
casos asociados de validación 14/13/13/14/13 = 67/72, 1.007 filas y 3 falsas
alertas preinyección. No comparar esta cobertura como mejora/retroceso controlado
frente al 71/72 global: cambia el conjunto usado para ajustar tolerancias.
Se verificaron manualmente los diez datasets: alineación, 153 features comunes,
ausencia de infinitos/IDs en X, una causa raíz por caso, sin pares duplicados,
grupos disjuntos, cobertura de auditorías y ventanas seleccionadas postinyección.
Se reprodujo la calibración del fold 1 registrando que solo leyó sus 57 train.
No se leyeron métricas de test, no hay pruebas nuevas ni operaciones Git.
Este paso prepara datos; NO incluye imputación, undersampling ni modelo.
Pendientes esos pasos, ranking, integración y test final. La CV sigue siendo
exploratoria: las reglas del detector fueron elegidas previamente usando
desarrollo. El ajuste por fold evita compartir tolerancias aprendidas pero no
elimina esa selección histórica. No afirmar independencia estadística completa.

Implementación vigente de asociación offline (2026-11-25): `src/window_association.py`
asocia los eventos del detector con los incidentes etiquetados, y
`python -m scripts.build_dataset` construye el dataset de desarrollo. Por caso,
selecciona como máximo el primer `DIAGNOSIS_DUE` completo cuyo inicio original
de sospecha y primera observación sean ambos posteriores o iguales a la
inyección. La inyección se usa solo para esta asociación y para generar `y`,
nunca para disparar el detector ni como feature. Conserva todos los servicios
observados, sin imputar ni submuestrear. `incident_audit.csv`, `window_audit.csv`
y `event_audit.csv` incluyen omisiones, falsas alertas, diagnósticos adicionales
e incompletos; la ausencia de una muestra supervisada no se oculta. La salida se
crea en un directorio nuevo bajo `data/processed/` y guarda X, y, grupos,
identificadores y snapshots de configuración/partición. Verificación local:
71/72 casos asociados, 1.067 filas, 153 features, una causa raíz por caso
asociado, 4 diagnósticos preinyección y 10 diagnósticos adicionales
postinyección. Es un dataset exploratorio condicionado a ventanas asociables,
no una evaluación integral ni apta aún para validación independiente: antes de
GroupKFold hay que recalibrar el detector dentro del train de cada fold y
regenerar ventanas/features. Los 18 casos de test no se usaron.

Etapa vigente: el usuario decidió posponer el perfeccionamiento del detector y
continuar con features. No reanudar ajustes de falsas alertas salvo nueva petición.
`src/feature_builder.py` implementa FeatureBuilder y FeatureBatch(X, identifiers),
con groups=incident_id. build recibe únicamente DIAGNOSIS_DUE, baseline y datos
hasta el diagnóstico; rechaza futuro/solapamiento y no consulta etiquetas.
Esquema v1 de 153 variables: 16 estadísticas por cada uno de los 8 sufijos,
resúmenes locales y contexto entre servicios (cambios normalizados, severidad,
percentiles, exceso frente a pares y orden de desviación). Los scores resumen
evidencia estadística disponible, no prueban causalidad ni salud de datos faltantes.
Métricas ausentes conservan NaN con indicadores, componentes totalmente ausentes
no se inventan; todos los servicios observados permanecen como candidatos.
El contexto se calcula ANTES de submuestrear train. No hay imputación aprendida
ni transformaciones ajustadas en el builder; ajustar cualquier imputador solo en train.
assign_root_cause_labels es una función offline separada: rechaza ventanas que
empiezan antes de inyección y exige causa raíz entre candidatos. No poner y/IDs,
inyección, tipo de fallo o causa raíz en X.
`python -m scripts.build_features --incident <case> [--labels] [--output <nuevo-dir>]`
exporta X, identifiers, y opcional y manifest a data/processed/features/<case>.
Solo desarrollo, primer diagnóstico completo seleccionado SIN consultar etiquetas;
sin diagnóstico produce mensaje explícito, no una muestra ficticia. No sobrescribe.
Ejemplo local generado: re2ss_carts_cpu_1, 15 servicios ×153 features, y=1/14.
Revisión de los 72 casos: 69 primeras ventanas etiquetables, 2 primeras ventanas
preinyección sin etiqueta y 1 caso sin diagnóstico. No confundir ese 69 con la
cobertura 71 del detector: puede haber diagnósticos válidos posteriores en casos
con falsa alerta inicial. Reporte en reports/features/development_validation.*.
La asociación de ventanas y construcción del dataset se documenta en la
actualización vigente superior. No se entrenó Random Forest. GroupKFold,
undersampling y ranking siguen pendientes.
Mantener la obligación de recalibrar detector/tolerancias dentro de train de
cada fold futuro; los features actuales son exploratorios con el detector vigente.
Se verificaron directamente esquema común, valores locales/contexto, NaN sin
infinitos, IDs separados, etiquetas y causalidad. No hay pruebas nuevas ni Git.

Revisión vigente de omisiones de desarrollo: detector equilibrado con referencia
fija adicional para aumentos acumulados de CPU (`fixed_reference_families=["cpu"]`)
y señal de aumento relativo de latencia (`relative_increase_factor_by_family`,
latency=10). La razón relativa solo se usa si hay >=2 valores finitos, escala
positiva, CV <= 0.25 (`relative_reference_max_cv`) y escala inicial >=
`minimum_reference_scale` aprendida con los mismos 48 casos de ajuste.
Ese piso es el cuantil 0.05 de max(abs(media),std) del calentamiento por métrica;
se añade a detector_calibration.json SIN cambiar tolerancias absolutas ni IDs
48/24. El detector no recibe etiquetas, inyección ni muestras futuras.
La referencia fija solo busca aumentos; las caídas desde un arranque cargado no
activan esa señal, aunque el detector móvil conserva las reglas de caídas.
Resultados actuales: 71/72 detecciones, 1 omisión, 4 falsas alertas en 2 casos,
retraso mediano 21 s entre detectados. Se recuperan carts_cpu_1, carts_delay_1/2/3,
carts_loss_1 y orders_cpu_1, sin perder ninguna detección previa. No aumentaron
las falsas alertas ni su ocupación preinyección (~0.17%). Los tiempos 60/5/5/10
siguen iguales. Reportes: reports/detector/development_review/ y
reports/detector/reference_increase/; comparación por IDs y diagnóstico de las
7 omisiones originales en comparison.json/original_miss_diagnostics.csv.
El evaluador genera miss_diagnostics.csv con métricas de las omisiones actuales.
Queda re2ss_carts_socket_1: calentamiento variable (CV de carts_latency-50 ~0.73)
y rangos solapados antes/después de inyección; no afirmar que es indetectable ni
forzar una excepción por caso. La hipótesis de arranque sano no garantiza
estabilidad real. Aceptar esta limitación documentada como base provisional antes
de features, salvo nuevas decisiones del usuario. Se compararon variantes en
desarrollo; estas cifras NO son independientes, ni resultados de test.
No se leyeron métricas de test. Verificaciones manuales de crecimiento gradual,
latencia relativa, referencias inválidas, faltantes, eventos y causalidad por
prefijos; calibración reproducible. No se añadieron pruebas ni se ejecutó Git.

Refinamiento vigente del detector: se analizaron las falsas alertas restantes y
se añadieron reglas configurables por familia (sin excepciones por incidente).
Configuración activa equilibrada en topology.json: anomaly_direction_by_family
latency=increase; minimum_deviation_multiplier_by_family latency=1.3;
decrease_deviation_multiplier_by_family network_sockets=1.8. El incremento de
sockets conserva el umbral anterior; caídas grandes siguen pudiendo alertar.
Las demás familias conservan dirección bilateral. Los mapas son opcionales y
se validan; sin ellos se reproduce el comportamiento previo. La pertenencia
de columnas a familias se obtiene del JSON, no separando nombres heurísticamente.
Las tolerancias aprendidas de 48 casos, el warmup y los tiempos 5/5/10 no cambiaron.
Verificación en 72 casos de desarrollo: falsas alertas 9→4, casos afectados 3→2,
ocupación normal en ALERT 1.63%→0.17%, detecciones nuevas 65/72 y 7 omisiones.
Misma cobertura agregada pero NO mismos casos: se pierde re2ss_carts_delay_3 y
se recupera re2ss_catalogue_loss_2. Retraso mediano 22 s entre detectados.
Reportes locales: family_refined/ (configuración activa), family_asymmetric/,
family_comparison.json (comparación por caso y advertencias).
La variante con multiplicador latency=1.5 y el mismo margen asimétrico de sockets
produce 0 falsas alertas en desarrollo pero solo 64 detecciones (8 omisiones);
está en reports/detector/family_asymmetric_strict/, no activada por defecto.
Se solicitó preferencia opcional al usuario; ante ausencia de respuesta se dejó
la variante equilibrada. Una instrucción posterior puede cambiar esa elección.
No afirmar eliminación total ni garantías fuera de desarrollo: estas variantes
se eligieron tras inspeccionar sus resultados, incluso comprobación interna.
Los 18 casos de test no se evaluaron. No se añadieron pruebas automatizadas ni
se ejecutó Git. Se comprobaron manualmente direcciones, eventos y causalidad.

Evaluación y reducción de falsas alertas implementadas: `src/detector_evaluation.py`
y `python -m scripts.evaluate_detector` evalúan SOLO los 72 casos de desarrollo.
Reportes locales excluidos de Git en `reports/detector/`: baseline, variantes,
comparación, CSV por caso/evento/fallo y JSON con instantáneas reproducibles.
No se leyeron métricas de los 18 casos de test para ajustar o evaluar.
`src/detector_calibration.py` y `python -m scripts.calibrate_detector` generan
tolerancias offline con datos normales preinyección de 48 casos de desarrollo;
otros 24 se reservan para comprobación interna (4 por fallo, semilla 42).
Archivo seleccionado: `config/detector_calibration.json`, referenciado por
`detector_calibration_file` en topology.json, ruta relativa al directorio del JSON.
Tolerancia = 2 × cuantil 0.999 de desvíos normales respecto al promedio de warmup
por columna. Umbral efectivo max(3×std, tolerancia), sin inyección en el detector.
84 métricas calibradas; istio-init_mem/socket sin calibrar conservan la regla
original. Validar IDs de ajuste/comprobación contra la partición antes de ejecutar.
El archivo de calibración no se sobrescribe automáticamente.
Resultados desarrollo antes/después: casos con falsa alerta 72→3, total falsas
alertas 72→9, fracción preinyección en ALERT 99.29%→1.63%, detecciones nuevas
postinyección 0→65/72; 7 casos sin detección nueva. Retraso mediano 22 s entre
detectados. En comprobación interna: 3/24 casos con falsas alertas, 22/24 con
nueva detección. Se compararon alternativas usando desarrollo; no presentar
comprobación interna como estimación independiente ni afirmar eliminación total.
re2ss_carts_cpu_1 dejó de dar falsa alerta pero tampoco detecta el fallo con este
ajuste. No hay etiqueta de fin de fallo para validar recuperación real.
Pendiente analizar las falsas alertas/omisiones restantes; no ocultar el compromiso
sensibilidad/falsas alarmas. Para el futuro GroupKFold del modelo, recalibrar
cualquier tolerancia aprendida exclusivamente con train del fold: la calibración
global actual es exploratoria y NO debe reutilizarse como si fuera independiente.
Verificaciones manuales: ajuste reproducible, separación de test, eventos 5/5/10,
faltantes y causalidad de las ventanas. No se añadieron pruebas automatizadas.

Partición ya persistida: `config/incident_split.json`, semilla 42, 72 desarrollo
y 18 test. `src/evaluation.py` implementa `IncidentSplit.create/load/validate/save`;
`python -m scripts.create_incident_split` crea el manifiesto una sola vez o valida
el existente sin sobrescribirlo. Método cíclico con servicios/fallos permutados,
una repetición elegida por pareja seleccionada: test tiene 3 casos por fallo y
3/4 por servicio, conservando al menos 2 repeticiones de cada pareja en desarrollo.
Selección solo con metadatos; no se consultan métricas ni calidad del detector.
La huella detecta cambios de IDs/etiquetas, no contenido de métricas.
El script del detector restringe casos a desarrollo por defecto; `--partition test`
es exclusivamente para evaluación final. No regenerar la reserva según resultados.
La implementación de GroupKFold, undersampling y entrenamiento sigue pendiente.
Esta actualización sustituye las menciones históricas a partición pendiente.

Estado vigente del detector (sustituye las referencias históricas a pendiente):
`src/window_detector.py` implementa `WindowDetector`, estados WARMUP/NORMAL/
OBSERVATION/ALERT y actualizaciones/eventos por muestra. JSON y validador incluyen
warmup_sec=60, anomaly_consecutive_sec=5, recovery_consecutive_sec=5 y
diagnosis_delay_sec=10. Máximo de segmento 120 s sin recuperar automáticamente.
`python -m scripts.run_window_detector --incident <case>` permite inspeccionar
un caso sin etiquetas. DIAGNOSIS_DUE contiene baseline previo a sospecha y
observaciones hasta el diagnóstico, pero no ejecuta ranking ni modelo.
El buffer de sospecha retiene como máximo max_anomaly_window_sec muestras;
conserva por separado el inicio original de la sospecha. Al terminar el archivo,
se informa diagnóstico/segmento/calentamiento/observación incompletos cuando corresponda.
Política inicial revisable para faltantes parciales: anomalía observable prevalece;
salud exige todas las columnas presentes evaluables. UNKNOWN interrumpe ambas
seguidillas y congela baseline. Opcionales ausentes no participan. Z-score
bilateral, desviación poblacional, mínimo 2 valores finitos por métrica y cambios
desde desviación cero con rtol=1e-9/atol=0 considerados anómalos. Estas decisiones
no están calibradas: faltantes persistentes pueden bloquear recuperación y
cambios pequeños en métricas constantes pueden producir alertas. Features,
modelo y partición persistida de 72/18 siguen pendientes. Se verificaron reglas
con secuencias controladas y ejecución de un caso real, sin medir calidad RCA
ni ajustar parámetros con los 90 casos. No se añadieron pruebas automatizadas.
La ejecución de integración de re2ss_carts_cpu_1 disparó ALERT_CONFIRMED en
1705595523, antes de la inyección registrada 1705596179: existe una falsa alarma
con los valores iniciales. No presentar el detector como calibrado ni efectivo;
fijar la partición 72/18 antes de ajustar reglas con datos de desarrollo.

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
