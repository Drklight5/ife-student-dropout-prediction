# IFE Student Dropout Prediction

Proyecto de predicción de deserción estudiantil para el curso **Desarrollo de Aplicaciones Avanzadas de Ciencias Computacionales** (ITESM, Sem. 8). El objetivo es predecir qué estudiantes universitarios desertarán antes de su segundo año, e investigar si los perfiles de riesgo son invariantes entre los modelos educativos **PreTec21** (cohortes AD14–AD18) y **Tec21** (cohortes AD19–AD20).

---

## Pregunta de Investigación

> ¿Existe un perfil de deserción invariante ante el cambio pedagógico entre modelos pre-Tec21 y Tec21? 
>> ¿Es posible identificar y caracterizar estos perfiles mediante clustering sobre las variables existentes en el conjunto de datos? ¿Cómo son afectadas las variables dictadas por la agrupación en su relación predictiva a la deserción al transitar entre modelos educativos?



Se formula como **clasificación binaria de retención** (`retention = 1` se quedó, `0` desertó), usando únicamente información disponible antes del inicio del primer semestre.

---

## Estructura del Repositorio

```
ife-student-dropout-prediction/
├── analysis/
│   └── exploration.ipynb      # Notebook principal ()
├── data/
│   ├── dataset.xlsx           # Dataset original (143,326 × 50) — no versionado
│   ├── dataset_cache.csv      # Caché CSV generado automáticamente (carga ~10× más rápida)
│   ├── resultados_componente2.csv  # Tabla comparativa de métricas exportada por el notebook
│   └── README.md
├── requirements.txt
└── README.md
```

> **Nota:** `dataset.xlsx` y `dataset_cache.csv` están en `.gitignore` por tamaño. Coloca el archivo original en `data/` antes de ejecutar el notebook.

---

## Dataset

| Atributo | Valor |
|---|---|
| Fuente | Instituto de Futuro de la Educación (IFE), Tec de Monterrey |
| Observaciones | 143,326 estudiantes (50 variables) |
| Subconjunto usado | 77,517 universitarios de pregrado |
| Variable objetivo | `retention` (binaria: 0 = desertó, 1 = se quedó) |
| Desbalance | 8.12% deserción / 91.88% retención |
| Split temporal | PreTec21 (AD14–18) = entrenamiento · Tec21 (AD19–20) = prueba |

Las variables con missingness notable son:

- `admission_test_norm` (~1.5% + ~20% "Does not apply" para alumnos Tec) — imputada con mediana
- `admission.rubric` (~23%) — imputada con mediana
- `first.generation` (~48%) — patrón MNAR; se codifica con categoría `-1` para preservar la señal de ausencia

---

## Metodología

### Preprocesamiento
1. Filtro a nivel universitario (`level == 'Undergraduate'`)
2. Eliminación de columnas con fuga de datos (`dropout.semester`, `average.first.period`, etc.)
3. Normalización de `admission.test` (escala PAA 400–1600 y PAL 0–100 → 0–1)
4. Ingeniería de variables: `has_extracurriculars`, `educ_padres_max`, `apoyo_financiero`
5. Codificación ordinal/label de variables categóricas
6. Imputación final con `SimpleImputer(strategy='median')` ajustado solo en PreTec21

### Modelos (Componente 2)

| Modelo | Rol |
|---|---|
| **K-Means** | Perfiles latentes de riesgo; test de invarianza z-score entre modelos educativos |
| **Regresión Logística** | Línea base interpretable; odds ratios comparables entre cohortes |
| **Random Forest** | Captura interacciones no lineales; validación de importancias por Gini + permutación |

- Manejo de desbalance: `class_weight='balanced'` en todos los modelos supervisados
- Umbral de clasificación: selección OOF sobre `precision_recall_curve` (no data snooping)
- Evaluación final: bootstrap CI con 1000 iteraciones sobre el conjunto Tec21

---

## Instalación

### Requisitos
- Python 3.11+ (desarrollado con Python 3.14)
- Las dependencias están en `requirements.txt`

### Pasos

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd ife-student-dropout-prediction

# 2. Crear y activar el entorno virtual
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Colocar el dataset
# Copiar dataset.xlsx a data/dataset.xlsx

# 5. Lanzar Jupyter
jupyter notebook analysis/exploration.ipynb
```

> **Primera ejecución:** la carga del `.xlsx` tarda ~30 s y genera `data/dataset_cache.csv`. Las ejecuciones siguientes cargan el CSV en ~3 s.

---

## Dependencias Principales

| Paquete | Versión | Uso |
|---|---|---|
| `scikit-learn` | 1.8.0 | Modelos, CV, métricas |
| `pandas` | 3.0.3 | Manipulación de datos |
| `numpy` | 2.4.6 | Operaciones numéricas |
| `scipy` | 1.17.1 | Hungarian algorithm, estadística |
| `matplotlib` | 3.10.9 | Visualizaciones |
| `openpyxl` | (vía pandas) | Lectura de `.xlsx` |

---

## Resultados (Componente 2)

Los resultados finales se exportan a `data/resultados_componente2.csv` al ejecutar la última celda del notebook. El archivo incluye AUC-ROC, Recall, F1 e intervalos de confianza bootstrap al 95% para cada modelo evaluado en el conjunto Tec21.

---

## Curso

**Desarrollo de Aplicaciones Avanzadas de Ciencias Computacionales — Grupo 503**
Instituto Tecnológico y de Estudios Superiores de Monterrey
