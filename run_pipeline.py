"""
run_pipeline.py
===============
Ejecuta el pipeline completo de predicción de deserción estudiantil IFE,
desde la imputación de datos hasta el análisis estadístico reproducible.

Uso:
    python run_pipeline.py              # corre todo (saltando pasos ya completados)
    python run_pipeline.py --force      # fuerza re-ejecución de todos los pasos
    python run_pipeline.py --from 3     # empieza desde el paso 3
    python run_pipeline.py --only 5     # corre solo el paso 5

Pasos del pipeline:
    1. Imputación de datos crudos              (data/preprocessing/02_imputation_pipeline.ipynb)
    2. Feature engineering                     (analysis/clustering/01_preprocessing_feature_engineering.ipynb)
    3. K-Means por régimen                     (analysis/clustering/02_kmeans_independiente.ipynb)
    4. Entrenamiento Random Forest             (analysis/modelos/experimentos/random_forest.ipynb)
    5. Análisis estadístico reproducible       (analysis/modelos/analisis_estadistico_reproducible.ipynb)

Prerrequisitos:
    - Python 3.9+
    - jupyter, nbconvert: pip install jupyter nbconvert
    - Paquetes científicos: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, shap
    - El archivo data/dataset.csv debe existir antes de correr el paso 1
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# Configuración del pipeline 

ROOT = Path(__file__).parent.resolve()

STEPS = [
    {
        "id": 1,
        "name": "Imputación de datos crudos",
        "notebook": ROOT / "data/preprocessing/02_imputation_pipeline.ipynb",
        "run_from": ROOT / "data/preprocessing",
        "requires": [ROOT / "data/dataset.csv"],
        "produces": [ROOT / "dataset_imputed.csv"],
    },
    {
        "id": 2,
        "name": "Feature engineering",
        "notebook": ROOT / "analysis/clustering/01_preprocessing_feature_engineering.ipynb",
        "run_from": ROOT / "analysis/clustering",
        "requires": [ROOT / "dataset_imputed.csv"],
        "produces": [ROOT / "data/processed/df_preprocessed.csv"],
    },
    {
        "id": 3,
        "name": "K-Means independiente por régimen",
        "notebook": ROOT / "analysis/clustering/02_kmeans_independiente.ipynb",
        "run_from": ROOT / "analysis/clustering",
        "requires": [ROOT / "data/processed/df_preprocessed.csv"],
        "produces": [
            ROOT / "data/processed/df_pre.csv",
            ROOT / "data/processed/df_tec.csv",
            ROOT / "data/processed/km_pre.pkl",
            ROOT / "data/processed/km_tec.pkl",
        ],
    },
    {
        "id": 4,
        "name": "Entrenamiento Random Forest (PreTec21 → Tec21)",
        "notebook": ROOT / "analysis/modelos/experimentos/random_forest.ipynb",
        "run_from": ROOT / "analysis/modelos/experimentos",
        "requires": [
            ROOT / "data/processed/df_pre.csv",
            ROOT / "data/processed/df_tec.csv",
        ],
        "produces": [
            ROOT / "data/processed/rf_model.pkl",
            ROOT / "data/processed/X_te_tec.npy",
            ROOT / "data/processed/y_te_tec.npy",
            ROOT / "data/processed/feat_pre.json",
        ],
    },
]


# Helpers 

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"


def log(msg, color=RESET):
    print(f"{color}{msg}{RESET}", flush=True)


def header(msg):
    log(f"\n{'─'*60}", CYAN)
    log(f"  {msg}", BOLD)
    log(f"{'─'*60}", CYAN)


def find_jupyter():
    """Devuelve la ruta al ejecutable de jupyter nbconvert."""
    for candidate in ["jupyter", "jupyter-nbconvert"]:
        result = subprocess.run(
            ["which", candidate], capture_output=True, text=True
        )
        if result.returncode == 0:
            return candidate
    # Try common local install paths
    local_bin = Path.home() / ".local/bin/jupyter"
    if local_bin.exists():
        return str(local_bin)
    return None


def check_missing(paths, label="archivo"):
    missing = [p for p in paths if not p.exists()]
    return missing


def run_notebook(step, timeout=600, force=False):
    """
    Ejecuta un notebook con nbconvert y guarda la versión ejecutada.
    Devuelve True si tuvo éxito, False si falló.
    """
    nb_path   = step["notebook"]
    run_from  = step["run_from"]
    out_path  = nb_path.parent / (nb_path.stem + "_ejecutado.ipynb")

    if not nb_path.exists():
        log(f"  ✗ Notebook no encontrado: {nb_path}", RED)
        return False

    # Copiar el notebook al directorio de ejecución si viene de otro lado
    # (el notebook analisis_estadistico_reproducible.ipynb puede estar en la raíz del workspace)
    nb_src = nb_path
    if not nb_src.exists():
        log(f"  ✗ {nb_src} no existe", RED)
        return False

    jupyter = find_jupyter()
    if not jupyter:
        log("  ✗ jupyter/nbconvert no encontrado. Instálalo con: pip install jupyter nbconvert", RED)
        return False

    cmd = [
        jupyter, "nbconvert",
        "--to", "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output", str(out_path),
        str(nb_src),
    ]

    log(f"  Ejecutando: {nb_path.name}", CYAN)
    log(f"  Directorio: {run_from}")
    t0 = time.time()

    result = subprocess.run(
        cmd, cwd=str(run_from), capture_output=True, text=True
    )

    elapsed = time.time() - t0

    if result.returncode == 0:
        log(f"  ✓ Completado en {elapsed:.1f}s → {out_path.name}", GREEN)
        return True
    else:
        log(f"  ✗ Error después de {elapsed:.1f}s", RED)
        # Mostrar las últimas líneas del error
        stderr_lines = result.stderr.strip().split("\n")
        for line in stderr_lines[-15:]:
            if line.strip():
                log(f"    {line}", RED)
        return False


# Lógica principal 

def run_pipeline(from_step=1, only_step=None, force=False):
    header("Pipeline IFE — Predicción de Deserción Estudiantil")
    log(f"  Raíz del proyecto: {ROOT}")

    # Seleccionar pasos a ejecutar
    if only_step is not None:
        steps_to_run = [s for s in STEPS if s["id"] == only_step]
    else:
        steps_to_run = [s for s in STEPS if s["id"] >= from_step]

    if not steps_to_run:
        log("No hay pasos que ejecutar con los parámetros dados.", YELLOW)
        return

    results = {}

    for step in steps_to_run:
        header(f"Paso {step['id']}: {step['name']}")

        # Verificar prerequisitos
        missing_req = check_missing(step["requires"])
        if missing_req:
            log(f"  ✗ Prerequisitos faltantes:", RED)
            for p in missing_req:
                log(f"    - {p}", RED)
            log("  Saltando este paso.", YELLOW)
            results[step["id"]] = "skipped"
            continue

        # Verificar si ya está completo (skip si no es force)
        if not force:
            produced = check_missing(step["produces"])
            if not produced:
                log(f"  ✓ Salida ya existe — saltando (usa --force para re-ejecutar)", GREEN)
                results[step["id"]] = "skipped"
                continue

        # Ejecutar
        ok = run_notebook(step, force=force)
        results[step["id"]] = "ok" if ok else "error"

        if not ok:
            log(f"\n  Pipeline interrumpido en paso {step['id']}.", RED)
            log("  Corrige el error y vuelve a correr con --from " + str(step["id"]), YELLOW)
            break

    # Resumen final
    header("Resumen")
    for step in STEPS:
        sid = step["id"]
        status = results.get(sid, "no_run")
        if status == "ok":
            icon, color = "✓", GREEN
        elif status == "skipped":
            icon, color = "↷", YELLOW
        elif status == "error":
            icon, color = "✗", RED
        else:
            icon, color = "·", RESET
        log(f"  {icon} Paso {sid}: {step['name']}", color)

    if all(v in ("ok", "skipped") for v in results.values()):
        log("\n  Pipeline completado exitosamente.\n", GREEN)
    else:
        log("\n  Pipeline terminó con errores.\n", RED)
        sys.exit(1)


# CLI 

def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline de análisis IFE de principio a fin.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ejecuta todos los pasos aunque la salida ya exista."
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1, metavar="N",
        help="Empieza desde el paso N (default: 1)."
    )
    parser.add_argument(
        "--only", dest="only_step", type=int, default=None, metavar="N",
        help="Ejecuta solo el paso N."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Muestra los pasos disponibles y sale."
    )
    args = parser.parse_args()

    if args.list:
        print("\nPasos del pipeline:\n")
        for s in STEPS:
            status = "✓" if all(p.exists() for p in s["produces"]) else "○"
            print(f"  {status} [{s['id']}] {s['name']}")
            print(f"       Notebook: {s['notebook'].relative_to(ROOT)}")
        print()
        return

    run_pipeline(
        from_step=args.from_step,
        only_step=args.only_step,
        force=args.force,
    )


if __name__ == "__main__":
    main()
