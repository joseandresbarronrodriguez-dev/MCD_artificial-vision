
"""
Práctica 6: Concurso de clasificación CIFAR-100

Archivo: train.py
Autor: José Andrés Barrón
Fecha: Octubre 2026

Descripción:
    Entrenamiento controlado de dos modelos CNN:

    Modelo A: RGB (3 canales).
    Modelo B: RGB + Sobel (4 canales).

Objetivos:
    1. Comparar rasgos aprendidos y diseñados.
    2. Utilizar las mismas condiciones experimentales.
    3. Guardar modelos e historiales.
    4. Generar gráficas de entrenamiento.
    5. Registrar tiempos y metadatos reproducibles.
    6. Preparar resultados para el análisis ARM.

Restricciones:
    - Máximo 30 épocas por entrenamiento.
    - Máximo 1,000,000 de parámetros entrenables.
    - No utilizar pesos preentrenados.
    - No utilizar el conjunto de prueba.
"""

import os

# IMPORTANTE:
# El backend debe configurarse antes de importar Keras.
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import json
import random
import time

from datetime import datetime
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import prepare_dataset
from features import add_sobel_channel
from model import build_cnn


# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

RANDOM_SEED = 42

MAX_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
LEARNING_RATE = 0.001

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


# ==========================================================
# CONFIGURACIÓN DE REPRODUCIBILIDAD
# ==========================================================

def set_global_seed(seed):
    """
    Establece las semillas para reducir la variabilidad
    entre ejecuciones.

    Args:
        seed (int):
            Semilla de generación pseudoaleatoria.

    Notes:
        No garantiza reproducibilidad bit a bit en
        todos los sistemas o plataformas.

        Ambos modelos utilizan la misma semilla para
        mantener condiciones iniciales comparables.
    """

    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


# ==========================================================
# CONFIGURACIÓN DEL MODELO
# ==========================================================

def compile_model(model):
    """
    Compila una CNN para clasificación multiclase.

    Args:
        model (keras.Model):
            Red neuronal previamente construida.

    Returns:
        keras.Model:
            Modelo compilado para entrenamiento.
    """

    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# ENTRENAMIENTO INDIVIDUAL
# ==========================================================

def train_single_model(
        model_name,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs,
        batch_size,
        output_dir
):
    """
    Entrena un modelo CNN bajo parámetros controlados.

    Args:
        model_name (str):
            Identificador del experimento.

        x_train (np.ndarray):
            Imágenes para entrenamiento.

        y_train (np.ndarray):
            Etiquetas de entrenamiento.

        x_val (np.ndarray):
            Imágenes de validación.

        y_val (np.ndarray):
            Etiquetas de validación.

        epochs (int):
            Cantidad de épocas solicitadas.

        batch_size (int):
            Tamaño de lote utilizado.

        output_dir (Path):
            Directorio donde se guardan los resultados.

    Returns:
        tuple:
            model:
                Modelo entrenado.

            history_dict:
                Historial de accuracy y loss.

            metadata:
                Información del entrenamiento.
    """

    input_channels = x_train.shape[-1]

    print("\n" + "=" * 60)
    print(f" ENTRENANDO MODELO: {model_name}")
    print("=" * 60)

    print(f"Forma entrenamiento: {x_train.shape}")
    print(f"Forma validación:     {x_val.shape}")
    print(f"Canales de entrada:   {input_channels}")
    print(f"Épocas:               {epochs}")
    print(f"Batch size:           {batch_size}")
    print(f"Learning rate:        {LEARNING_RATE}")

    # Reiniciamos la semilla antes de construir cada CNN.
    # Las formas de los pesos de la primera convolución
    # son diferentes entre A y B debido al canal Sobel.
    set_global_seed(RANDOM_SEED)

    model = build_cnn(
        input_channels=input_channels
    )

    model = compile_model(model)

    # ------------------------------------------------------
    # REGISTRO TEMPORAL
    # ------------------------------------------------------

    timestamp_start = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    start_time = time.perf_counter()

    # ------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1
    )

    elapsed_seconds = time.perf_counter() - start_time

    timestamp_end = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    # ------------------------------------------------------
    # GUARDADO DEL MODELO
    # ------------------------------------------------------

    model_path = output_dir / f"{model_name.lower()}.keras"

    # Se guarda localmente para la evaluación final.
    # Los modelos entrenados no se entregan en Git.
    model.save(model_path)

    # ------------------------------------------------------
    # METADATOS DEL EXPERIMENTO
    # ------------------------------------------------------

    trainable_parameters = sum(
        int(np.prod(weight.shape))
        for weight in model.trainable_weights
    )

    metadata = {
        "model_name": model_name,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "seed": RANDOM_SEED,
        "epochs_requested": epochs,
        "epochs_executed": len(history_dict["loss"]),
        "batch_size": batch_size,
        "learning_rate": LEARNING_RATE,
        "input_channels": input_channels,
        "train_samples": len(x_train),
        "validation_samples": len(x_val),
        "trainable_parameters": trainable_parameters,
        "training_time_seconds": round(
            elapsed_seconds, 4
        ),
        "backend": keras.backend.backend(),
        "python_environment": "Python 3.11",
        "keras_version": keras.__version__,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "model_path": str(model_path)
    }

    metadata_path = output_dir / (
        f"{model_name.lower()}_metadata.json"
    )

    with open(
            metadata_path,
            "w",
            encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nEntrenamiento completado.")
    print(f"Tiempo total: {elapsed_seconds:.2f} segundos")
    print(f"Modelo guardado: {model_path}")
    print(f"Metadatos: {metadata_path}")

    return model, history_dict, metadata


# ==========================================================
# GUARDADO DE HISTORIALES
# ==========================================================

def save_history(history_dict, output_path):
    """
    Guarda los resultados de cada época en JSON.

    Args:
        history_dict (dict):
            Métricas del entrenamiento.

        output_path (Path):
            Ruta del archivo de salida.
    """

    with open(
            output_path,
            "w",
            encoding="utf-8"
    ) as file:
        json.dump(
            history_dict,
            file,
            indent=4
        )


# ==========================================================
# VISUALIZACIÓN DEL ENTRENAMIENTO
# ==========================================================

def plot_history(history_dict, model_name, output_dir):
    """
    Genera las gráficas de pérdida y exactitud.

    Args:
        history_dict (dict):
            Historial de métricas.

        model_name (str):
            Identificador del modelo.

        output_dir (Path):
            Directorio de resultados.
    """

    epoch_numbers = np.arange(
        1,
        len(history_dict["loss"]) + 1
    )

    # ------------------------------------------------------
    # CURVA DE PÉRDIDA
    # ------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        epoch_numbers,
        history_dict["loss"],
        marker="o",
        label="Entrenamiento"
    )

    ax.plot(
        epoch_numbers,
        history_dict["val_loss"],
        marker="s",
        label="Validación"
    )

    ax.set_title(f"{model_name} - Función de pérdida")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"{model_name.lower()}_loss.png",
        dpi=200
    )

    plt.close(fig)

    # ------------------------------------------------------
    # CURVA DE ACCURACY
    # ------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        epoch_numbers,
        history_dict["accuracy"],
        marker="o",
        label="Entrenamiento"
    )

    ax.plot(
        epoch_numbers,
        history_dict["val_accuracy"],
        marker="s",
        label="Validación"
    )

    ax.set_title(f"{model_name} - Accuracy")
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"{model_name.lower()}_accuracy.png",
        dpi=200
    )

    plt.close(fig)


# ==========================================================
# RESUMEN COMPARATIVO
# ==========================================================

def summarize_results(history_a, history_b, metadata_a, metadata_b):
    """
    Presenta los resultados finales de A contra B.

    Args:
        history_a (dict):
            Historial del Modelo A.

        history_b (dict):
            Historial del Modelo B.

        metadata_a (dict):
            Metadatos del Modelo A.

        metadata_b (dict):
            Metadatos del Modelo B.
    """

    print("\n" + "=" * 60)
    print(" RESUMEN COMPARATIVO A vs B")
    print("=" * 60)

    for name, history, metadata in (
            ("A - RGB", history_a, metadata_a),
            ("B - RGB + SOBEL", history_b, metadata_b)
    ):
        print(f"\nModelo: {name}")
        print(
            f"Accuracy entrenamiento: "
            f"{history['accuracy'][-1]:.4f}"
        )
        print(
            f"Accuracy validación:    "
            f"{history['val_accuracy'][-1]:.4f}"
        )
        print(
            f"Loss validación:        "
            f"{history['val_loss'][-1]:.4f}"
        )
        print(
            f"Tiempo entrenamiento:   "
            f"{metadata['training_time_seconds']:.2f} s"
        )


# ==========================================================
# ARGUMENTOS DE EJECUCIÓN
# ==========================================================

def parse_arguments():
    """
    Procesa los argumentos enviados desde PowerShell.

    Returns:
        argparse.Namespace:
            Configuración seleccionada.
    """

    parser = argparse.ArgumentParser(
        description="Entrenamiento CIFAR-100 A vs B"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Número de épocas (1 a 30)."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Tamaño del lote."
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nombre opcional del experimento."
    )

    args = parser.parse_args()

    if not 1 <= args.epochs <= MAX_EPOCHS:
        parser.error(
            "El número de épocas debe estar entre 1 y 30."
        )

    if args.batch_size < 1:
        parser.error(
            "El batch size debe ser positivo."
        )

    return args


# ==========================================================
# EJECUCIÓN PRINCIPAL
# ==========================================================

def main():
    """
    Ejecuta el flujo completo de entrenamiento A vs B.

    Etapas:
        1. Leer configuración.
        2. Preparar dataset.
        3. Generar canal Sobel.
        4. Entrenar ambos modelos.
        5. Guardar resultados.
        6. Mostrar resumen comparativo.
    """

    args = parse_arguments()

    # Cada ejecución guarda sus resultados en una
    # carpeta independiente para evitar sobrescribir
    # experimentos anteriores.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = args.run_name or (
        f"cifar100_{args.epochs}epochs_{timestamp}"
    )

    output_dir = RESULTS_DIR / run_name

    if output_dir.exists():
        raise FileExistsError(
            f"Ya existe el experimento: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    print("=" * 60)
    print(" CIFAR-100 TRAINING")
    print(" PERFIL 4 - YOLO EN PLATAFORMAS ARM")
    print("=" * 60)

    print(f"Experimento: {run_name}")
    print(f"Épocas:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Semilla:     {RANDOM_SEED}")
    print(f"Resultados:  {output_dir}")

    # ------------------------------------------------------
    # 1. CARGA DEL DATASET
    # ------------------------------------------------------

    dataset = prepare_dataset()

    x_train_rgb = dataset["x_train"]
    y_train = dataset["y_train"]

    x_val_rgb = dataset["x_val"]
    y_val = dataset["y_val"]

    # ------------------------------------------------------
    # 2. PREPARACIÓN RGB + SOBEL
    # ------------------------------------------------------

    print("\nGenerando imágenes RGB + Sobel...")

    preprocessing_start = time.perf_counter()

    x_train_sobel = add_sobel_channel(x_train_rgb)
    x_val_sobel = add_sobel_channel(x_val_rgb)

    preprocessing_seconds = (
            time.perf_counter() - preprocessing_start
    )

    print(f"RGB:        {x_train_rgb.shape}")
    print(f"RGB+Sobel:  {x_train_sobel.shape}")
    print(
        f"Tiempo de preprocesamiento Sobel: "
        f"{preprocessing_seconds:.2f} s"
    )

    # ------------------------------------------------------
    # 3. ENTRENAMIENTO MODELO A
    # ------------------------------------------------------

    model_a, history_a, metadata_a = train_single_model(
        model_name="A_RGB",
        x_train=x_train_rgb,
        y_train=y_train,
        x_val=x_val_rgb,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=output_dir
    )

    # ------------------------------------------------------
    # 4. ENTRENAMIENTO MODELO B
    # ------------------------------------------------------

    model_b, history_b, metadata_b = train_single_model(
        model_name="B_RGB_SOBEL",
        x_train=x_train_sobel,
        y_train=y_train,
        x_val=x_val_sobel,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=output_dir
    )

    # ------------------------------------------------------
    # 5. GUARDADO DE EVIDENCIAS
    # ------------------------------------------------------

    save_history(
        history_a,
        output_dir / "history_a_rgb.json"
    )

    save_history(
        history_b,
        output_dir / "history_b_rgb_sobel.json"
    )

    plot_history(
        history_a,
        "A_RGB",
        output_dir
    )

    plot_history(
        history_b,
        "B_RGB_SOBEL",
        output_dir
    )

    # Guardamos información general del experimento.
    experiment_info = {
        "run_name": run_name,
        "seed": RANDOM_SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sobel_preprocessing_seconds": round(
            preprocessing_seconds, 4
        ),
        "test_set_used": False,
        "models": {
            "A_RGB": metadata_a,
            "B_RGB_SOBEL": metadata_b
        }
    }

    with open(
            output_dir / "experiment.json",
            "w",
            encoding="utf-8"
    ) as file:
        json.dump(
            experiment_info,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ------------------------------------------------------
    # 6. RESULTADOS FINALES
    # ------------------------------------------------------

    summarize_results(
        history_a,
        history_b,
        metadata_a,
        metadata_b
    )

    print("\nEntrenamiento finalizado correctamente.")
    print(f"Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()
