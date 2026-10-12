
"""
Práctica 6: Concurso de clasificación CIFAR-100

Archivo: visualize_features.py
Autor: José Andrés Barrón
Fecha: Octubre 2026

Descripción:
    Visualiza imágenes RGB de CIFAR-100 y los canales
    Sobel que utilizaremos en el Modelo B.

Objetivos:
    1. Verificar visualmente la extracción de rasgos.
    2. Comparar imágenes RGB y magnitudes Sobel.
    3. Generar evidencias para el reporte.
    4. Conservar resultados reproducibles.

Consideraciones:
    Este módulo utiliza exclusivamente datos de
    entrenamiento, sin acceder al conjunto de prueba.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import prepare_dataset
from features import extract_sobel_channel


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

# Ruta del módulo y directorio de resultados.
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

RANDOM_SEED = 42

# Cantidad de imágenes que se mostrarán.
NUM_IMAGES = 5


# ==========================================================
# VISUALIZACIÓN DE CARACTERÍSTICAS
# ==========================================================

def visualize_rgb_sobel(images, labels, num_images=5):
    """
    Construye una comparación visual RGB contra Sobel.

    Args:
        images (np.ndarray):
            Imágenes RGB normalizadas de dimensiones
            (N, 32, 32, 3).

        labels (np.ndarray):
            Etiquetas correspondientes a las imágenes.

        num_images (int):
            Número de ejemplos que se visualizarán.

    Returns:
        Path:
            Ruta de la imagen PNG generada.

    Raises:
        ValueError:
            Si no hay suficientes imágenes disponibles.

    Notes:
        Se utiliza una semilla fija para seleccionar
        siempre las mismas imágenes entre ejecuciones.
    """

    if len(images) != len(labels):
        raise ValueError(
            "Las imágenes y etiquetas no coinciden."
        )

    if num_images < 1 or num_images > len(images):
        raise ValueError(
            "Cantidad de imágenes solicitada no válida."
        )

    # Generador independiente para que la selección
    # sea reproducible sin alterar otras semillas.
    rng = np.random.default_rng(RANDOM_SEED)

    selected_indices = rng.choice(
        len(images),
        size=num_images,
        replace=False
    )

    # Cada columna representa una imagen.
    # Fila 1: Imagen original.
    # Fila 2: Canal Sobel.
    fig, axes = plt.subplots(
        2,
        num_images,
        figsize=(3 * num_images, 6),
        squeeze=False
    )

    for column, image_index in enumerate(selected_indices):

        rgb_image = images[image_index]
        label = int(labels[image_index])

        # Reutilizamos nuestra implementación Sobel.
        sobel_image = extract_sobel_channel(
            rgb_image
        )[..., 0]

        # --------------------------------------------------
        # IMAGEN RGB
        # --------------------------------------------------

        axes[0, column].imshow(
            rgb_image,
            interpolation="nearest"
        )

        axes[0, column].set_title(
            f"RGB - Clase {label}"
        )

        axes[0, column].axis("off")

        # --------------------------------------------------
        # MAGNITUD SOBEL
        # --------------------------------------------------

        axes[1, column].imshow(
            sobel_image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest"
        )

        axes[1, column].set_title(
            f"Sobel - Clase {label}"
        )

        axes[1, column].axis("off")

    fig.suptitle(
        "CIFAR-100: Comparación RGB vs Sobel",
        fontsize=15
    )

    fig.tight_layout()

    # Creamos el directorio si todavía no existe.
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = RESULTS_DIR / "rgb_vs_sobel.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ==========================================================
# EJECUCIÓN PRINCIPAL
# ==========================================================

if __name__ == "__main__":

    print("=" * 55)
    print(" CIFAR-100 FEATURE VISUALIZATION")
    print(" PERFIL 4 - YOLO EN PLATAFORMAS ARM")
    print("=" * 55)

    # Cargamos únicamente entrenamiento y validación.
    dataset = prepare_dataset()

    # Para evitar contaminar la evaluación final,
    # utilizamos solo imágenes de entrenamiento.
    output_file = visualize_rgb_sobel(
        images=dataset["x_train"],
        labels=dataset["y_train"],
        num_images=NUM_IMAGES
    )

    print("\nVisualización generada correctamente.")
    print(f"Archivo: {output_file}")
