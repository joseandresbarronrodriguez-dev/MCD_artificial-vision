
"""
Práctica 6: Concurso de clasificación CIFAR-100

Archivo: dataset.py

Descripción:
    Módulo encargado de cargar, dividir, normalizar
    y validar el conjunto de datos CIFAR-100.

Objetivo:
    Preparar los datos para entrenar redes CNN,
    garantizando reproducibilidad y evitando
    utilizar el conjunto de prueba durante el
    desarrollo del modelo.

Autor: José Andrés Barrón
Fecha: Octubre 2026
"""

import numpy as np

from keras.datasets import cifar100
from sklearn.model_selection import train_test_split


# ==========================================================
# CONFIGURACIÓN GLOBAL
# ==========================================================

# Semilla fija para reproducir la partición de datos.
RANDOM_SEED = 42

# Porcentaje de los datos originales de entrenamiento
# que se utilizará como conjunto de validación.
VALIDATION_SIZE = 0.10

# Dimensiones esperadas de las imágenes.
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3

# Número total de clases finas de CIFAR-100.
NUM_CLASSES = 100


# ==========================================================
# CARGA DEL DATASET
# ==========================================================

def load_cifar100():
    """
    Carga CIFAR-100 con las 100 clases finas.

    Returns:
        tuple:
            x_train: Imágenes originales de entrenamiento.
            y_train: Etiquetas originales de entrenamiento.
            x_test: Imágenes reservadas para prueba final.
            y_test: Etiquetas reservadas para prueba final.

    Notes:
        El conjunto de prueba no debe utilizarse
        para ajustar los hiperparámetros del modelo.
    """

    print("\nCargando CIFAR-100...")

    # Keras descarga el conjunto de datos automáticamente
    # si todavía no se encuentra en caché.
    (x_train, y_train), (x_test, y_test) = (
        cifar100.load_data(label_mode="fine")
    )

    # Convertimos las etiquetas de matrices (N, 1)
    # a vectores de una dimensión (N,).
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    print("Dataset cargado correctamente.")

    return x_train, y_train, x_test, y_test


# ==========================================================
# NORMALIZACIÓN
# ==========================================================

def normalize_images(images):
    """
    Normaliza las imágenes al intervalo [0, 1].

    Args:
        images (np.ndarray):
            Imágenes con valores de píxel de 0 a 255.

    Returns:
        np.ndarray:
            Imágenes normalizadas en formato float32.

    Notes:
        La normalización facilita el entrenamiento
        de redes neuronales y evita trabajar con
        magnitudes de entrada innecesariamente altas.
    """

    # Convertimos a float32 para reducir el consumo
    # de memoria respecto al uso de float64.
    images = images.astype(np.float32)

    # Escalamos los valores al intervalo [0, 1].
    images /= 255.0

    return images


# ==========================================================
# PARTICIÓN DE ENTRENAMIENTO Y VALIDACIÓN
# ==========================================================

def split_training_validation(x_train, y_train):
    """
    Divide el entrenamiento original en dos conjuntos.

    Se utiliza una partición estratificada para conservar
    la representación proporcional de las 100 clases.

    Args:
        x_train (np.ndarray):
            Imágenes originales de entrenamiento.

        y_train (np.ndarray):
            Etiquetas originales de entrenamiento.

    Returns:
        tuple:
            x_train_final: Imágenes para entrenar.
            x_val: Imágenes de validación.
            y_train_final: Etiquetas para entrenar.
            y_val: Etiquetas de validación.
    """

    (
        x_train_final,
        x_val,
        y_train_final,
        y_val
    ) = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train
    )

    return (
        x_train_final,
        x_val,
        y_train_final,
        y_val
    )


# ==========================================================
# VALIDACIÓN DE DATOS
# ==========================================================

def validate_dataset(images, labels, dataset_name):
    """
    Verifica la estructura de las imágenes y etiquetas.

    Args:
        images (np.ndarray):
            Conjunto de imágenes a verificar.

        labels (np.ndarray):
            Etiquetas correspondientes.

        dataset_name (str):
            Nombre del conjunto para mostrar resultados.

    Raises:
        ValueError:
            Si las dimensiones o etiquetas no son válidas.
    """

    # Verificamos que cada imagen tenga 32x32 píxeles
    # y exactamente tres canales RGB.
    expected_shape = (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS
    )

    if images.ndim != 4 or images.shape[1:] != expected_shape:
        raise ValueError(
            f"Dimensiones incorrectas en {dataset_name}: "
            f"{images.shape}"
        )

    # Cada imagen debe tener una etiqueta.
    if len(images) != len(labels):
        raise ValueError(
            f"Cantidad de imágenes y etiquetas diferente "
            f"en {dataset_name}."
        )

    # Las etiquetas válidas van desde 0 hasta 99.
    if np.any(labels < 0) or np.any(labels >= NUM_CLASSES):
        raise ValueError(
            f"Etiquetas fuera del rango [0, 99] "
            f"en {dataset_name}."
        )

    # Verificamos valores numéricos finitos.
    if not np.isfinite(images).all():
        raise ValueError(
            f"Existen valores no finitos en {dataset_name}."
        )

    print(f"\n{dataset_name}")
    print(f"  Imágenes:   {images.shape}")
    print(f"  Etiquetas:  {labels.shape}")
    print(f"  Tipo:       {images.dtype}")
    print(f"  Valor min:  {images.min():.4f}")
    print(f"  Valor max:  {images.max():.4f}")
    print(f"  Clases:     {len(np.unique(labels))}")


# ==========================================================
# PREPARACIÓN GENERAL
# ==========================================================

def prepare_dataset():
    """
    Ejecuta el proceso de preparación de CIFAR-100.

    Returns:
        dict:
            Contiene los conjuntos de entrenamiento
            y validación normalizados.

    Notes:
        Aunque el cargador proporciona imágenes de
        prueba, esta función no las devuelve ni las
        evalúa. Se reservarán para la fase final.
    """

    x_train, y_train, _, _ = load_cifar100()

    # Separamos los datos antes de normalizarlos.
    (
        x_train,
        x_val,
        y_train,
        y_val
    ) = split_training_validation(x_train, y_train)

    # Aplicamos exactamente la misma normalización
    # a entrenamiento y validación.
    x_train = normalize_images(x_train)
    x_val = normalize_images(x_val)

    # Validamos ambos conjuntos.
    validate_dataset(
        x_train,
        y_train,
        "ENTRENAMIENTO"
    )

    validate_dataset(
        x_val,
        y_val,
        "VALIDACIÓN"
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val
    }


# ==========================================================
# EJECUCIÓN INDEPENDIENTE
# ==========================================================

if __name__ == "__main__":

    print("=" * 50)
    print(" CIFAR-100 DATASET PREPARATION")
    print(" PERFIL 4 - YOLO EN PLATAFORMAS ARM")
    print("=" * 50)

    dataset = prepare_dataset()

    print("\nPreparación completada correctamente.")
    print("El conjunto de prueba queda reservado.")
