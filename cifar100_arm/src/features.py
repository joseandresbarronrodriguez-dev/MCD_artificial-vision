
"""
Práctica 6: Concurso de clasificación CIFAR-100

Archivo: features.py
Autor: José Andrés Barrón
Fecha: Octubre 2026

Descripción:
    Implementa la extracción de características mediante
    el operador Sobel para crear un canal adicional a
    imágenes RGB del conjunto CIFAR-100.

Objetivo:
    Comparar dos configuraciones de entrada:

    Modelo A: RGB (3 canales).
    Modelo B: RGB + Sobel (4 canales).

    Este módulo utiliza OpenCV y NumPy, sin implementar
    manualmente la operación de convolución.

Consideraciones ARM:
    La extracción del canal Sobel representa un costo
    adicional de preprocesamiento que posteriormente
    podremos medir y comparar con la inferencia CNN.
"""

import cv2
import numpy as np


# ==========================================================
# CONFIGURACIÓN GLOBAL
# ==========================================================

# Dimensiones originales de las imágenes CIFAR-100.
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
RGB_CHANNELS = 3

# Kernel Sobel 3x3.
SOBEL_KERNEL_SIZE = 3

# Para entradas en [0, 1], el kernel Sobel 3x3
# presenta magnitudes máximas acotadas por sqrt(32).
# Usamos una escala fija para mantener consistencia.
SOBEL_NORMALIZATION = np.sqrt(32.0)


# ==========================================================
# EXTRACCIÓN DEL CANAL SOBEL
# ==========================================================

def extract_sobel_channel(image):
    """
    Calcula la magnitud del gradiente Sobel.

    Args:
        image (np.ndarray):
            Imagen RGB normalizada con dimensiones
            (32, 32, 3) y valores en [0, 1].

    Returns:
        np.ndarray:
            Canal Sobel con dimensiones (32, 32, 1),
            valores normalizados en [0, 1] y tipo float32.

    Raises:
        ValueError:
            Si las dimensiones, valores o formato de
            la imagen no son adecuados.

    Notes:
        Utilizamos la magnitud del gradiente:

            G = sqrt(Gx^2 + Gy^2)

        Gx representa cambios horizontales de intensidad.
        Gy representa cambios verticales de intensidad.

        La magnitud conserva la intensidad del borde,
        pero no su orientación.
    """

    # ------------------------------------------------------
    # VALIDACIÓN DE ENTRADA
    # ------------------------------------------------------

    expected_shape = (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        RGB_CHANNELS
    )

    if image.shape != expected_shape:
        raise ValueError(
            f"Forma incorrecta: {image.shape}. "
            f"Se esperaba {expected_shape}."
        )

    if not np.isfinite(image).all():
        raise ValueError(
            "La imagen contiene valores no finitos."
        )

    if np.min(image) < 0 or np.max(image) > 1:
        raise ValueError(
            "Los píxeles deben estar en [0, 1]."
        )

    # Convertimos a float32 para asegurar un tipo
    # de dato compatible con OpenCV.
    rgb_image = image.astype(np.float32)

    # ------------------------------------------------------
    # CONVERSIÓN RGB A ESCALA DE GRISES
    # ------------------------------------------------------

    # CIFAR-100 entrega imágenes RGB.
    # cv2.COLOR_RGB2GRAY respeta ese orden.
    gray_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY
    )

    # ------------------------------------------------------
    # CÁLCULO DE GRADIENTES
    # ------------------------------------------------------

    # Gradiente horizontal.
    gradient_x = cv2.Sobel(
        gray_image,
        cv2.CV_32F,
        dx=1,
        dy=0,
        ksize=SOBEL_KERNEL_SIZE
    )

    # Gradiente vertical.
    gradient_y = cv2.Sobel(
        gray_image,
        cv2.CV_32F,
        dx=0,
        dy=1,
        ksize=SOBEL_KERNEL_SIZE
    )

    # ------------------------------------------------------
    # MAGNITUD Y NORMALIZACIÓN
    # ------------------------------------------------------

    magnitude = cv2.magnitude(
        gradient_x,
        gradient_y
    )

    # Normalización constante y no por imagen.
    # Esto permite conservar la escala relativa
    # de los gradientes entre diferentes imágenes.
    magnitude = magnitude / SOBEL_NORMALIZATION

    magnitude = np.clip(
        magnitude,
        0.0,
        1.0
    ).astype(np.float32)

    # Agregamos la dimensión del canal.
    return magnitude[..., np.newaxis]


# ==========================================================
# CONSTRUCCIÓN DE ENTRADAS RGB + SOBEL
# ==========================================================

def add_sobel_channel(images):
    """
    Genera imágenes de cuatro canales RGB + Sobel.

    Args:
        images (np.ndarray):
            Conjunto RGB con dimensiones
            (N, 32, 32, 3), normalizado en [0, 1].

    Returns:
        np.ndarray:
            Conjunto RGB + Sobel de dimensiones
            (N, 32, 32, 4), tipo float32.

    Notes:
        Orden de canales:
            0: Rojo
            1: Verde
            2: Azul
            3: Magnitud Sobel

        La función no modifica el arreglo original.
    """

    if images.ndim != 4 or images.shape[1:] != (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            RGB_CHANNELS
    ):
        raise ValueError(
            f"Dimensiones incorrectas: {images.shape}"
        )

    # Reservamos memoria para los cuatro canales.
    result = np.empty(
        (
            len(images),
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            RGB_CHANNELS + 1
        ),
        dtype=np.float32
    )

    # Copiamos los canales RGB originales.
    result[..., :RGB_CHANNELS] = images

    # Calculamos Sobel para cada imagen.
    for index, image in enumerate(images):
        sobel_channel = extract_sobel_channel(image)

        result[index, ..., RGB_CHANNELS] = (
            sobel_channel[..., 0]
        )

    return result


# ==========================================================
# PRUEBA INDEPENDIENTE
# ==========================================================

if __name__ == "__main__":

    print("=" * 55)
    print(" CIFAR-100 SOBEL FEATURE TEST")
    print(" PERFIL 4 - YOLO EN PLATAFORMAS ARM")
    print("=" * 55)

    # Imagen artificial con una transición vertical.
    # La mitad izquierda es negra y la derecha blanca.
    test_image = np.zeros(
        (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            RGB_CHANNELS
        ),
        dtype=np.float32
    )

    test_image[:, 16:, :] = 1.0

    # Extraemos el canal de bordes.
    sobel = extract_sobel_channel(test_image)

    # Construimos una muestra RGB + Sobel.
    combined = add_sobel_channel(
        test_image[np.newaxis, ...]
    )

    # Validamos dimensiones y contenido.
    assert sobel.shape == (32, 32, 1)
    assert combined.shape == (1, 32, 32, 4)

    assert np.min(sobel) >= 0.0
    assert np.max(sobel) <= 1.0
    assert np.max(sobel) > 0.0

    assert np.array_equal(
        combined[0, ..., :3],
        test_image
    )

    assert np.array_equal(
        combined[0, ..., 3:],
        sobel
    )

    print("\nImagen RGB:", test_image.shape)
    print("Canal Sobel:", sobel.shape)
    print("Imagen RGB + Sobel:", combined.shape)

    print(
        "Magnitud mínima:",
        float(np.min(sobel))
    )

    print(
        "Magnitud máxima:",
        float(np.max(sobel))
    )

    print("\nPruebas completadas correctamente.")
