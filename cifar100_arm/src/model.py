
"""
Práctica 6: Concurso de clasificación CIFAR-100

Archivo: model.py

Descripción:
    Define una CNN configurable para clasificar las
    100 clases de CIFAR-100.

Objetivos:
    1. Construir una CNN desde cero.
    2. Mantener menos de 1,000,000 de parámetros
       entrenables.
    3. Permitir entradas RGB o RGB + rasgo diseñado.
    4. Facilitar el análisis de costo computacional
       de las capas convolucionales.

Autor: José Andrés Barrón
Fecha: Octubre 2026
"""

import os

# El backend debe configurarse antes de importar Keras.
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers


# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

IMAGE_SIZE = 32
NUM_CLASSES = 100

# Límite establecido por las reglas del concurso.
MAX_TRAINABLE_PARAMS = 1_000_000


# ==========================================================
# BLOQUE CONVOLUCIONAL
# ==========================================================

def convolutional_block(x, filters, block_name):
    """
    Construye un bloque de extracción de características.

    Estructura:
        Conv2D -> BatchNormalization -> ReLU
        Conv2D -> BatchNormalization -> ReLU
        MaxPooling2D

    Args:
        x:
            Tensor de entrada del bloque.

        filters (int):
            Número de filtros de convolución.

        block_name (str):
            Prefijo para identificar las capas.

    Returns:
        Tensor:
            Salida del bloque convolucional.

    Notes:
        Los kernels 3x3 utilizan padding='same' para
        conservar la resolución espacial antes del
        pooling.

        MaxPooling2D reduce a la mitad las dimensiones
        espaciales, disminuyendo el costo de las capas
        posteriores.
    """

    for conv_index in range(1, 3):

        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"{block_name}_conv{conv_index}"
        )(x)

        x = layers.BatchNormalization(
            name=f"{block_name}_bn{conv_index}"
        )(x)

        x = layers.Activation(
            "relu",
            name=f"{block_name}_relu{conv_index}"
        )(x)

    x = layers.MaxPooling2D(
        pool_size=2,
        strides=2,
        name=f"{block_name}_pool"
    )(x)

    return x


# ==========================================================
# CONSTRUCCIÓN DE LA CNN
# ==========================================================

def build_cnn(input_channels=3):
    """
    Construye la red convolucional de CIFAR-100.

    Args:
        input_channels (int):
            Cantidad de canales de la imagen.

            3: RGB (Modelo A).
            4: RGB + Sobel (Modelo B).

    Returns:
        keras.Model:
            CNN creada desde cero, sin pesos
            preentrenados.

    Raises:
        ValueError:
            Si el número de canales no es válido
            o se excede el límite de parámetros.
    """

    if input_channels not in (3, 4):
        raise ValueError(
            "Esta práctica admite 3 o 4 canales."
        )

    inputs = keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, input_channels),
        name="input_image"
    )

    # Bloque 1: aprende características locales
    # como bordes, cambios de color y texturas.
    x = convolutional_block(
        inputs,
        filters=32,
        block_name="block1"
    )

    # Bloque 2: combina características locales
    # para formar representaciones más complejas.
    x = convolutional_block(
        x,
        filters=64,
        block_name="block2"
    )

    # Bloque 3: aprende representaciones de mayor
    # nivel para diferenciar objetos y clases.
    x = convolutional_block(
        x,
        filters=128,
        block_name="block3"
    )

    # Reduce cada mapa de características a un
    # valor promedio, evitando una Dense grande.
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    # Regularización para reducir sobreajuste.
    x = layers.Dropout(
        rate=0.30,
        name="dropout"
    )(x)

    # Salida multiclase: 100 probabilidades.
    outputs = layers.Dense(
        units=NUM_CLASSES,
        activation="softmax",
        name="classifier"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"CIFAR100_CNN_{input_channels}ch"
    )

    # Keras permite contar todos los parámetros,
    # incluyendo entrenables y no entrenables.
    trainable_params = sum(
        keras.ops.convert_to_numpy(
            keras.ops.prod(keras.ops.convert_to_tensor(w.shape))
        ).item()
        for w in model.trainable_weights
    )

    if trainable_params > MAX_TRAINABLE_PARAMS:
        raise ValueError(
            f"Se excede el límite de parámetros: "
            f"{trainable_params}"
        )

    print("\nModelo construido correctamente.")
    print(f"Canales de entrada: {input_channels}")
    print(f"Parámetros entrenables: {trainable_params:,}")

    return model


# ==========================================================
# PRUEBA INDEPENDIENTE DEL MÓDULO
# ==========================================================

if __name__ == "__main__":

    print("=" * 55)
    print(" CIFAR-100 CNN ARCHITECTURE")
    print(" PERFIL 4 - YOLO EN PLATAFORMAS ARM")
    print("=" * 55)

    print("\nMODELO A - RGB")
    model_a = build_cnn(input_channels=3)
    model_a.summary()

    print("\nMODELO B - RGB + SOBEL")
    model_b = build_cnn(input_channels=4)
    model_b.summary()
