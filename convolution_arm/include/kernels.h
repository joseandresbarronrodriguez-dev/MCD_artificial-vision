//
// Created by andru on 9/5/2026.
//
#ifndef KERNELS_H
#define KERNELS_H

/*
 * ============================================================
 * Archivo: kernels.h
 * Práctica: Convolución separable y fusión de kernels
 *
 * Descripción:
 *   Define las estructuras y funciones necesarias para generar
 *   kernels de convolución utilizados en el benchmark.
 *
 * Objetivos:
 *   - Crear kernels gaussianos 1D.
 *   - Construir kernels gaussianos 2D.
 *   - Preparar la base para convolución directa y separable.
 *
 * El código de este módulo está implementado en C puro para
 * conservar portabilidad entre PC x86-64 y ARM64.
 * ============================================================
 */

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Kernel de convolución unidimensional.
 */
typedef struct
{
    int size;       /**< Número de coeficientes. */
    float *data;    /**< Coeficientes del kernel. */

} Kernel1D;


/**
 * @brief Kernel de convolución bidimensional cuadrado.
 */
typedef struct
{
    int size;       /**< Tamaño del kernel: size x size. */
    float *data;    /**< Coeficientes almacenados linealmente. */

} Kernel2D;


/**
 * @brief Genera un kernel gaussiano 1D normalizado.
 *
 * El kernel se calcula utilizando:
 *
 *      G(x) = exp(-x^2 / (2*sigma^2))
 *
 * Después de calcular los coeficientes, se normalizan para que:
 *
 *      sum(G) = 1
 *
 * @param kernel Estructura de salida.
 * @param size   Tamaño impar del kernel.
 * @param sigma  Desviación estándar de la distribución.
 *
 * @return 1 si la creación fue exitosa.
 * @return 0 si los parámetros son inválidos o falla malloc().
 */
int gaussian_kernel_1d_create(Kernel1D *kernel,
                              int size,
                              float sigma);


/**
 * @brief Construye un kernel 2D mediante producto exterior.
 *
 * Dados dos kernels 1D:
 *
 *      horizontal = h
 *      vertical   = v
 *
 * se calcula:
 *
 *      K(y,x) = v(y) * h(x)
 *
 * Para el caso gaussiano:
 *
 *      G2D = G1D^T * G1D
 *
 * @param horizontal Kernel horizontal.
 * @param vertical   Kernel vertical.
 * @param output     Kernel 2D resultante.
 *
 * @return 1 si la operación fue exitosa.
 * @return 0 en caso de error.
 */
int kernel_outer_product(const Kernel1D *horizontal,
                         const Kernel1D *vertical,
                         Kernel2D *output);


/**
 * @brief Libera la memoria de un Kernel1D.
 */
void kernel1d_free(Kernel1D *kernel);


/**
 * @brief Libera la memoria de un Kernel2D.
 */
void kernel2d_free(Kernel2D *kernel);

    /**
     * @brief Devuelve un kernel Sobel horizontal 3x3.
     *
     * Detecta principalmente cambios de intensidad en X.
     *
     * @return Kernel Sobel X.
     */
    Kernel2D sobel_x_kernel(void);


    /**
     * @brief Devuelve un kernel Sobel vertical 3x3.
     *
     * Detecta principalmente cambios de intensidad en Y.
     *
     * @return Kernel Sobel Y.
     */
    Kernel2D sobel_y_kernel(void);

    /**
 * @brief Compone dos kernels 2D mediante convolución discreta.
 *
 * Si primero se aplica kernelA y después kernelB, la operación
 * puede representarse mediante un único kernel fusionado:
 *
 *      K_fused = kernelA * kernelB
 *
 * El tamaño resultante es:
 *
 *      size_out = sizeA + sizeB - 1
 *
 * Ejemplo:
 *
 *      Gaussiano 5x5 + Sobel 3x3
 *
 *      -> kernel fusionado 7x7
 *
 * @param kernelA Primer kernel.
 * @param kernelB Segundo kernel.
 * @param output  Kernel fusionado.
 *
 * @return 1 si la operación fue exitosa.
 * @return 0 en caso de error.
 */
    int kernel_compose_2d(
            const Kernel2D *kernelA,
            const Kernel2D *kernelB,
            Kernel2D *output);

#ifdef __cplusplus
}
#endif

#endif