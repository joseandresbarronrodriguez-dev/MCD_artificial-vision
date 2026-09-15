//
// Created by andru on 9/5/2026.
//
#include "kernels.h"

#include <math.h>
#include <stdlib.h>

/*
 * ============================================================
 * Archivo: kernels.c
 *
 * Descripción:
 *   Implementa la generación de kernels utilizados por los
 *   algoritmos de convolución.
 *
 * Esta primera versión incluye:
 *   - Kernel gaussiano 1D.
 *   - Producto exterior para obtener el kernel 2D.
 *
 * La separación 1D/2D permite demostrar posteriormente la
 * propiedad de separabilidad del filtro gaussiano.
 * ============================================================
 */


/**
 * @brief Genera un kernel gaussiano 1D normalizado.
 */
int gaussian_kernel_1d_create(Kernel1D *kernel,
                              int size,
                              float sigma)
{
    int i;
    int radius;
    float sum = 0.0f;

    /*
     * Validación de parámetros.
     *
     * El tamaño debe ser:
     *   - positivo;
     *   - impar;
     *   - al menos 3.
     */
    if (kernel == NULL ||
        size < 3 ||
        (size % 2) == 0 ||
        sigma <= 0.0f)
    {
        return 0;
    }

    kernel->size = size;

    /*
     * Reserva de memoria para los coeficientes.
     */
    kernel->data =
            (float *) malloc(
                    (size_t) size * sizeof(float));

    if (kernel->data == NULL)
    {
        kernel->size = 0;

        return 0;
    }

    radius = size / 2;

    /*
     * Se evalúa la función gaussiana alrededor de x = 0.
     *
     * Ejemplo para size = 5:
     *
     * x = -2, -1, 0, 1, 2
     */
    for (i = -radius;
         i <= radius;
         ++i)
    {
        float x = (float) i;

        float value =
                expf(
                    -(x * x) /
                    (2.0f * sigma * sigma)
                );

        kernel->data[i + radius] =
                value;

        sum += value;
    }

    /*
     * Normalización:
     *
     * La suma total debe ser aproximadamente 1.
     */
    for (i = 0;
         i < size;
         ++i)
    {
        kernel->data[i] /= sum;
    }

    return 1;
}


/**
 * @brief Construye un kernel 2D mediante producto exterior.
 */
int kernel_outer_product(const Kernel1D *horizontal,
                         const Kernel1D *vertical,
                         Kernel2D *output)
{
    int x;
    int y;
    int size;

    if (horizontal == NULL ||
        vertical == NULL ||
        output == NULL)
    {
        return 0;
    }

    if (horizontal->data == NULL ||
        vertical->data == NULL)
    {
        return 0;
    }

    /*
     * Para esta práctica utilizaremos kernels cuadrados,
     * por lo que ambos vectores deben tener el mismo tamaño.
     */
    if (horizontal->size != vertical->size)
    {
        return 0;
    }

    size = horizontal->size;

    output->size = size;

    /*
     * Un kernel 2D de size x size requiere:
     *
     *      size * size
     *
     * coeficientes float.
     */
    output->data =
            (float *) malloc(
                    (size_t) size *
                    (size_t) size *
                    sizeof(float));

    if (output->data == NULL)
    {
        output->size = 0;

        return 0;
    }

    /*
     * Producto exterior:
     *
     *      K(y,x) = vertical[y] * horizontal[x]
     *
     * El kernel 2D también se almacena linealmente:
     *
     *      data[y * size + x]
     */
    for (y = 0;
         y < size;
         ++y)
    {
        for (x = 0;
             x < size;
             ++x)
        {
            output->data[
                (size_t) y * size + x
            ] =
                vertical->data[y] *
                horizontal->data[x];
        }
    }

    return 1;
}


/**
 * @brief Libera un kernel 1D.
 */
void kernel1d_free(Kernel1D *kernel)
{
    if (kernel == NULL)
    {
        return;
    }

    free(kernel->data);

    kernel->data = NULL;
    kernel->size = 0;
}


/**
 * @brief Libera un kernel 2D.
 */
void kernel2d_free(Kernel2D *kernel)
{
    if (kernel == NULL)
    {
        return;
    }

    free(kernel->data);

    kernel->data = NULL;
    kernel->size = 0;
}

/*
 * ============================================================
 * Kernels Sobel
 * ============================================================
 *
 * Sobel X:
 *
 *  -1   0   1
 *  -2   0   2
 *  -1   0   1
 *
 * Sobel Y:
 *
 *  -1  -2  -1
 *   0   0   0
 *   1   2   1
 *
 * Se almacenan como arreglos estáticos para evitar reservar
 * memoria dinámicamente cada vez que se utilizan.
 */

static float SOBEL_X_DATA[9] =
{
    -1.0f,  0.0f,  1.0f,
    -2.0f,  0.0f,  2.0f,
    -1.0f,  0.0f,  1.0f
};


static float SOBEL_Y_DATA[9] =
{
    -1.0f, -2.0f, -1.0f,
     0.0f,  0.0f,  0.0f,
     1.0f,  2.0f,  1.0f
};


Kernel2D sobel_x_kernel(void)
{
    Kernel2D kernel;

    kernel.size = 3;
    kernel.data = SOBEL_X_DATA;

    return kernel;
}


Kernel2D sobel_y_kernel(void)
{
    Kernel2D kernel;

    kernel.size = 3;
    kernel.data = SOBEL_Y_DATA;

    return kernel;
}

/**
 * @brief Compone dos kernels 2D.
 */
int kernel_compose_2d(
        const Kernel2D *kernelA,
        const Kernel2D *kernelB,
        Kernel2D *output)
{
    int ax;
    int ay;
    int bx;
    int by;

    int outputSize;

    if (kernelA == NULL ||
        kernelB == NULL ||
        output == NULL)
    {
        return 0;
    }

    if (kernelA->data == NULL ||
        kernelB->data == NULL)
    {
        return 0;
    }

    /*
     * Al convolucionar dos kernels discretos, el tamaño final es:
     *
     *      n + m - 1
     */
    outputSize =
            kernelA->size +
            kernelB->size -
            1;

    output->size =
            outputSize;


    output->data =
            (float *) calloc(
                    (size_t) outputSize *
                    (size_t) outputSize,
                    sizeof(float)
            );


    if (output->data == NULL)
    {
        output->size = 0;

        return 0;
    }


    /*
     * Convolución entre kernels.
     *
     * Cada coeficiente de A se multiplica por cada coeficiente
     * de B y se acumula en la posición correspondiente del
     * kernel fusionado.
     */
    for (ay = 0;
         ay < kernelA->size;
         ++ay)
    {
        for (ax = 0;
             ax < kernelA->size;
             ++ax)
        {
            const float valueA =
                    kernelA->data[
                            (size_t) ay *
                            kernelA->size +
                            ax
                    ];


            for (by = 0;
                 by < kernelB->size;
                 ++by)
            {
                for (bx = 0;
                     bx < kernelB->size;
                     ++bx)
                {
                    const float valueB =
                            kernelB->data[
                                    (size_t) by *
                                    kernelB->size +
                                    bx
                            ];


                    const int outY =
                            ay + by;

                    const int outX =
                            ax + bx;


                    output->data[
                            (size_t) outY *
                            outputSize +
                            outX
                    ] +=
                            valueA *
                            valueB;
                }
            }
        }
    }


    return 1;
}