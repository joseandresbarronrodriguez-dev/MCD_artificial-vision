//
// Created by andru on 9/5/2026.
//
#include "convolution.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * ============================================================
 * Archivo: convolution.c
 *
 * Descripción:
 *   Implementa convolución espacial directa desde cero.
 *
 * Características de esta primera versión:
 *   - Entrada de 8 bits.
 *   - Kernel 2D arbitrario.
 *   - Salida float.
 *   - Política de bordes: zero border.
 *   - Implementación escalar, sin SIMD ni OpenCV.
 *
 * Esta será la versión base contra la que posteriormente se
 * compararán:
 *
 *   1. Convolución separable.
 *   2. Fusión de kernels.
 *   3. Optimizaciones ARM/NEON.
 * ============================================================
 */


/**
 * @brief Reserva memoria para una imagen float.
 */
int image_f32_create(ImageF32 *image,
                     int width,
                     int height)
{
    size_t total_elements;

    if (image == NULL)
    {
        return 0;
    }

    if (width <= 0 || height <= 0)
    {
        return 0;
    }

    total_elements =
            (size_t) width *
            (size_t) height;

    image->data =
            (float *) malloc(
                    total_elements *
                    sizeof(float));

    if (image->data == NULL)
    {
        image->width = 0;
        image->height = 0;

        return 0;
    }

    image->width = width;
    image->height = height;

    /*
     * Inicializamos la salida en cero.
     *
     * Esto permite que los bordes permanezcan en cero sin
     * necesidad de escribirlos explícitamente después.
     */
    memset(image->data,
           0,
           total_elements * sizeof(float));

    return 1;
}


/**
 * @brief Libera una imagen de punto flotante.
 */
void image_f32_free(ImageF32 *image)
{
    if (image == NULL)
    {
        return;
    }

    free(image->data);

    image->data = NULL;
    image->width = 0;
    image->height = 0;
}


/**
 * @brief Aplica convolución 2D directa sobre una imagen.
 */
int convolution_2d_direct_u8_f32(const ImageU8 *input,
                                 const Kernel2D *kernel,
                                 ImageF32 *output)
{
    int x;
    int y;
    int kx;
    int ky;
    int radius;

    /*
     * Validación de punteros.
     */
    if (input == NULL ||
        kernel == NULL ||
        output == NULL)
    {
        return 0;
    }

    if (input->data == NULL ||
        kernel->data == NULL)
    {
        return 0;
    }

    /*
     * El kernel debe ser impar para tener un centro definido.
     */
    if (kernel->size < 1 ||
        (kernel->size % 2) == 0)
    {
        return 0;
    }

    /*
     * Creamos la imagen de salida con las mismas dimensiones
     * que la imagen de entrada.
     */
    if (!image_f32_create(output,
                          input->width,
                          input->height))
    {
        return 0;
    }

    radius = kernel->size / 2;

    /*
     * Recorremos únicamente los píxeles donde el kernel cabe
     * completamente dentro de la imagen.
     *
     * Esto evita accesos fuera de rango.
     */
    for (y = radius;
         y < input->height - radius;
         ++y)
    {
        for (x = radius;
             x < input->width - radius;
             ++x)
        {
            float accumulator = 0.0f;

            /*
             * Recorrido completo del kernel 2D.
             *
             * Para un kernel k x k se realizan aproximadamente
             * k^2 multiplicaciones por píxel.
             */
            for (ky = -radius;
                 ky <= radius;
                 ++ky)
            {
                for (kx = -radius;
                     kx <= radius;
                     ++kx)
                {
                    /*
                     * Coordenadas del píxel de entrada asociado
                     * al coeficiente actual del kernel.
                     */
                    const int input_x = x + kx;
                    const int input_y = y + ky;

                    /*
                     * Índice lineal de la imagen:
                     *
                     *     index = y * width + x
                     */
                    const size_t input_index =
                            (size_t) input_y *
                            input->width +
                            input_x;

                    /*
                     * Índice lineal dentro del kernel.
                     */
                    const size_t kernel_index =
                            (size_t) (ky + radius) *
                            kernel->size +
                            (kx + radius);

                    /*
                     * Multiplicación-acumulación principal.
                     */
                    accumulator +=
                            (float) input->data[input_index] *
                            kernel->data[kernel_index];
                }
            }

            /*
             * Almacenamos el resultado de la convolución.
             */
            output->data[
                (size_t) y * output->width + x
            ] = accumulator;
        }
    }

    return 1;
}

/**
 * @brief Convolución horizontal 1D.
 */
int convolution_1d_horizontal_u8_f32(
        const ImageU8 *input,
        const Kernel1D *kernel,
        ImageF32 *output)
{
    int x;
    int y;
    int kx;
    int radius;

    /*
     * Validación de argumentos.
     */
    if (input == NULL ||
        kernel == NULL ||
        output == NULL)
    {
        return 0;
    }

    if (input->data == NULL ||
        kernel->data == NULL)
    {
        return 0;
    }

    if (kernel->size < 1 ||
        (kernel->size % 2) == 0)
    {
        return 0;
    }

    /*
     * Reservamos la imagen de salida.
     */
    if (!image_f32_create(
            output,
            input->width,
            input->height))
    {
        return 0;
    }

    radius = kernel->size / 2;

    /*
     * Recorremos todas las filas.
     *
     * La restricción de borde se aplica únicamente en X,
     * porque esta etapa solo trabaja horizontalmente.
     */
    for (y = 0;
         y < input->height;
         ++y)
    {
        for (x = radius;
             x < input->width - radius;
             ++x)
        {
            float accumulator = 0.0f;

            /*
             * El acceso horizontal tiene buena localidad de
             * memoria, ya que los píxeles vecinos son
             * contiguos.
             */
            for (kx = -radius;
                 kx <= radius;
                 ++kx)
            {
                const size_t input_index =
                        (size_t) y *
                        input->width +
                        (x + kx);

                const int kernel_index =
                        kx + radius;

                accumulator +=
                        (float) input->data[input_index] *
                        kernel->data[kernel_index];
            }

            output->data[
                    (size_t) y * output->width + x
            ] = accumulator;
        }
    }

    return 1;
}


/**
 * @brief Convolución vertical 1D sobre una imagen float.
 */
int convolution_1d_vertical_f32(
        const ImageF32 *input,
        const Kernel1D *kernel,
        ImageF32 *output)
{
    int x;
    int y;
    int ky;
    int radius;

    if (input == NULL ||
        kernel == NULL ||
        output == NULL)
    {
        return 0;
    }

    if (input->data == NULL ||
        kernel->data == NULL)
    {
        return 0;
    }

    if (kernel->size < 1 ||
        (kernel->size % 2) == 0)
    {
        return 0;
    }

    if (!image_f32_create(
            output,
            input->width,
            input->height))
    {
        return 0;
    }

    radius = kernel->size / 2;

    /*
     * En esta segunda pasada limitamos tanto X como Y.
     *
     * X se limita porque la imagen horizontal contiene ceros
     * en sus bordes laterales.
     */
    for (y = radius;
         y < input->height - radius;
         ++y)
    {
        for (x = radius;
             x < input->width - radius;
             ++x)
        {
            float accumulator = 0.0f;

            /*
             * El acceso vertical salta width posiciones entre
             * muestras consecutivas.
             *
             * Esta diferencia en localidad de memoria será
             * interesante al medir posteriormente sobre ARM.
             */
            for (ky = -radius;
                 ky <= radius;
                 ++ky)
            {
                const size_t input_index =
                        (size_t) (y + ky) *
                        input->width +
                        x;

                const int kernel_index =
                        ky + radius;

                accumulator +=
                        input->data[input_index] *
                        kernel->data[kernel_index];
            }

            output->data[
                    (size_t) y * output->width + x
            ] = accumulator;
        }
    }

    return 1;
}


/**
 * @brief Filtro gaussiano completo utilizando separabilidad.
 */
int convolution_gaussian_separable(
        const ImageU8 *input,
        const Kernel1D *kernel,
        ImageF32 *intermediate,
        ImageF32 *output)
{
    /*
     * Primera pasada:
     *
     *     imagen -> horizontal
     */
    if (!convolution_1d_horizontal_u8_f32(
            input,
            kernel,
            intermediate))
    {
        return 0;
    }

    /*
     * Segunda pasada:
     *
     *     horizontal -> vertical
     */
    if (!convolution_1d_vertical_f32(
            intermediate,
            kernel,
            output))
    {
        /*
         * Si falla la segunda etapa, liberamos el buffer que
         * fue reservado durante la primera.
         */
        image_f32_free(intermediate);

        return 0;
    }

    return 1;
}

/**
 * @brief Convolución 2D entre una imagen float y un kernel 2D.
 */
int convolution_2d_f32_f32(
        const ImageF32 *input,
        const Kernel2D *kernel,
        ImageF32 *output)
{
    int x;
    int y;
    int kx;
    int ky;
    int radius;

    if (input == NULL ||
        kernel == NULL ||
        output == NULL)
    {
        return 0;
    }

    if (input->data == NULL ||
        kernel->data == NULL)
    {
        return 0;
    }

    if (kernel->size < 1 ||
        (kernel->size % 2) == 0)
    {
        return 0;
    }

    if (!image_f32_create(
            output,
            input->width,
            input->height))
    {
        return 0;
    }

    radius = kernel->size / 2;

    /*
     * Recorremos solamente la región válida.
     */
    for (y = radius;
         y < input->height - radius;
         ++y)
    {
        for (x = radius;
             x < input->width - radius;
             ++x)
        {
            float accumulator = 0.0f;

            for (ky = -radius;
                 ky <= radius;
                 ++ky)
            {
                for (kx = -radius;
                     kx <= radius;
                     ++kx)
                {
                    const size_t inputIndex =
                            (size_t)(y + ky)
                            * input->width
                            + (x + kx);

                    const size_t kernelIndex =
                            (size_t)(ky + radius)
                            * kernel->size
                            + (kx + radius);

                    accumulator +=
                            input->data[inputIndex]
                            *
                            kernel->data[kernelIndex];
                }
            }

            output->data[
                    (size_t)y * output->width + x
            ] = accumulator;
        }
    }

    return 1;
}


/**
 * @brief Calcula la magnitud del gradiente Sobel.
 */
int gradient_magnitude_f32(
        const ImageF32 *gx,
        const ImageF32 *gy,
        ImageF32 *magnitude)
{
    int x;
    int y;

    if (gx == NULL ||
        gy == NULL ||
        magnitude == NULL)
    {
        return 0;
    }

    if (gx->data == NULL ||
        gy->data == NULL)
    {
        return 0;
    }

    if (gx->width != gy->width ||
        gx->height != gy->height)
    {
        return 0;
    }

    if (!image_f32_create(
            magnitude,
            gx->width,
            gx->height))
    {
        return 0;
    }

    for (y = 0;
         y < gx->height;
         ++y)
    {
        for (x = 0;
             x < gx->width;
             ++x)
        {
            const size_t index =
                    (size_t)y * gx->width + x;

            const float valueX =
                    gx->data[index];

            const float valueY =
                    gy->data[index];

            magnitude->data[index] =
                    sqrtf(
                        valueX * valueX
                        +
                        valueY * valueY
                    );
        }
    }

    return 1;
}

/*
 * ============================================================
 * VERSIONES "INTO"
 * ============================================================
 *
 * Estas funciones NO reservan memoria.
 *
 * Se utilizan específicamente para benchmarks donde queremos
 * excluir malloc/free del tiempo medido.
 */


static int validate_f32_output(
        const ImageF32 *output,
        int width,
        int height)
{
    if (output == NULL ||
        output->data == NULL)
    {
        return 0;
    }

    if (output->width != width ||
        output->height != height)
    {
        return 0;
    }

    return 1;
}


int convolution_2d_direct_u8_f32_into(
        const ImageU8 *input,
        const Kernel2D *kernel,
        ImageF32 *output)
{
    int x;
    int y;
    int kx;
    int ky;

    int radius;


    if (input == NULL ||
        kernel == NULL)
    {
        return 0;
    }


    if (input->data == NULL ||
        kernel->data == NULL)
    {
        return 0;
    }


    if (!validate_f32_output(
            output,
            input->width,
            input->height))
    {
        return 0;
    }


    radius =
            kernel->size / 2;


    /*
     * Dejamos bordes en cero.
     */
    memset(
            output->data,
            0,
            (size_t) output->width *
            (size_t) output->height *
            sizeof(float)
    );


    for (y = radius;
         y < input->height - radius;
         ++y)
    {
        for (x = radius;
             x < input->width - radius;
             ++x)
        {
            float accumulator = 0.0f;


            for (ky = -radius;
                 ky <= radius;
                 ++ky)
            {
                for (kx = -radius;
                     kx <= radius;
                     ++kx)
                {
                    const size_t inputIndex =
                            (size_t) (y + ky)
                            *
                            input->width
                            +
                            (x + kx);


                    const size_t kernelIndex =
                            (size_t) (ky + radius)
                            *
                            kernel->size
                            +
                            (kx + radius);


                    accumulator +=
                            (float) input->data[inputIndex]
                            *
                            kernel->data[kernelIndex];
                }
            }


            output->data[
                    (size_t) y *
                    output->width
                    +
                    x
            ] =
                    accumulator;
        }
    }


    return 1;
}


int convolution_2d_f32_f32_into(
        const ImageF32 *input,
        const Kernel2D *kernel,
        ImageF32 *output)
{
    int x;
    int y;
    int kx;
    int ky;

    int radius;


    if (input == NULL ||
        kernel == NULL)
    {
        return 0;
    }


    if (input->data == NULL ||
        kernel->data == NULL)
    {
        return 0;
    }


    if (!validate_f32_output(
            output,
            input->width,
            input->height))
    {
        return 0;
    }


    radius =
            kernel->size / 2;


    memset(
            output->data,
            0,
            (size_t) output->width *
            (size_t) output->height *
            sizeof(float)
    );


    for (y = radius;
         y < input->height - radius;
         ++y)
    {
        for (x = radius;
             x < input->width - radius;
             ++x)
        {
            float accumulator = 0.0f;


            for (ky = -radius;
                 ky <= radius;
                 ++ky)
            {
                for (kx = -radius;
                     kx <= radius;
                     ++kx)
                {
                    const size_t inputIndex =
                            (size_t) (y + ky)
                            *
                            input->width
                            +
                            (x + kx);


                    const size_t kernelIndex =
                            (size_t) (ky + radius)
                            *
                            kernel->size
                            +
                            (kx + radius);


                    accumulator +=
                            input->data[inputIndex]
                            *
                            kernel->data[kernelIndex];
                }
            }


            output->data[
                    (size_t) y *
                    output->width
                    +
                    x
            ] =
                    accumulator;
        }
    }


    return 1;
}


int convolution_1d_horizontal_u8_f32_into(
        const ImageU8 *input,
        const Kernel1D *kernel,
        ImageF32 *output)
{
    int x;
    int y;
    int kx;

    int radius;


    if (input == NULL ||
        kernel == NULL)
    {
        return 0;
    }


    if (input->data == NULL ||
        kernel->data == NULL)
    {
        return 0;
    }


    if (!validate_f32_output(
            output,
            input->width,
            input->height))
    {
        return 0;
    }


    radius =
            kernel->size / 2;


    memset(
            output->data,
            0,
            (size_t) output->width *
            (size_t) output->height *
            sizeof(float)
    );


    for (y = 0;
         y < input->height;
         ++y)
    {
        for (x = radius;
             x < input->width - radius;
             ++x)
        {
            float accumulator = 0.0f;


            for (kx = -radius;
                 kx <= radius;
                 ++kx)
            {
                const size_t inputIndex =
                        (size_t) y *
                        input->width
                        +
                        (x + kx);


                const int kernelIndex =
                        kx + radius;


                accumulator +=
                        (float) input->data[inputIndex]
                        *
                        kernel->data[kernelIndex];
            }


            output->data[
                    (size_t) y *
                    output->width
                    +
                    x
            ] =
                    accumulator;
        }
    }


    return 1;
}


int convolution_1d_vertical_f32_into(
        const ImageF32 *input,
        const Kernel1D *kernel,
        ImageF32 *output)
{
    int x;
    int y;
    int ky;

    int radius;


    if (input == NULL ||
        kernel == NULL)
    {
        return 0;
    }


    if (input->data == NULL ||
        kernel->data == NULL)
    {
        return 0;
    }


    if (!validate_f32_output(
            output,
            input->width,
            input->height))
    {
        return 0;
    }


    radius =
            kernel->size / 2;


    memset(
            output->data,
            0,
            (size_t) output->width *
            (size_t) output->height *
            sizeof(float)
    );


    for (y = radius;
         y < input->height - radius;
         ++y)
    {
        for (x = radius;
             x < input->width - radius;
             ++x)
        {
            float accumulator = 0.0f;


            for (ky = -radius;
                 ky <= radius;
                 ++ky)
            {
                const size_t inputIndex =
                        (size_t) (y + ky)
                        *
                        input->width
                        +
                        x;


                const int kernelIndex =
                        ky + radius;


                accumulator +=
                        input->data[inputIndex]
                        *
                        kernel->data[kernelIndex];
            }


            output->data[
                    (size_t) y *
                    output->width
                    +
                    x
            ] =
                    accumulator;
        }
    }


    return 1;
}


int convolution_gaussian_separable_into(
        const ImageU8 *input,
        const Kernel1D *kernel,
        ImageF32 *intermediate,
        ImageF32 *output)
{
    if (!convolution_1d_horizontal_u8_f32_into(
            input,
            kernel,
            intermediate))
    {
        return 0;
    }


    if (!convolution_1d_vertical_f32_into(
            intermediate,
            kernel,
            output))
    {
        return 0;
    }


    return 1;
}


int gradient_magnitude_f32_into(
        const ImageF32 *gx,
        const ImageF32 *gy,
        ImageF32 *magnitude)
{
    int x;
    int y;


    if (gx == NULL ||
        gy == NULL)
    {
        return 0;
    }


    if (gx->data == NULL ||
        gy->data == NULL)
    {
        return 0;
    }


    if (!validate_f32_output(
            magnitude,
            gx->width,
            gx->height))
    {
        return 0;
    }


    if (gx->width != gy->width ||
        gx->height != gy->height)
    {
        return 0;
    }


    for (y = 0;
         y < gx->height;
         ++y)
    {
        for (x = 0;
             x < gx->width;
             ++x)
        {
            const size_t index =
                    (size_t) y *
                    gx->width
                    +
                    x;


            const float valueX =
                    gx->data[index];


            const float valueY =
                    gy->data[index];


            magnitude->data[index] =
                    sqrtf(
                            valueX * valueX
                            +
                            valueY * valueY
                    );
        }
    }


    return 1;
}