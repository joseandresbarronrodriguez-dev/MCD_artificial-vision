//
// Created by andru on 9/5/2026.
//
#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "image.h"
#include "kernels.h"

/*
 * ============================================================
 * Archivo: convolution.h
 * Práctica: Convolución separable y fusión de kernels
 *
 * Descripción:
 *   Define las funciones de convolución espacial implementadas
 *   manualmente en C.
 *
 * Objetivo:
 *   Medir posteriormente el costo computacional de distintas
 *   estrategias de filtrado sobre PC y ARM64.
 *
 * Importante:
 *   Este módulo NO utiliza OpenCV para realizar convoluciones.
 * ============================================================
 */


/**
 * @brief Representa una imagen de punto flotante de un canal.
 *
 * Se utiliza como salida intermedia de las convoluciones para
 * evitar pérdidas de precisión durante operaciones sucesivas.
 */
typedef struct
{
    int width;
    int height;
    float *data;

} ImageF32;


#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Reserva memoria para una ImageF32.
 *
 * @param image  Estructura de salida.
 * @param width  Ancho de la imagen.
 * @param height Alto de la imagen.
 *
 * @return 1 si la reserva fue exitosa.
 * @return 0 en caso de error.
 */
int image_f32_create(ImageF32 *image,
                     int width,
                     int height);


/**
 * @brief Libera la memoria asociada a una ImageF32.
 *
 * @param image Imagen a liberar.
 */
void image_f32_free(ImageF32 *image);


/**
 * @brief Aplica una convolución 2D directa.
 *
 * La operación se calcula píxel por píxel:
 *
 *     output(x,y) =
 *         sum_i sum_j
 *         input(x+i, y+j) * kernel(i,j)
 *
 * La implementación utiliza una política de borde simple:
 * los píxeles donde el kernel no cabe completamente se dejan
 * con valor cero.
 *
 * Complejidad aproximada:
 *
 *     O(width * height * k^2)
 *
 * @param input  Imagen de entrada de 8 bits.
 * @param kernel Kernel 2D cuadrado.
 * @param output Imagen de salida en punto flotante.
 *
 * @return 1 si la operación fue exitosa.
 * @return 0 en caso de error.
 */
int convolution_2d_direct_u8_f32(const ImageU8 *input,
                                 const Kernel2D *kernel,
                                 ImageF32 *output);

    /**
     * @brief Aplica convolución horizontal 1D sobre una ImageU8.
     *
     * La entrada es una imagen de 8 bits y la salida se almacena
     * como float para conservar precisión.
     *
     * Solo se procesan posiciones donde el kernel cabe
     * completamente en la dirección horizontal.
     *
     * Complejidad aproximada:
     *
     *     O(width * height * k)
     *
     * @param input  Imagen de entrada.
     * @param kernel Kernel 1D.
     * @param output Imagen de salida.
     *
     * @return 1 si la operación fue exitosa.
     * @return 0 en caso de error.
     */
    int convolution_1d_horizontal_u8_f32(
            const ImageU8 *input,
            const Kernel1D *kernel,
            ImageF32 *output);


    /**
     * @brief Aplica convolución vertical 1D sobre una ImageF32.
     *
     * Esta función constituye la segunda etapa de la convolución
     * gaussiana separable.
     *
     * La entrada ya es float porque proviene de la primera pasada
     * horizontal.
     *
     * Complejidad aproximada:
     *
     *     O(width * height * k)
     *
     * @param input  Imagen float de entrada.
     * @param kernel Kernel 1D.
     * @param output Imagen float de salida.
     *
     * @return 1 si la operación fue exitosa.
     * @return 0 en caso de error.
     */
    int convolution_1d_vertical_f32(
            const ImageF32 *input,
            const Kernel1D *kernel,
            ImageF32 *output);


    /**
     * @brief Aplica un filtro gaussiano separable completo.
     *
     * Ejecuta:
     *
     *     1. Convolución horizontal.
     *     2. Convolución vertical.
     *
     * El resultado debe ser numéricamente equivalente al obtenido
     * utilizando el kernel gaussiano 2D generado mediante producto
     * exterior.
     *
     * @param input        Imagen original.
     * @param kernel       Kernel gaussiano 1D.
     * @param intermediate Buffer intermedio horizontal.
     * @param output       Resultado final.
     *
     * @return 1 si la operación fue exitosa.
     * @return 0 en caso de error.
     */
    int convolution_gaussian_separable(
            const ImageU8 *input,
            const Kernel1D *kernel,
            ImageF32 *intermediate,
            ImageF32 *output);

        /**
         * @brief Aplica una convolución 2D sobre una ImageF32.
         *
         * Se utiliza principalmente para aplicar Sobel sobre una imagen
         * previamente suavizada.
         *
         * @param input  Imagen float de entrada.
         * @param kernel Kernel 2D.
         * @param output Imagen float resultante.
         *
         * @return 1 si la operación fue exitosa.
         * @return 0 en caso de error.
         */
        int convolution_2d_f32_f32(
                const ImageF32 *input,
                const Kernel2D *kernel,
                ImageF32 *output);


        /**
         * @brief Calcula la magnitud del gradiente.
         *
         * Para cada píxel:
         *
         *      M = sqrt(Gx^2 + Gy^2)
         *
         * @param gx Gradiente horizontal.
         * @param gy Gradiente vertical.
         * @param magnitude Imagen de salida.
         *
         * @return 1 si la operación fue exitosa.
         * @return 0 en caso de error.
         */
        int gradient_magnitude_f32(
                const ImageF32 *gx,
                const ImageF32 *gy,
                ImageF32 *magnitude);

        /**
 * @brief Convolución 2D U8 -> F32 usando un buffer ya reservado.
 *
 * El tamaño de output debe coincidir con input.
 *
 * No reserva memoria.
 */
        int convolution_2d_direct_u8_f32_into(
                const ImageU8 *input,
                const Kernel2D *kernel,
                ImageF32 *output);


        /**
         * @brief Convolución 2D F32 -> F32 usando un buffer ya reservado.
         *
         * No reserva memoria.
         */
        int convolution_2d_f32_f32_into(
                const ImageF32 *input,
                const Kernel2D *kernel,
                ImageF32 *output);


        /**
         * @brief Convolución horizontal 1D U8 -> F32
         * usando un buffer ya reservado.
         */
        int convolution_1d_horizontal_u8_f32_into(
                const ImageU8 *input,
                const Kernel1D *kernel,
                ImageF32 *output);


        /**
         * @brief Convolución vertical 1D F32 -> F32
         * usando un buffer ya reservado.
         */
        int convolution_1d_vertical_f32_into(
                const ImageF32 *input,
                const Kernel1D *kernel,
                ImageF32 *output);


        /**
         * @brief Gaussiano separable utilizando buffers preasignados.
         */
        int convolution_gaussian_separable_into(
                const ImageU8 *input,
                const Kernel1D *kernel,
                ImageF32 *intermediate,
                ImageF32 *output);


        /**
         * @brief Calcula magnitud sobre un buffer ya reservado.
         */
        int gradient_magnitude_f32_into(
                const ImageF32 *gx,
                const ImageF32 *gy,
                ImageF32 *magnitude);

#ifdef __cplusplus
}
#endif

#endif