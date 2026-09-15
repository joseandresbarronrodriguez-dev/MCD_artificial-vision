//
// Created by andru on 9/5/2026.
//
#ifndef IMAGE_H
#define IMAGE_H

#include <stddef.h>
#include <stdint.h>

/*
 * Imagen de 8 bits y un canal.
 *
 * Los píxeles se almacenan de forma lineal en memoria:
 *
 * pixel(x, y) = data[y * width + x]
 *
 * Ejemplo 4x3:
 *
 * p00 p01 p02 p03
 * p10 p11 p12 p13
 * p20 p21 p22 p23
 *
 * Memoria:
 *
 * [p00][p01][p02][p03][p10][p11][p12][p13][p20]...
 */
typedef struct
{
    int width;
    int height;
    uint8_t *data;

} ImageU8;


/*
 * Estas funciones están implementadas en C.
 *
 * Cuando image.h se incluye desde C++, extern "C"
 * evita que el compilador C++ modifique los nombres
 * de las funciones durante el enlazado.
 */
#ifdef __cplusplus
extern "C" {
#endif

    /*
     * Reserva memoria para una imagen width x height.
     *
     * Retorna:
     * 1 -> éxito
     * 0 -> error
     */
    int image_u8_create(ImageU8 *image,
                        int width,
                        int height);


    /*
     * Copia una imagen existente hacia una ImageU8.
     *
     * src_stride representa cuántos bytes existen
     * entre el inicio de una fila y la siguiente.
     *
     * Esto permite recibir imágenes desde OpenCV
     * incluso cuando cv::Mat tiene padding.
     */
    int image_u8_copy_from_buffer(ImageU8 *image,
                                  const uint8_t *src,
                                  int width,
                                  int height,
                                  size_t src_stride);


    /*
     * Libera la memoria reservada.
     */
    void image_u8_free(ImageU8 *image);


#ifdef __cplusplus
}
#endif

#endif