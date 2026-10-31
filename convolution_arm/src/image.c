//
// Created by andru on 9/5/2026.
//
#include "image.h"

#include <stdlib.h>
#include <string.h>


int image_u8_create(ImageU8 *image,
                    int width,
                    int height)
{
    size_t total_bytes;

    if (image == NULL)
    {
        return 0;
    }

    if (width <= 0 || height <= 0)
    {
        return 0;
    }

    total_bytes = (size_t) width * (size_t) height;

    image->data = (uint8_t *) malloc(total_bytes);

    if (image->data == NULL)
    {
        image->width = 0;
        image->height = 0;

        return 0;
    }

    image->width = width;
    image->height = height;

    return 1;
}


int image_u8_copy_from_buffer(ImageU8 *image,
                              const uint8_t *src,
                              int width,
                              int height,
                              size_t src_stride)
{
    int y;

    if (image == NULL || src == NULL)
    {
        return 0;
    }

    if (!image_u8_create(image, width, height))
    {
        return 0;
    }

    /*
     * Copiamos fila por fila.
     *
     * Así no suponemos que la imagen proveniente
     * de OpenCV sea necesariamente continua.
     */
    for (y = 0; y < height; ++y)
    {
        const uint8_t *src_row =
                src + (size_t) y * src_stride;

        uint8_t *dst_row =
                image->data + (size_t) y * width;

        memcpy(dst_row,
               src_row,
               (size_t) width);
    }

    return 1;
}


void image_u8_free(ImageU8 *image)
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