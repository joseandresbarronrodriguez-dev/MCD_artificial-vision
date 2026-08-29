//
// Created by andru on 8/24/2026.
//
/**
 * @file LibRawReference.cpp
 * @brief Implementación de la reconstrucción de referencia.
 *
 * LibRaw realiza:
 *
 * - Corrección de nivel negro.
 * - White balance de cámara.
 * - Demosaicing.
 * - Conversión al espacio sRGB.
 * - Función de transferencia para visualización.
 *
 * La rotación se desactiva deliberadamente para aplicarla
 * posteriormente de la misma manera que a nuestro resultado.
 */

#include "LibRawReference.h"

#include <libraw/libraw.h>

#include <stdexcept>


cv::Mat LibRawReference::render(
    const std::string& filePath,
    int quality)
{
    LibRaw processor;


    // ========================================================
    // 1. Parámetros de procesamiento
    // ========================================================

    /*
     * Utilizamos el white balance almacenado por la cámara.
     */
    processor.imgdata.params.use_camera_wb = 1;

    /*
     * Espacio de salida:
     *
     * 1 -> sRGB.
     */
    processor.imgdata.params.output_color = 1;

    /*
     * Calidad del algoritmo de demosaicing.
     *
     * 0 = Linear
     * 3 = AHD
     */
    processor.imgdata.params.user_qual =
        quality;

    /*
     * Desactivamos el ajuste automático de brillo para
     * mantener una comparación más controlada.
     */
    processor.imgdata.params.no_auto_bright = 1;

    /*
     * La referencia visual será de 8 bits.
     */
    processor.imgdata.params.output_bps = 8;

    /*
     * Desactivamos la rotación de LibRaw.
     *
     * Posteriormente aplicaremos exactamente la misma
     * transformación a nuestro resultado y a la referencia.
     */
    processor.imgdata.params.user_flip = 0;

    /*
     * Función de transferencia sRGB.
     */
    processor.imgdata.params.gamm[0] =
        1.0 / 2.4;

    processor.imgdata.params.gamm[1] =
        12.92;


    // ========================================================
    // 2. Abrir archivo
    // ========================================================

    int result =
        processor.open_file(
            filePath.c_str()
        );

    if (result != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(
            "LibRawReference: error al abrir RAW: " +
            std::string(
                libraw_strerror(result)
            )
        );
    }


    // ========================================================
    // 3. Desempaquetar RAW
    // ========================================================

    result =
        processor.unpack();

    if (result != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(
            "LibRawReference: error en unpack."
        );
    }


    // ========================================================
    // 4. Procesamiento completo LibRaw
    // ========================================================

    result =
        processor.dcraw_process();

    if (result != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(
            "LibRawReference: error en dcraw_process."
        );
    }


    // ========================================================
    // 5. Obtener dimensiones de salida
    // ========================================================

    int width = 0;
    int height = 0;
    int colors = 0;
    int bits = 0;

    processor.get_mem_image_format(
        &width,
        &height,
        &colors,
        &bits
    );

    if (colors != 3 || bits != 8)
    {
        throw std::runtime_error(
            "LibRawReference: formato inesperado."
        );
    }


    // ========================================================
    // 6. Crear imagen OpenCV
    // ========================================================

    cv::Mat reference(
        height,
        width,
        CV_8UC3
    );

    /*
     * copy_mem_image permite copiar directamente en BGR.
     *
     * bgr = 1
     */
    result =
        processor.copy_mem_image(
            reference.data,
            static_cast<int>(
                reference.step
            ),
            1
        );

    if (result != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(
            "LibRawReference: error copiando imagen."
        );
    }

    processor.recycle();

    return reference;
}