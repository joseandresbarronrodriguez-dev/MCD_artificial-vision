//
// Created by andru on 8/24/2026.
//
/**
 * @file LibRawReference.h
 * @brief Generación de imágenes de referencia mediante LibRaw.
 *
 * Este módulo permite comparar nuestro algoritmo de debayering
 * con implementaciones maduras disponibles en LibRaw.
 *
 * Se utiliza exclusivamente como referencia experimental.
 */

#ifndef LIBRAW_REFERENCE_H
#define LIBRAW_REFERENCE_H

#include <opencv2/core.hpp>

#include <string>


class LibRawReference
{
public:

    /**
     * @brief Procesa un archivo RAW utilizando LibRaw.
     *
     * Calidad de interpolación:
     *
     * 0 -> Linear
     * 1 -> VNG
     * 2 -> PPG
     * 3 -> AHD
     *
     * @param filePath Ruta al archivo RAW.
     * @param quality Método de demosaicing LibRaw.
     *
     * @return Imagen BGR de 8 bits sin rotación.
     */
    static cv::Mat render(
        const std::string& filePath,
        int quality
    );
};

#endif