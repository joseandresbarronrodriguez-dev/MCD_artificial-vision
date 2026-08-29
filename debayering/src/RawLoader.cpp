//
// Created by andru on 8/23/2026.
//
#include "RawLoader.h"

#include <libraw/libraw.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

RawImageData RawLoader::load(const std::string& filePath)
{
    LibRaw processor;

    // ========================================================
    // 1. Abrir archivo RAW
    // ========================================================

    int result = processor.open_file(filePath.c_str());

    if (result != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(
            "No se pudo abrir el archivo RAW: " +
            std::string(libraw_strerror(result))
        );
    }

    // ========================================================
    // 2. Desempaquetar datos RAW
    // ========================================================

    result = processor.unpack();

    if (result != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(
            "No se pudo desempaquetar el RAW: " +
            std::string(libraw_strerror(result))
        );
    }

    // ========================================================
    // 3. Comprobar que existe raw_image
    // ========================================================

    if (processor.imgdata.rawdata.raw_image == nullptr)
    {
        throw std::runtime_error(
            "LibRaw no proporciono un plano Bayer raw_image."
        );
    }

    RawImageData output;

    // ========================================================
    // 4. Metadatos de cámara
    // ========================================================

    output.cameraMake =
        processor.imgdata.idata.make;

    output.cameraModel =
        processor.imgdata.idata.model;

    // ========================================================
    // 5. Dimensiones RAW
    // ========================================================

    output.rawWidth =
        processor.imgdata.sizes.raw_width;

    output.rawHeight =
        processor.imgdata.sizes.raw_height;

    output.width =
        processor.imgdata.sizes.width;

    output.height =
        processor.imgdata.sizes.height;

    output.topMargin =
        processor.imgdata.sizes.top_margin;

    output.leftMargin =
        processor.imgdata.sizes.left_margin;

    output.maximum =
        processor.imgdata.color.maximum;

    output.blackLevel =
        processor.imgdata.color.black;

    // ========================================================
    // Orientación de la cámara
    // ========================================================

    output.orientation =
        processor.imgdata.sizes.flip;

    // ========================================================
    // Balance de blancos de cámara
    // ========================================================

    /*
     * LibRaw proporciona cam_mul con los multiplicadores
     * almacenados por la cámara para los componentes del CFA.
     *
     * Para un Bayer clásico normalmente corresponden a:
     *
     * cam_mul[0] -> R
     * cam_mul[1] -> G1
     * cam_mul[2] -> B
     * cam_mul[3] -> G2
     *
     * Los dos verdes pertenecen a las dos posiciones verdes
     * existentes dentro de la celda Bayer de 2x2.
     */

    float wbR =
        processor.imgdata.color.cam_mul[0];

    float wbG1 =
        processor.imgdata.color.cam_mul[1];

    float wbB =
        processor.imgdata.color.cam_mul[2];

    float wbG2 =
        processor.imgdata.color.cam_mul[3];

    /*
     * Algunos archivos pueden no proporcionar el segundo verde.
     * Si existe, utilizamos el promedio G1/G2.
     */
    float wbG = wbG1;

    if (wbG2 > 0.0f)
    {
        wbG =
            (wbG1 + wbG2) / 2.0f;
    }


    /*
     * Validamos que los multiplicadores sean utilizables.
     *
     * Si cam_mul no contiene información válida recurrimos a
     * pre_mul, que proporciona multiplicadores alternativos
     * calculados por LibRaw.
     */
    if (
        wbR <= 0.0f ||
        wbG <= 0.0f ||
        wbB <= 0.0f
    )
    {
        wbR =
            processor.imgdata.color.pre_mul[0];

        wbG1 =
            processor.imgdata.color.pre_mul[1];

        wbB =
            processor.imgdata.color.pre_mul[2];

        wbG2 =
            processor.imgdata.color.pre_mul[3];

        wbG = wbG1;

        if (wbG2 > 0.0f)
        {
            wbG =
                (wbG1 + wbG2) / 2.0f;
        }
    }


    /*
     * Normalizamos respecto al verde.
     *
     * Ejemplo:
     *
     * R = 2.10
     * G = 1.00
     * B = 1.65
     *
     * Así evitamos cambiar innecesariamente la luminancia
     * global de la imagen.
     */
    if (wbG > 0.0f)
    {
        output.whiteBalanceR =
            wbR / wbG;

        output.whiteBalanceG =
            1.0f;

        output.whiteBalanceB =
            wbB / wbG;
    }

    // ========================================================
    // Matriz de corrección de color
    // ========================================================

    /*
     * LibRaw proporciona rgb_cam como una matriz 3x4.
     *
     * Para sensores Bayer clásicos trabajamos con tres canales
     * reconstruidos (R, G, B), por lo que usaremos las primeras
     * tres columnas para construir una matriz 3x3.
     *
     * Esta matriz se utilizará únicamente con fines de visualización.
     */
    bool validMatrix = false;

    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            if (processor.imgdata.color.rgb_cam[row][col] != 0.0f)
            {
                validMatrix = true;
            }
        }
    }

    if (validMatrix)
    {
        output.cameraToRgb = cv::Matx33f(
            static_cast<float>(processor.imgdata.color.rgb_cam[0][0]),
            static_cast<float>(processor.imgdata.color.rgb_cam[0][1]),
            static_cast<float>(processor.imgdata.color.rgb_cam[0][2]),

            static_cast<float>(processor.imgdata.color.rgb_cam[1][0]),
            static_cast<float>(processor.imgdata.color.rgb_cam[1][1]),
            static_cast<float>(processor.imgdata.color.rgb_cam[1][2]),

            static_cast<float>(processor.imgdata.color.rgb_cam[2][0]),
            static_cast<float>(processor.imgdata.color.rgb_cam[2][1]),
            static_cast<float>(processor.imgdata.color.rgb_cam[2][2])
        );

        output.hasColorMatrix = true;
    }

    // ========================================================
    // 6. Detectar patrón Bayer
    // ========================================================

    output.bayerPattern =
        detectBayerPattern(processor);

    // ========================================================
    // 7. Crear Mat de 16 bits
    // ========================================================

    output.rawImage = cv::Mat(
        output.height,
        output.width,
        CV_16UC1
    );

    const auto* raw =
        processor.imgdata.rawdata.raw_image;

    // LibRaw expresa raw_pitch en bytes.
    std::size_t pixelsPerRow =
        processor.imgdata.sizes.raw_pitch /
        sizeof(std::uint16_t);

    // Algunos archivos pueden no proporcionar un pitch
    // diferente al ancho.
    if (pixelsPerRow == 0)
    {
        pixelsPerRow = output.rawWidth;
    }

    // ========================================================
    // 8. Copiar únicamente el área visible
    // ========================================================

    for (int y = 0; y < output.height; ++y)
    {
        auto* destination =
            output.rawImage.ptr<std::uint16_t>(y);

        const int rawY =
            y + output.topMargin;

        for (int x = 0; x < output.width; ++x)
        {
            const int rawX =
                x + output.leftMargin;

            destination[x] =
                raw[
                    static_cast<std::size_t>(rawY) *
                    pixelsPerRow +
                    rawX
                ];
        }
    }

    // ========================================================
    // 9. Cerrar LibRaw
    // ========================================================

    processor.recycle();

    return output;
}

std::string RawLoader::detectBayerPattern(
    LibRaw& processor)
{
    std::string pattern;

    // Analizamos el bloque CFA de 2x2.
    for (int row = 0; row < 2; ++row)
    {
        for (int col = 0; col < 2; ++col)
        {
            const int colorIndex =
                processor.COLOR(row, col);

            char color =
                processor.imgdata.idata.cdesc[colorIndex];

            // Algunos sensores diferencian dos verdes.
            // Para nuestro patrón Bayer ambos son G.
            if (color == 'g')
            {
                color = 'G';
            }

            pattern += color;
        }
    }

    // Validamos los cuatro patrones Bayer clásicos.
    if (
        pattern != "RGGB" &&
        pattern != "BGGR" &&
        pattern != "GRBG" &&
        pattern != "GBRG"
    )
    {
        return "UNKNOWN(" + pattern + ")";
    }

    return pattern;
}