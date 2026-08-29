/**
* @file ImageUtils.h
 * @brief Utilidades para guardar y visualizar imágenes reconstruidas.
 *
 * Este módulo mantiene separada la reconstrucción Bayer de las
 * operaciones destinadas únicamente a visualización y almacenamiento.
 *
 * La imagen generada por Debayer se mantiene internamente en RGB,
 * mientras que OpenCV utiliza BGR para escritura y visualización.
 */

#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <opencv2/core.hpp>

#include <string>
#include <algorithm>
#include <stdexcept>

class ImageUtils
{
public:

    /**
     * @brief Guarda directamente la reconstrucción RGB de 16 bits.
     *
     * No aplica balance de blancos, gamma ni corrección de color.
     * Únicamente convierte de RGB a BGR porque OpenCV utiliza
     * internamente el orden BGR.
     *
     * @param rgb Imagen CV_16UC3 en orden RGB.
     * @param outputPath Ruta del archivo PNG.
     */
    static void saveLinear16(
        const cv::Mat& rgb,
        const std::string& outputPath
    );


    /**
     * @brief Genera una versión de 8 bits adecuada para visualización.
     *
     * Para producir la vista previa se realizan únicamente
     * transformaciones destinadas a visualización:
     *
     * 1. Eliminación aproximada del nivel negro.
     * 2. Normalización utilizando el nivel de saturación del sensor.
     * 3. Aplicación de gamma.
     * 4. Conversión RGB -> BGR.
     *
     * Estas operaciones NO modifican el resultado utilizado para
     * evaluar el algoritmo de debayering.
     *
     * @param rgb Imagen RGB lineal CV_16UC3.
     * @param blackLevel Nivel negro reportado por LibRaw.
     * @param maximum Valor máximo reportado por LibRaw.
     * @param outputPath Archivo PNG de salida.
     */
    static void savePreview8(
        const cv::Mat& rgb,
        unsigned int blackLevel,
        unsigned int maximum,
        float wbR,
        float wbG,
        float wbB,
        const cv::Matx33f& colorMatrix,
        const std::string& outputPath
    );
    static cv::Mat applyOrientation(
        const cv::Mat& image,
        int orientation
    );

    static void saveComparison(
        const cv::Mat& left,
        const cv::Mat& right,
        const std::string& leftLabel,
        const std::string& rightLabel,
        const std::string& outputPath
    );

    static cv::Mat createPreview8(
        const cv::Mat& rgb,
        unsigned int blackLevel,
        unsigned int maximum,
        float wbR,
        float wbG,
        float wbB,
        const cv::Matx33f& colorMatrix
    );
};

#endif