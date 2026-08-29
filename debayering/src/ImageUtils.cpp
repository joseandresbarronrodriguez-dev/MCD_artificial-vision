/**
 * @file ImageUtils.cpp
 * @brief Implementación de utilidades para almacenar resultados RGB.
 */

#include "ImageUtils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>


void ImageUtils::saveLinear16(
    const cv::Mat& rgb,
    const std::string& outputPath)
{
    if (rgb.empty())
    {
        throw std::invalid_argument(
            "No se puede guardar una imagen RGB vacia."
        );
    }

    if (rgb.type() != CV_16UC3)
    {
        throw std::invalid_argument(
            "saveLinear16 requiere una imagen CV_16UC3."
        );
    }

    /*
     * Nuestro algoritmo utiliza:
     *
     * [0] R
     * [1] G
     * [2] B
     *
     * OpenCV espera:
     *
     * [0] B
     * [1] G
     * [2] R
     *
     * Por ello realizamos únicamente un intercambio de canales.
     */
    cv::Mat bgr;

    cv::cvtColor(
        rgb,
        bgr,
        cv::COLOR_RGB2BGR
    );

    if (!cv::imwrite(outputPath, bgr))
    {
        throw std::runtime_error(
            "No se pudo guardar: " + outputPath
        );
    }
}


void ImageUtils::savePreview8(
    const cv::Mat& rgb,
    unsigned int blackLevel,
    unsigned int maximum,
    float wbR,
    float wbG,
    float wbB,
    const cv::Matx33f& colorMatrix,
    const std::string& outputPath)
{
    cv::Mat preview =
        createPreview8(
            rgb,
            blackLevel,
            maximum,
            wbR,
            wbG,
            wbB,
            colorMatrix
        );

    if (!cv::imwrite(
            outputPath,
            preview
        ))
    {
        throw std::runtime_error(
            "No se pudo guardar preview: " +
            outputPath
        );
    }
}

static void saveComparison(
    const cv::Mat& left,
    const cv::Mat& right,
    const std::string& leftLabel,
    const std::string& rightLabel,
    const std::string& outputPath
);

/**
 * @brief Aplica la orientación almacenada en los metadatos RAW.
 *
 * LibRaw entrega el mosaico RAW utilizando la orientación física
 * del sensor. Por esta razón la transformación se realiza después
 * del proceso de debayering.
 *
 * Valores utilizados:
 *
 * 0 -> Sin rotación.
 * 3 -> 180 grados.
 * 5 -> 90 grados antihorario.
 * 6 -> 90 grados horario.
 *
 * @param image Imagen reconstruida.
 * @param orientation Código de orientación reportado por LibRaw.
 *
 * @return Imagen correctamente orientada.
 */
cv::Mat ImageUtils::applyOrientation(
    const cv::Mat& image,
    int orientation)
{
    if (image.empty())
    {
        throw std::invalid_argument(
            "No se puede orientar una imagen vacia."
        );
    }

    cv::Mat result;

    switch (orientation)
    {
        /*
         * Imagen almacenada en la orientación correcta.
         */
        case 0:
        {
            result = image.clone();
            break;
        }

            /*
             * Rotación de 180 grados.
             */
        case 3:
        {
            cv::rotate(
                image,
                result,
                cv::ROTATE_180
            );

            break;
        }

            /*
             * Rotación de 90 grados antihorario.
             */
        case 5:
        {
            cv::rotate(
                image,
                result,
                cv::ROTATE_90_COUNTERCLOCKWISE
            );

            break;
        }

            /*
             * Rotación de 90 grados horario.
             */
        case 6:
        {
            cv::rotate(
                image,
                result,
                cv::ROTATE_90_CLOCKWISE
            );

            break;
        }

            /*
             * En caso de recibir una orientación no contemplada
             * conservamos la imagen original.
             */
        default:
        {
            result = image.clone();
            break;
        }
    }

    return result;
}

/**
 * @brief Genera una comparación visual lado a lado.
 *
 * La función crea versiones reducidas de ambas imágenes para
 * facilitar la inspección visual de los resultados.
 *
 * Cada lado recibe además una etiqueta indicando el método
 * utilizado para reconstruir la imagen.
 *
 * Las imágenes originales no son modificadas.
 *
 * @param left Imagen ubicada a la izquierda.
 * @param right Imagen ubicada a la derecha.
 * @param leftLabel Etiqueta de la imagen izquierda.
 * @param rightLabel Etiqueta de la imagen derecha.
 * @param outputPath Ruta donde se guardará la comparación.
 */
void ImageUtils::saveComparison(
    const cv::Mat& left,
    const cv::Mat& right,
    const std::string& leftLabel,
    const std::string& rightLabel,
    const std::string& outputPath)
{
    // ========================================================
    // Validación
    // ========================================================

    if (left.empty() || right.empty())
    {
        throw std::invalid_argument(
            "Las imagenes de comparacion no pueden estar vacias."
        );
    }


    // ========================================================
    // Altura máxima utilizada únicamente para visualización
    // ========================================================

    constexpr int maxHeight = 1000;


    // ========================================================
    // Función auxiliar para preparar cada imagen
    // ========================================================

    /*
     * Capturamos explícitamente maxHeight.
     *
     * De esta manera la lambda puede utilizar la variable
     * definida fuera de su cuerpo.
     */
    auto prepare =
        [maxHeight](
            const cv::Mat& input,
            const std::string& label
        ) -> cv::Mat
    {
        cv::Mat resized;


        // ----------------------------------------------------
        // Calcular escala
        // ----------------------------------------------------

        const double scale =
            std::min(
                1.0,
                static_cast<double>(maxHeight) /
                static_cast<double>(input.rows)
            );


        // ----------------------------------------------------
        // Redimensionar
        // ----------------------------------------------------

        cv::resize(
            input,
            resized,
            cv::Size(),
            scale,
            scale,
            cv::INTER_AREA
        );


        // ----------------------------------------------------
        // Crear espacio superior para la etiqueta
        // ----------------------------------------------------

        cv::Mat labeled;

        cv::copyMakeBorder(
            resized,
            labeled,
            60,
            0,
            0,
            0,
            cv::BORDER_CONSTANT,
            cv::Scalar(25, 25, 25)
        );


        // ----------------------------------------------------
        // Escribir nombre del método
        // ----------------------------------------------------

        cv::putText(
            labeled,
            label,
            cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX,
            0.9,
            cv::Scalar(255, 255, 255),
            2,
            cv::LINE_AA
        );

        return labeled;
    };


    // ========================================================
    // Preparar ambas imágenes
    // ========================================================

    cv::Mat leftPrepared =
        prepare(
            left,
            leftLabel
        );

    cv::Mat rightPrepared =
        prepare(
            right,
            rightLabel
        );


    // ========================================================
    // Concatenar horizontalmente
    // ========================================================

    cv::Mat comparison;

    cv::hconcat(
        leftPrepared,
        rightPrepared,
        comparison
    );


    // ========================================================
    // Guardar resultado
    // ========================================================

    if (!cv::imwrite(
            outputPath,
            comparison
        ))
    {
        throw std::runtime_error(
            "No se pudo guardar comparacion: " +
            outputPath
        );
    }
}

cv::Mat ImageUtils::createPreview8(
    const cv::Mat& rgb,
    unsigned int blackLevel,
    unsigned int maximum,
    float wbR,
    float wbG,
    float wbB,
    const cv::Matx33f& colorMatrix)
{
    if (rgb.empty())
    {
        throw std::invalid_argument(
            "No se puede crear preview de una imagen vacia."
        );
    }

    if (rgb.type() != CV_16UC3)
    {
        throw std::invalid_argument(
            "createPreview8 requiere CV_16UC3."
        );
    }

    if (maximum <= blackLevel)
    {
        throw std::invalid_argument(
            "Rango RAW invalido."
        );
    }


    // ========================================================
    // 1. Conversión a punto flotante
    // ========================================================

    cv::Mat floating;

    rgb.convertTo(
        floating,
        CV_32FC3
    );


    // ========================================================
    // 2. Corrección de nivel negro
    // ========================================================

    floating -= static_cast<float>(
        blackLevel
    );

    cv::max(
        floating,
        0.0,
        floating
    );


    // ========================================================
    // 3. Normalización
    // ========================================================

    const float sensorRange =
        static_cast<float>(
            maximum - blackLevel
        );

    floating /= sensorRange;


    // ========================================================
    // 4. Balance de blancos
    // ========================================================

    std::vector<cv::Mat> channels;

    cv::split(
        floating,
        channels
    );

    channels[0] *= wbR;
    channels[1] *= wbG;
    channels[2] *= wbB;

    cv::merge(
        channels,
        floating
    );


    // ========================================================
    // 5. Corrección de color
    // ========================================================

    cv::transform(
        floating,
        floating,
        colorMatrix
    );


    // ========================================================
    // 6. Limitar rango
    // ========================================================

    cv::max(
        floating,
        0.0,
        floating
    );

    cv::min(
        floating,
        1.0,
        floating
    );


    // ========================================================
    // 7. Transferencia sRGB
    // ========================================================

    for (int row = 0; row < floating.rows; ++row)
    {
        auto* pixels =
            floating.ptr<cv::Vec3f>(row);

        for (int col = 0; col < floating.cols; ++col)
        {
            for (int channel = 0;
                 channel < 3;
                 ++channel)
            {
                float& value =
                    pixels[col][channel];

                if (value <= 0.0031308f)
                {
                    value =
                        12.92f * value;
                }
                else
                {
                    value =
                        1.055f *
                        std::pow(
                            value,
                            1.0f / 2.4f
                        )
                        - 0.055f;
                }
            }
        }
    }


    // ========================================================
    // 8. Conversión a 8 bits
    // ========================================================

    cv::Mat preview8;

    floating.convertTo(
        preview8,
        CV_8UC3,
        255.0
    );


    // ========================================================
    // 9. RGB -> BGR
    // ========================================================

    cv::Mat bgr;

    cv::cvtColor(
        preview8,
        bgr,
        cv::COLOR_RGB2BGR
    );

    return bgr;
}

