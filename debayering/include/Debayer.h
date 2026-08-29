/**
 * @file Debayer.h
 * @brief Declaración del algoritmo de reconstrucción Bayer a RGB.
 *
 * Este módulo implementa un algoritmo de demosaicing bilineal
 * desarrollado manualmente. Su objetivo es reconstruir los canales
 * R, G y B faltantes a partir de un mosaico Bayer de un solo canal.
 *
 * El algoritmo soporta los patrones Bayer clásicos:
 *
 *  - RGGB
 *  - BGGR
 *  - GRBG
 *  - GBRG
 *
 * La implementación evita utilizar cv::demosaicing(), ya que el
 * propósito de la práctica es implementar explícitamente la
 * interpolación.
 */

#ifndef DEBAYER_H
#define DEBAYER_H

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>

/**
 * @brief Colores presentes físicamente en el patrón CFA Bayer.
 */
enum class BayerColor
{
    RED,
    GREEN,
    BLUE
};

/**
 * @brief Patrones Bayer soportados.
 */
enum class BayerPattern
{
    RGGB,
    BGGR,
    GRBG,
    GBRG
};


/**
 * @class Debayer
 * @brief Implementa algoritmos de reconstrucción de mosaicos Bayer.
 *
 * La entrada debe ser una imagen RAW de un solo canal y 16 bits
 * (CV_16UC1).
 *
 * La salida es una imagen RGB de tres canales y 16 bits
 * (CV_16UC3).
 */
class Debayer
{
public:

    /**
     * @brief Realiza demosaicing mediante interpolación bilineal.
     *
     * Para cada píxel se conserva el canal originalmente medido
     * por el sensor y se estiman los otros dos canales utilizando
     * los píxeles vecinos correspondientes al mismo color.
     *
     * @param raw Imagen Bayer de entrada CV_16UC1.
     * @param pattern Patrón Bayer del sensor como texto.
     *
     * @return Imagen reconstruida CV_16UC3.
     *
     * La organización interna de los canales es:
     *
     * canal 0 -> R
     * canal 1 -> G
     * canal 2 -> B
     *
     * @throws std::invalid_argument si la imagen o patrón no son válidos.
     */
    static cv::Mat bilinear(
        const cv::Mat& raw,
        const std::string& pattern
    );

private:

    /**
     * @brief Convierte el patrón Bayer desde texto hacia enumeración.
     */
    static BayerPattern parsePattern(
        const std::string& pattern
    );

    /**
     * @brief Determina qué color existe físicamente en una coordenada.
     *
     * @param row Fila del píxel.
     * @param col Columna del píxel.
     * @param pattern Patrón Bayer utilizado.
     *
     * @return Color correspondiente a la posición solicitada.
     */
    static BayerColor colorAt(
        int row,
        int col,
        BayerPattern pattern
    );

    /**
     * @brief Interpola un canal faltante utilizando vecinos Bayer.
     *
     * Solo participan en el promedio aquellos píxeles vecinos que
     * contienen físicamente el color solicitado.
     *
     * @param raw Imagen RAW Bayer.
     * @param row Fila actual.
     * @param col Columna actual.
     * @param targetColor Canal que se desea reconstruir.
     * @param pattern Patrón Bayer.
     *
     * @return Intensidad interpolada de 16 bits.
     */
    static std::uint16_t interpolateChannel(
        const cv::Mat& raw,
        int row,
        int col,
        BayerColor targetColor,
        BayerPattern pattern
    );
};

#endif //DEBAYERING_DEBAYER_H