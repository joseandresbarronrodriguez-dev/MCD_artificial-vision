//
// Created by andru on 8/23/2026.
//
/**
 * @file Debayer.cpp
 * @brief Implementación del algoritmo bilineal de demosaicing.
 *
 * El algoritmo reconstruye una imagen RGB completa a partir de
 * un mosaico Bayer de un solo canal.
 *
 * La implementación se mantiene deliberadamente sencilla para
 * poder estudiar posteriormente su costo computacional y realizar
 * optimizaciones específicas para plataformas ARM.
 */

#include "Debayer.h"

#include <algorithm>
#include <stdexcept>


BayerPattern Debayer::parsePattern(
    const std::string& pattern)
{
    if (pattern == "RGGB")
        return BayerPattern::RGGB;

    if (pattern == "BGGR")
        return BayerPattern::BGGR;

    if (pattern == "GRBG")
        return BayerPattern::GRBG;

    if (pattern == "GBRG")
        return BayerPattern::GBRG;

    throw std::invalid_argument(
        "Patron Bayer no soportado: " + pattern
    );
}


BayerColor Debayer::colorAt(
    int row,
    int col,
    BayerPattern pattern)
{
    /*
     * El patrón Bayer se repite cada 2 filas y 2 columnas.
     *
     * Por ese motivo solamente necesitamos conocer la paridad
     * de las coordenadas para determinar el color del píxel.
     */

    const bool evenRow = (row % 2) == 0;
    const bool evenCol = (col % 2) == 0;

    switch (pattern)
    {
        // R G
        // G B
        case BayerPattern::RGGB:

            if (evenRow && evenCol)
                return BayerColor::RED;

            if (!evenRow && !evenCol)
                return BayerColor::BLUE;

            return BayerColor::GREEN;


        // B G
        // G R
        case BayerPattern::BGGR:

            if (evenRow && evenCol)
                return BayerColor::BLUE;

            if (!evenRow && !evenCol)
                return BayerColor::RED;

            return BayerColor::GREEN;


        // G R
        // B G
        case BayerPattern::GRBG:

            if (evenRow && !evenCol)
                return BayerColor::RED;

            if (!evenRow && evenCol)
                return BayerColor::BLUE;

            return BayerColor::GREEN;


        // G B
        // R G
        case BayerPattern::GBRG:

            if (evenRow && !evenCol)
                return BayerColor::BLUE;

            if (!evenRow && evenCol)
                return BayerColor::RED;

            return BayerColor::GREEN;
    }

    throw std::runtime_error(
        "Patron Bayer invalido."
    );
}


std::uint16_t Debayer::interpolateChannel(
    const cv::Mat& raw,
    int row,
    int col,
    BayerColor targetColor,
    BayerPattern pattern)
{
    /*
     * Si el píxel contiene físicamente el canal solicitado,
     * no es necesario interpolarlo.
     */
    if (colorAt(row, col, pattern) == targetColor)
    {
        return raw.at<std::uint16_t>(row, col);
    }

    /*
     * Para una interpolación bilineal Bayer basta analizar
     * el vecindario 3x3 alrededor del píxel.
     *
     * Dependiendo de la posición:
     *
     * - Verde puede tener 4 vecinos.
     * - Rojo o azul pueden encontrarse en diagonales.
     * - En posiciones verdes habrá normalmente 2 vecinos
     *   rojos o azules.
     *
     * Al filtrar los vecinos utilizando colorAt(), el mismo
     * algoritmo funciona para los cuatro patrones Bayer.
     */

    std::uint64_t sum = 0;
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int neighborRow = row + dy;
            const int neighborCol = col + dx;

            /*
             * Evitamos acceder fuera de la imagen.
             * Esto permite procesar también los píxeles
             * ubicados en los bordes.
             */
            if (
                neighborRow < 0 ||
                neighborRow >= raw.rows ||
                neighborCol < 0 ||
                neighborCol >= raw.cols
            )
            {
                continue;
            }

            /*
             * Únicamente utilizamos vecinos donde el sensor
             * realmente midió el color que queremos estimar.
             */
            if (
                colorAt(
                    neighborRow,
                    neighborCol,
                    pattern
                ) == targetColor
            )
            {
                sum += raw.at<std::uint16_t>(
                    neighborRow,
                    neighborCol
                );

                ++count;
            }
        }
    }

    /*
     * Esta condición normalmente no debería ocurrir en un
     * mosaico Bayer válido. Se mantiene como protección ante
     * entradas incorrectas.
     */
    if (count == 0)
    {
        return 0;
    }

    return static_cast<std::uint16_t>(
        sum / count
    );
}


cv::Mat Debayer::bilinear(
    const cv::Mat& raw,
    const std::string& patternString)
{
    /*
     * Validamos que la entrada corresponda al formato
     * esperado por nuestro algoritmo.
     */
    if (raw.empty())
    {
        throw std::invalid_argument(
            "La imagen RAW esta vacia."
        );
    }

    if (raw.type() != CV_16UC1)
    {
        throw std::invalid_argument(
            "Debayer::bilinear requiere una imagen CV_16UC1."
        );
    }

    const BayerPattern pattern =
        parsePattern(patternString);

    /*
     * Imagen reconstruida.
     *
     * CV_16UC3:
     * - unsigned short (16 bits)
     * - 3 canales
     *
     * Para esta práctica almacenaremos:
     *
     * [0] = R
     * [1] = G
     * [2] = B
     */
    cv::Mat rgb(
        raw.rows,
        raw.cols,
        CV_16UC3
    );

    /*
     * Recorremos todos los píxeles visibles del sensor.
     */
    for (int row = 0; row < raw.rows; ++row)
    {
        for (int col = 0; col < raw.cols; ++col)
        {
            const BayerColor originalColor =
                colorAt(
                    row,
                    col,
                    pattern
                );

            cv::Vec<std::uint16_t, 3> pixel;

            /*
             * Canal rojo.
             */
            if (originalColor == BayerColor::RED)
            {
                pixel[0] =
                    raw.at<std::uint16_t>(row, col);
            }
            else
            {
                pixel[0] =
                    interpolateChannel(
                        raw,
                        row,
                        col,
                        BayerColor::RED,
                        pattern
                    );
            }

            /*
             * Canal verde.
             */
            if (originalColor == BayerColor::GREEN)
            {
                pixel[1] =
                    raw.at<std::uint16_t>(row, col);
            }
            else
            {
                pixel[1] =
                    interpolateChannel(
                        raw,
                        row,
                        col,
                        BayerColor::GREEN,
                        pattern
                    );
            }

            /*
             * Canal azul.
             */
            if (originalColor == BayerColor::BLUE)
            {
                pixel[2] =
                    raw.at<std::uint16_t>(row, col);
            }
            else
            {
                pixel[2] =
                    interpolateChannel(
                        raw,
                        row,
                        col,
                        BayerColor::BLUE,
                        pattern
                    );
            }

            rgb.at<cv::Vec<std::uint16_t, 3>>(
                row,
                col
            ) = pixel;
        }
    }

    return rgb;
}