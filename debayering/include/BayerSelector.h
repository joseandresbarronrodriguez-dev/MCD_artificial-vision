/**
 * @file BayerSelector.h
 * @brief Selección robusta del patrón CFA Bayer.
 *
 * Se evalúan los cuatro patrones Bayer clásicos utilizando
 * múltiples regiones de la imagen y varias métricas:
 *
 * - Error cuadrático medio RGB.
 * - PSNR.
 * - Error estructural basado en gradientes.
 * - Error de crominancia.
 *
 * El objetivo es reducir la posibilidad de seleccionar un CFA
 * incorrecto debido a una única región poco representativa.
 */

#ifndef BAYER_SELECTOR_H
#define BAYER_SELECTOR_H

#include <opencv2/core.hpp>

#include <string>
#include <vector>


/**
 * @struct BayerScore
 * @brief Métricas obtenidas para un patrón Bayer candidato.
 */
struct BayerScore
{
    std::string pattern;

    double mse = 0.0;
    double psnr = 0.0;

    double gradientError = 0.0;
    double chromaError = 0.0;

    /**
     * Score normalizado final.
     *
     * Mayor valor = mejor candidato.
     */
    double finalScore = 0.0;
};


/**
 * @struct BayerSelectionResult
 * @brief Resultado completo del selector CFA.
 */
struct BayerSelectionResult
{
    /**
     * Patrón seleccionado finalmente.
     */
    std::string selectedPattern;

    /**
     * Mejor patrón puramente experimental.
     */
    std::string empiricalWinner;

    /**
     * Diferencia entre el mejor y segundo mejor score.
     *
     * Una diferencia pequeña indica baja confianza.
     */
    double confidenceMargin = 0.0;

    /**
     * Indica si se utilizó la metadata como criterio
     * de respaldo debido a baja confianza.
     */
    bool usedMetadataFallback = false;

    std::vector<BayerScore> scores;
};


class BayerSelector
{
public:

    /**
     * @brief Evalúa automáticamente los cuatro patrones Bayer.
     *
     * Se analizan múltiples regiones distribuidas sobre la imagen.
     *
     * @param raw Mosaico RAW CV_16UC1.
     * @param reference Imagen BGR de referencia.
     * @param metadataPattern CFA informado por LibRaw.
     * @param blackLevel Nivel negro.
     * @param maximum Nivel máximo del sensor.
     * @param wbR Balance de blancos rojo.
     * @param wbG Balance de blancos verde.
     * @param wbB Balance de blancos azul.
     * @param colorMatrix Matriz Camera RGB -> sRGB.
     *
     * @return Resultado del proceso de selección.
     */
    static BayerSelectionResult select(
        const cv::Mat& raw,
        const cv::Mat& reference,
        const std::string& metadataPattern,
        unsigned int blackLevel,
        unsigned int maximum,
        float wbR,
        float wbG,
        float wbB,
        const cv::Matx33f& colorMatrix
    );
};

#endif