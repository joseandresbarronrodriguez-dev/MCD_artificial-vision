/**
 * @file BayerSelector.cpp
 * @brief Implementación robusta del selector CFA Bayer.
 */

#include "BayerSelector.h"

#include "Debayer.h"
#include "ImageUtils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>


namespace
{

/**
 * @brief Calcula el error cuadrático medio entre dos imágenes.
 */
double calculateMSE(
    const cv::Mat& first,
    const cv::Mat& second)
{
    cv::Mat difference;

    cv::absdiff(
        first,
        second,
        difference
    );

    difference.convertTo(
        difference,
        CV_32F
    );

    difference =
        difference.mul(
            difference
        );

    const cv::Scalar error =
        cv::mean(difference);

    return
        (
            error[0] +
            error[1] +
            error[2]
        ) /
        3.0;
}


/**
 * @brief Calcula PSNR a partir del MSE.
 */
double calculatePSNR(
    double mse)
{
    if (mse <= 1e-12)
    {
        return
            std::numeric_limits<double>::infinity();
    }

    return
        10.0 *
        std::log10(
            (255.0 * 255.0) /
            mse
        );
}


/**
 * @brief Evalúa diferencias estructurales mediante gradientes.
 *
 * Un patrón CFA incorrecto suele generar:
 *
 * - bordes falsos;
 * - zipper;
 * - estructuras alternadas;
 *
 * Por ello comparamos la magnitud de los gradientes.
 */
double calculateGradientError(
    const cv::Mat& first,
    const cv::Mat& second)
{
    cv::Mat grayFirst;
    cv::Mat graySecond;

    cv::cvtColor(
        first,
        grayFirst,
        cv::COLOR_BGR2GRAY
    );

    cv::cvtColor(
        second,
        graySecond,
        cv::COLOR_BGR2GRAY
    );


    cv::Mat firstX;
    cv::Mat firstY;

    cv::Mat secondX;
    cv::Mat secondY;


    cv::Sobel(
        grayFirst,
        firstX,
        CV_32F,
        1,
        0,
        3
    );

    cv::Sobel(
        grayFirst,
        firstY,
        CV_32F,
        0,
        1,
        3
    );


    cv::Sobel(
        graySecond,
        secondX,
        CV_32F,
        1,
        0,
        3
    );

    cv::Sobel(
        graySecond,
        secondY,
        CV_32F,
        0,
        1,
        3
    );


    cv::Mat magnitudeFirst;
    cv::Mat magnitudeSecond;

    cv::magnitude(
        firstX,
        firstY,
        magnitudeFirst
    );

    cv::magnitude(
        secondX,
        secondY,
        magnitudeSecond
    );


    cv::Mat difference;

    cv::absdiff(
        magnitudeFirst,
        magnitudeSecond,
        difference
    );

    return
        cv::mean(difference)[0];
}


/**
 * @brief Evalúa únicamente diferencias de crominancia.
 *
 * Convertimos BGR -> YCrCb y comparamos Cr y Cb.
 *
 * Esto ayuda a penalizar patrones CFA que producen
 * falsos colores.
 */
double calculateChromaError(
    const cv::Mat& first,
    const cv::Mat& second)
{
    cv::Mat ycrcbFirst;
    cv::Mat ycrcbSecond;

    cv::cvtColor(
        first,
        ycrcbFirst,
        cv::COLOR_BGR2YCrCb
    );

    cv::cvtColor(
        second,
        ycrcbSecond,
        cv::COLOR_BGR2YCrCb
    );


    std::vector<cv::Mat> firstChannels;
    std::vector<cv::Mat> secondChannels;

    cv::split(
        ycrcbFirst,
        firstChannels
    );

    cv::split(
        ycrcbSecond,
        secondChannels
    );


    /*
     * Canal 0 = luminancia Y
     * Canal 1 = Cr
     * Canal 2 = Cb
     *
     * Solo evaluamos crominancia.
     */
    cv::Mat crDifference;
    cv::Mat cbDifference;


    cv::absdiff(
        firstChannels[1],
        secondChannels[1],
        crDifference
    );

    cv::absdiff(
        firstChannels[2],
        secondChannels[2],
        cbDifference
    );


    crDifference.convertTo(
        crDifference,
        CV_32F
    );

    cbDifference.convertTo(
        cbDifference,
        CV_32F
    );


    crDifference =
        crDifference.mul(
            crDifference
        );

    cbDifference =
        cbDifference.mul(
            cbDifference
        );


    return
        (
            cv::mean(crDifference)[0] +
            cv::mean(cbDifference)[0]
        ) /
        2.0;
}


/**
 * @brief Normalización donde un valor grande es mejor.
 */
double normalizeHigh(
    double value,
    double minimum,
    double maximum)
{
    if (std::abs(maximum - minimum) < 1e-12)
    {
        return 1.0;
    }

    return
        (value - minimum) /
        (maximum - minimum);
}


/**
 * @brief Normalización donde un valor pequeño es mejor.
 */
double normalizeLow(
    double value,
    double minimum,
    double maximum)
{
    return
        1.0 -
        normalizeHigh(
            value,
            minimum,
            maximum
        );
}

} // namespace


BayerSelectionResult BayerSelector::select(
    const cv::Mat& raw,
    const cv::Mat& reference,
    const std::string& metadataPattern,
    unsigned int blackLevel,
    unsigned int maximum,
    float wbR,
    float wbG,
    float wbB,
    const cv::Matx33f& colorMatrix)
{
    if (raw.empty() || reference.empty())
    {
        throw std::invalid_argument(
            "BayerSelector recibio una imagen vacia."
        );
    }


    // ========================================================
    // 1. Alinear referencia con las dimensiones RAW
    // ========================================================

    cv::Mat referenceAligned;

    if (
        reference.cols != raw.cols ||
        reference.rows != raw.rows
    )
    {
        cv::resize(
            reference,
            referenceAligned,
            cv::Size(
                raw.cols,
                raw.rows
            ),
            0.0,
            0.0,
            cv::INTER_AREA
        );
    }
    else
    {
        referenceAligned =
            reference;
    }


    // ========================================================
    // 2. Tamaño de las regiones de análisis
    // ========================================================

    int roiSize =
        std::min(
            1000,
            std::min(
                raw.cols,
                raw.rows
            )
        );

    /*
     * La región debe tener tamaño par para conservar
     * correctamente la periodicidad Bayer 2x2.
     */
    roiSize -=
        roiSize % 2;


    // ========================================================
    // 3. Posiciones de análisis
    // ========================================================

    /*
     * Coordenadas normalizadas aproximadas:
     *
     * (0.25, 0.25)
     * (0.75, 0.25)
     * (0.50, 0.50)
     * (0.25, 0.75)
     * (0.75, 0.75)
     */
    const std::vector<cv::Point2d> positions =
    {
        {0.25, 0.25},
        {0.75, 0.25},
        {0.50, 0.50},
        {0.25, 0.75},
        {0.75, 0.75}
    };


    std::vector<cv::Rect> regions;


    for (const cv::Point2d& position :
         positions)
    {
        int centerX =
            static_cast<int>(
                raw.cols *
                position.x
            );

        int centerY =
            static_cast<int>(
                raw.rows *
                position.y
            );


        int x =
            centerX -
            roiSize / 2;

        int y =
            centerY -
            roiSize / 2;


        x =
            std::clamp(
                x,
                0,
                raw.cols - roiSize
            );

        y =
            std::clamp(
                y,
                0,
                raw.rows - roiSize
            );


        /*
         * MUY IMPORTANTE:
         *
         * El CFA tiene periodicidad 2x2.
         *
         * La región siempre debe comenzar en coordenadas pares
         * para conservar la fase Bayer original.
         */
        x -= x % 2;
        y -= y % 2;


        regions.emplace_back(
            x,
            y,
            roiSize,
            roiSize
        );
    }


    // ========================================================
    // 4. Patrones candidatos
    // ========================================================

    const std::vector<std::string> patterns =
    {
        "RGGB",
        "BGGR",
        "GRBG",
        "GBRG"
    };


    BayerSelectionResult result;


    // ========================================================
    // 5. Evaluar cada candidato
    // ========================================================

    for (const std::string& pattern :
         patterns)
    {
        double accumulatedMSE = 0.0;
        double accumulatedGradient = 0.0;
        double accumulatedChroma = 0.0;


        for (const cv::Rect& roi :
             regions)
        {
            // ----------------------------------------------
            // Extraer mosaico de prueba
            // ----------------------------------------------

            cv::Mat rawROI =
                raw(roi).clone();


            cv::Mat referenceROI =
                referenceAligned(
                    roi
                ).clone();


            // ----------------------------------------------
            // Nuestro debayering
            // ----------------------------------------------

            cv::Mat candidateRGB =
                Debayer::bilinear(
                    rawROI,
                    pattern
                );


            // ----------------------------------------------
            // Pipeline visual
            // ----------------------------------------------

            cv::Mat candidatePreview =
                ImageUtils::createPreview8(
                    candidateRGB,
                    blackLevel,
                    maximum,
                    wbR,
                    wbG,
                    wbB,
                    colorMatrix
                );


            // ----------------------------------------------
            // Métricas
            // ----------------------------------------------

            accumulatedMSE +=
                calculateMSE(
                    candidatePreview,
                    referenceROI
                );


            accumulatedGradient +=
                calculateGradientError(
                    candidatePreview,
                    referenceROI
                );


            accumulatedChroma +=
                calculateChromaError(
                    candidatePreview,
                    referenceROI
                );
        }


        const double regionCount =
            static_cast<double>(
                regions.size()
            );


        BayerScore score;

        score.pattern =
            pattern;

        score.mse =
            accumulatedMSE /
            regionCount;

        score.psnr =
            calculatePSNR(
                score.mse
            );

        score.gradientError =
            accumulatedGradient /
            regionCount;

        score.chromaError =
            accumulatedChroma /
            regionCount;


        result.scores.push_back(
            score
        );
    }


    // ========================================================
    // 6. Buscar rangos de normalización
    // ========================================================

    double minPSNR =
        std::numeric_limits<double>::max();

    double maxPSNR =
        std::numeric_limits<double>::lowest();

    double minGradient =
        std::numeric_limits<double>::max();

    double maxGradient =
        std::numeric_limits<double>::lowest();

    double minChroma =
        std::numeric_limits<double>::max();

    double maxChroma =
        std::numeric_limits<double>::lowest();


    for (const BayerScore& score :
         result.scores)
    {
        minPSNR =
            std::min(
                minPSNR,
                score.psnr
            );

        maxPSNR =
            std::max(
                maxPSNR,
                score.psnr
            );


        minGradient =
            std::min(
                minGradient,
                score.gradientError
            );

        maxGradient =
            std::max(
                maxGradient,
                score.gradientError
            );


        minChroma =
            std::min(
                minChroma,
                score.chromaError
            );

        maxChroma =
            std::max(
                maxChroma,
                score.chromaError
            );
    }


    // ========================================================
    // 7. Calcular score combinado
    // ========================================================

    /*
     * Pesos:
     *
     * 40% similitud RGB
     * 40% estructura / bordes
     * 20% crominancia
     *
     * La estructura tiene mucho peso porque los patrones Bayer
     * incorrectos suelen producir zipper y artefactos en bordes.
     */
    for (BayerScore& score :
         result.scores)
    {
        const double psnrScore =
            normalizeHigh(
                score.psnr,
                minPSNR,
                maxPSNR
            );


        const double gradientScore =
            normalizeLow(
                score.gradientError,
                minGradient,
                maxGradient
            );


        const double chromaScore =
            normalizeLow(
                score.chromaError,
                minChroma,
                maxChroma
            );


        score.finalScore =
            0.40 * psnrScore +
            0.40 * gradientScore +
            0.20 * chromaScore;
    }


    // ========================================================
    // 8. Ordenar candidatos
    // ========================================================

    std::sort(
        result.scores.begin(),
        result.scores.end(),
        [](const BayerScore& a,
           const BayerScore& b)
        {
            return
                a.finalScore >
                b.finalScore;
        }
    );


    result.empiricalWinner =
        result.scores.front().pattern;


    // ========================================================
    // 9. Calcular confianza
    // ========================================================

    if (result.scores.size() >= 2)
    {
        result.confidenceMargin =
            result.scores[0].finalScore -
            result.scores[1].finalScore;
    }


    // ========================================================
    // 10. Selección final
    // ========================================================

    /*
     * Si el selector no tiene una ventaja clara entre los dos
     * primeros candidatos, usamos el CFA de metadata como
     * mecanismo de respaldo.
     *
     * Esto evita cambiar el CFA basándonos en diferencias muy
     * pequeñas o ruido experimental.
     */
    constexpr double confidenceThreshold =
        0.08;


    const bool metadataValid =
        metadataPattern == "RGGB" ||
        metadataPattern == "BGGR" ||
        metadataPattern == "GRBG" ||
        metadataPattern == "GBRG";


    if (
        result.confidenceMargin <
            confidenceThreshold &&
        metadataValid
    )
    {
        result.selectedPattern =
            metadataPattern;

        result.usedMetadataFallback =
            true;
    }
    else
    {
        result.selectedPattern =
            result.empiricalWinner;
    }


    return result;
}