/**
 * ============================================================
 * TAREA 3 - HOUGH + HARRIS ARM BENCHMARK
 * ============================================================
 *
 * Objetivo:
 *
 *   Implementar y comparar:
 *
 *      1. Acumulador de Hough propio usando Eigen.
 *      2. cv::HoughLines().
 *
 *      3. Score de Harris propio usando Eigen.
 *      4. cv::cornerHarris().
 *      5. cv::cornerMinEigenVal().
 *
 * Además:
 *
 *      - validación numérica
 *      - visualización
 *      - Non-Maximum Suppression para Harris
 *      - benchmark reproducible
 *
 * ============================================================
 */

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Implementaciones propias
#include "Benchmark.h"
#include "HarrisScore.h"
#include "HoughAccumulator.h"


namespace fs = std::filesystem;


// ============================================================
// DIBUJAR LÍNEA DE HOUGH
// ============================================================

void drawHoughLine(
        cv::Mat& image,
        float rho,
        float theta,
        const cv::Scalar& color
)
{
    const double a =
            std::cos(theta);

    const double b =
            std::sin(theta);


    const double x0 =
            a * rho;

    const double y0 =
            b * rho;


    cv::Point point1;
    cv::Point point2;


    point1.x =
            cvRound(
                    x0 +
                    1000.0 * (-b)
            );

    point1.y =
            cvRound(
                    y0 +
                    1000.0 * a
            );


    point2.x =
            cvRound(
                    x0 -
                    1000.0 * (-b)
            );

    point2.y =
            cvRound(
                    y0 -
                    1000.0 * a
            );


    cv::line(
            image,
            point1,
            point2,
            color,
            2,
            cv::LINE_AA
    );
}


// ============================================================
// IMPRIMIR RESULTADOS DE BENCHMARK
// ============================================================

void printBenchmarkStats(
        const BenchmarkStats& stats
)
{
    std::cout
            << std::left
            << std::setw(28)
            << stats.name

            << " Mean: "
            << std::setw(10)
            << std::fixed
            << std::setprecision(4)
            << stats.meanMs
            << " ms"

            << " | SD: "
            << std::setw(10)
            << stats.stdDevMs

            << " | Median: "
            << std::setw(10)
            << stats.medianMs

            << " | P95: "
            << std::setw(10)
            << stats.p95Ms

            << " | Min: "
            << std::setw(10)
            << stats.minMs

            << " | Max: "
            << stats.maxMs

            << "\n";
}


// ============================================================
// VISUALIZACIÓN HARRIS + NMS
// ============================================================

/**
 * Dibuja únicamente máximos locales fuertes del mapa Harris.
 *
 * Etapas:
 *
 *      Harris response
 *             ↓
 *       threshold
 *             ↓
 *          NMS
 *             ↓
 *         esquinas
 */
cv::Mat drawHarrisCorners(
        const cv::Mat& original,
        const cv::Mat& response,
        double thresholdRatio,
        int nmsKernel,
        const cv::Scalar& color
)
{
    if (original.empty())
    {
        throw std::invalid_argument(
                "La imagen original esta vacia."
        );
    }


    if (response.empty() ||
        response.type() != CV_32FC1)
    {
        throw std::invalid_argument(
                "La respuesta Harris debe ser CV_32FC1."
        );
    }


    if (nmsKernel < 3 ||
        nmsKernel % 2 == 0)
    {
        throw std::invalid_argument(
                "El kernel NMS debe ser impar y >= 3."
        );
    }


    cv::Mat output =
            original.clone();


    // --------------------------------------------------------
    // Obtener valor máximo Harris
    // --------------------------------------------------------

    double minResponse;
    double maxResponse;


    cv::minMaxLoc(
            response,
            &minResponse,
            &maxResponse
    );


    const float threshold =
            static_cast<float>(
                    maxResponse *
                    thresholdRatio
            );


    // --------------------------------------------------------
    // Non-Maximum Suppression
    // --------------------------------------------------------
    //
    // Dilatar nos permite obtener el máximo de cada
    // vecindario.
    // --------------------------------------------------------

    cv::Mat localMaximum;


    cv::Mat nmsStructuringElement =
            cv::getStructuringElement(
                    cv::MORPH_RECT,
                    cv::Size(
                            nmsKernel,
                            nmsKernel
                    )
            );


    cv::dilate(
            response,
            localMaximum,
            nmsStructuringElement
    );


    // --------------------------------------------------------
    // Dibujar máximos locales
    // --------------------------------------------------------

    int cornerCount =
            0;


    for (int y = 0;
         y < response.rows;
         ++y)
    {
        for (int x = 0;
             x < response.cols;
             ++x)
        {
            const float value =
                    response.at<float>(
                            y,
                            x
                    );


            const float neighborhoodMaximum =
                    localMaximum.at<float>(
                            y,
                            x
                    );


            if (value > threshold &&
                value >= neighborhoodMaximum)
            {
                cv::circle(
                        output,
                        cv::Point(
                                x,
                                y
                        ),

                        // radio
                        4,

                        color,

                        // grosor
                        2,

                        cv::LINE_AA
                );


                cornerCount++;
            }
        }
    }


    std::cout
            << "Esquinas detectadas:  "
            << cornerCount
            << "\n";


    return output;
}


// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[])
{
    // Crear directorio de resultados si no existe.
    fs::create_directories(
            "results"
    );


    std::cout
            << "\n========================================\n"
            << " HOUGH + HARRIS ARM BENCHMARK\n"
            << "========================================\n";


    // ========================================================
    // 1. IMAGEN DE ENTRADA
    // ========================================================

    std::string imagePath =
            "data/input/test.jpg";


    if (argc > 1)
    {
        imagePath =
                argv[1];
    }


    std::cout
            << "\nImagen de entrada: "
            << imagePath
            << "\n";


    cv::Mat image =
            cv::imread(
                    imagePath,
                    cv::IMREAD_COLOR
            );


    if (image.empty())
    {
        std::cerr
                << "\nERROR: No se pudo cargar la imagen.\n"
                << "Ruta: "
                << imagePath
                << "\n";

        return 1;
    }


    std::cout
            << "Resolucion: "
            << image.cols
            << " x "
            << image.rows
            << "\n";


    // ========================================================
    // 2. ESCALA DE GRISES
    // ========================================================

    cv::Mat gray;


    cv::cvtColor(
            image,
            gray,
            cv::COLOR_BGR2GRAY
    );


    // ========================================================
    // 3. CANNY
    // ========================================================

    constexpr double CANNY_LOW =
            50.0;

    constexpr double CANNY_HIGH =
            150.0;


    cv::Mat edges;


    cv::Canny(
            gray,
            edges,
            CANNY_LOW,
            CANNY_HIGH
    );


    const int edgePixels =
            cv::countNonZero(
                    edges
            );


    std::cout
            << "Pixeles de borde: "
            << edgePixels
            << "\n";


    cv::imwrite(
            "results/edges.png",
            edges
    );


    // ========================================================
    // 4. HOUGH PROPIO
    // ========================================================

    constexpr int THETA_BINS =
            180;


    HoughAccumulator hough(
            THETA_BINS
    );


    HoughResult houghResult =
            hough.compute(
                    edges
            );


    Eigen::Index maxRhoIndex;
    Eigen::Index maxThetaIndex;


    const int maxVotes =
            houghResult
                    .accumulator
                    .maxCoeff(
                            &maxRhoIndex,
                            &maxThetaIndex
                    );


    const int detectedRho =
            static_cast<int>(
                    maxRhoIndex
            )
            -
            houghResult.rhoMax;


    const float detectedThetaDegrees =
            180.0f
            *
            static_cast<float>(
                    maxThetaIndex
            )
            /
            static_cast<float>(
                    houghResult.thetaBins
            );


    const float detectedThetaRadians =
            detectedThetaDegrees
            *
            static_cast<float>(CV_PI)
            /
            180.0f;


    std::cout
            << "\n========================================\n"
            << " ACUMULADOR DE HOUGH\n"
            << "========================================\n";


    std::cout
            << "Rho maximo:          "
            << houghResult.rhoMax
            << "\n";


    std::cout
            << "Bins de rho:         "
            << houghResult.accumulator.rows()
            << "\n";


    std::cout
            << "Bins de theta:       "
            << houghResult.accumulator.cols()
            << "\n";


    std::cout
            << "Voto maximo:         "
            << maxVotes
            << "\n";


    std::cout
            << "Rho detectado:       "
            << detectedRho
            << "\n";


    std::cout
            << "Theta detectado:     "
            << detectedThetaDegrees
            << " grados\n";


    // ========================================================
    // 5. HOUGH OPENCV
    // ========================================================

    constexpr int HOUGH_THRESHOLD =
            100;


    std::vector<cv::Vec2f> opencvLines;


    cv::HoughLines(
            edges,
            opencvLines,
            1.0,
            CV_PI / 180.0,
            HOUGH_THRESHOLD
    );


    std::cout
            << "\n========================================\n"
            << " COMPARACION HOUGH CON OPENCV\n"
            << "========================================\n";


    std::cout
            << "Lineas OpenCV:       "
            << opencvLines.size()
            << "\n";


    if (!opencvLines.empty())
    {
        const float opencvThetaDegrees =
                opencvLines[0][1]
                *
                180.0f
                /
                static_cast<float>(CV_PI);


        std::cout
                << "Rho OpenCV:          "
                << opencvLines[0][0]
                << "\n";


        std::cout
                << "Theta OpenCV:        "
                << opencvThetaDegrees
                << " grados\n";
    }


    // ========================================================
    // 6. VISUALIZACIÓN HOUGH
    // ========================================================

    cv::Mat customHoughImage =
            image.clone();


    drawHoughLine(
            customHoughImage,
            static_cast<float>(
                    detectedRho
            ),
            detectedThetaRadians,
            cv::Scalar(
                    0,
                    0,
                    255
            )
    );


    cv::imwrite(
            "results/hough_custom.jpg",
            customHoughImage
    );


    if (!opencvLines.empty())
    {
        cv::Mat opencvHoughImage =
                image.clone();


        drawHoughLine(
                opencvHoughImage,
                opencvLines[0][0],
                opencvLines[0][1],
                cv::Scalar(
                        0,
                        255,
                        0
                )
        );


        cv::imwrite(
                "results/hough_opencv.jpg",
                opencvHoughImage
        );
    }


    // ========================================================
    // CONFIGURACION INTERACTIVA DE HARRIS
    // ========================================================

    int harrisBlockSize = 3;
    int harrisSobelKernel = 3;

    double harrisThresholdRatio = 0.05;
    int harrisNmsKernel = 5;

    constexpr float HARRIS_K = 0.04f;


    std::cout
                    << "\n========================================\n"
                    << " CONFIGURACION HARRIS\n"
                    << "========================================\n";


    // ========================================================
    // 1. BLOCK SIZE
    // ========================================================

    std::cout
                    << "\nBLOCK SIZE\n"
                    << "Controla el tamaño de la region utilizada para\n"
                    << "analizar los cambios de intensidad.\n\n"

                    << "Valores recomendados:\n"
                    << "  3 -> mejor localizacion y menor costo\n"
                    << "  5 -> balance entre precision y robustez\n"
                    << "  7 -> mayor robustez\n"
                    << "  9 -> regiones grandes, mayor costo\n\n"

                    << "Seleccione blockSize (3, 5, 7, 9): ";


    std::cin
                    >> harrisBlockSize;


    if (harrisBlockSize != 3 &&
        harrisBlockSize != 5 &&
        harrisBlockSize != 7 &&
        harrisBlockSize != 9) {
            std::cout
                            << "Valor invalido. Se utilizara 3.\n";

            harrisBlockSize = 3;
    }


    // ========================================================
    // 2. KERNEL SOBEL
    // ========================================================

    std::cout
                    << "\nSOBEL KERNEL\n"
                    << "Controla el tamano del filtro utilizado para\n"
                    << "calcular los gradientes Ix e Iy.\n\n"

                    << "Valores recomendados:\n"
                    << "  3 -> rapido y buena localizacion\n"
                    << "  5 -> gradiente mas suave\n"
                    << "  7 -> mayor suavizado y mayor costo\n\n"

                    << "Seleccione kernel Sobel (3, 5, 7): ";


    std::cin
                    >> harrisSobelKernel;


    if (harrisSobelKernel != 3 &&
        harrisSobelKernel != 5 &&
        harrisSobelKernel != 7) {
            std::cout
                            << "Valor invalido. Se utilizara 3.\n";

            harrisSobelKernel = 3;
    }


    // ========================================================
    // 3. THRESHOLD HARRIS
    // ========================================================
    //
    // El threshold se expresa como fraccion del score maximo.
    //
    // Ejemplo:
    //
    //      0.05 = 5 % del maximo
    //      0.10 = 10 % del maximo
    //
    // Un threshold mayor produce MENOS esquinas.
    // ========================================================

    std::cout
                    << "\nTHRESHOLD HARRIS\n"
                    << "Controla que tan fuerte debe ser una esquina.\n\n"

                    << "Valores sugeridos:\n"
                    << "  0.03 -> alta sensibilidad, muchas esquinas\n"
                    << "  0.05 -> sensibilidad media\n"
                    << "  0.10 -> solo esquinas fuertes\n"
                    << "  0.15 -> muy selectivo\n"
                    << "  0.20 -> solo respuestas muy fuertes\n\n"

                    << "Rango permitido: 0.01 - 0.30\n"
                    << "Seleccione threshold: ";


    std::cin
                    >> harrisThresholdRatio;


    if (harrisThresholdRatio < 0.01 ||
        harrisThresholdRatio > 0.30) {
            std::cout
                            << "Valor fuera de rango. Se utilizara 0.05.\n";

            harrisThresholdRatio = 0.05;
    }


    // ========================================================
    // 4. NON-MAXIMUM SUPPRESSION
    // ========================================================
    //
    // NMS elimina respuestas cercanas y conserva solamente
    // los maximos locales.
    //
    // Un kernel NMS mas grande produce menos esquinas cercanas.
    // ========================================================

    std::cout
                    << "\nNON-MAXIMUM SUPPRESSION (NMS)\n"
                    << "Controla la separacion entre esquinas detectadas.\n\n"

                    << "Valores recomendados:\n"
                    << "  3 -> permite esquinas muy cercanas\n"
                    << "  5 -> balanceado\n"
                    << "  7 -> mayor separacion\n"
                    << "  9 -> muy selectivo\n\n"

                    << "Seleccione kernel NMS (3, 5, 7, 9): ";


    std::cin
                    >> harrisNmsKernel;


    if (harrisNmsKernel != 3 &&
        harrisNmsKernel != 5 &&
        harrisNmsKernel != 7 &&
        harrisNmsKernel != 9) {
            std::cout
                            << "Valor invalido. Se utilizara 5.\n";

            harrisNmsKernel = 5;
    }


// ========================================================
// MOSTRAR CONFIGURACION FINAL
// ========================================================

std::cout
        << "\n========================================\n"
        << " CONFIGURACION SELECCIONADA\n"
        << "========================================\n"

        << "Block size:            "
        << harrisBlockSize
        << "\n"

        << "Sobel kernel:          "
        << harrisSobelKernel
        << "\n"

        << "Harris k:              "
        << HARRIS_K
        << "\n"

        << "Threshold ratio:       "
        << harrisThresholdRatio
        << "\n"

        << "NMS kernel:            "
        << harrisNmsKernel
        << "\n";


    // ========================================================
    // 8. HARRIS PROPIO
    // ========================================================

    HarrisScore harris(
            harrisBlockSize,
            harrisSobelKernel,
            HARRIS_K
    );


    cv::Mat harrisResponse =
            harris.compute(
                    gray
            );


    double harrisMin;
    double harrisMax;


    cv::Point harrisMinLocation;
    cv::Point harrisMaxLocation;


    cv::minMaxLoc(
            harrisResponse,
            &harrisMin,
            &harrisMax,
            &harrisMinLocation,
            &harrisMaxLocation
    );


    std::cout
            << "\n========================================\n"
            << " HARRIS SCORE PROPIO\n"
            << "========================================\n";


    std::cout
            << "Tamaño respuesta:    "
            << harrisResponse.cols
            << " x "
            << harrisResponse.rows
            << "\n";


    std::cout
            << "Tipo OpenCV:         "
            << harrisResponse.type()
            << "\n";


    std::cout
            << "Score minimo:        "
            << harrisMin
            << "\n";


    std::cout
            << "Score maximo:        "
            << harrisMax
            << "\n";


    std::cout
            << "Maximo localizado:   ("
            << harrisMaxLocation.x
            << ", "
            << harrisMaxLocation.y
            << ")\n";


    // ========================================================
    // 9. MAPA HARRIS PROPIO
    // ========================================================

    cv::Mat harrisNormalized;


    cv::normalize(
            harrisResponse,
            harrisNormalized,
            0,
            255,
            cv::NORM_MINMAX,
            CV_8UC1
    );


    cv::imwrite(
            "results/harris_custom.jpg",
            harrisNormalized
    );


    // ========================================================
    // 10. HARRIS OPENCV
    // ========================================================

    cv::Mat gray32f;


    gray.convertTo(
            gray32f,
            CV_32F,
            1.0 / 255.0
    );


    cv::Mat opencvHarrisResponse;


    cv::cornerHarris(
            gray32f,
            opencvHarrisResponse,
            harrisBlockSize,
            harrisSobelKernel,
            HARRIS_K
    );


    double opencvHarrisMin;
    double opencvHarrisMax;


    cv::Point opencvHarrisMinLocation;
    cv::Point opencvHarrisMaxLocation;


    cv::minMaxLoc(
            opencvHarrisResponse,
            &opencvHarrisMin,
            &opencvHarrisMax,
            &opencvHarrisMinLocation,
            &opencvHarrisMaxLocation
    );


    // ========================================================
    // 11. COMPARACIÓN HARRIS
    // ========================================================

    cv::Mat customHarrisNorm;
    cv::Mat opencvHarrisNorm;


    cv::normalize(
            harrisResponse,
            customHarrisNorm,
            0.0,
            1.0,
            cv::NORM_MINMAX,
            CV_32F
    );


    cv::normalize(
            opencvHarrisResponse,
            opencvHarrisNorm,
            0.0,
            1.0,
            cv::NORM_MINMAX,
            CV_32F
    );


    const double harrisRmse =
            cv::norm(
                    customHarrisNorm,
                    opencvHarrisNorm,
                    cv::NORM_L2
            )
            /
            std::sqrt(
                    static_cast<double>(
                            gray.total()
                    )
            );


    std::cout
            << "\n========================================\n"
            << " COMPARACION HARRIS CON OPENCV\n"
            << "========================================\n";


    std::cout
            << "Score minimo OpenCV: "
            << opencvHarrisMin
            << "\n";


    std::cout
            << "Score maximo OpenCV: "
            << opencvHarrisMax
            << "\n";


    std::cout
            << "Maximo OpenCV:       ("
            << opencvHarrisMaxLocation.x
            << ", "
            << opencvHarrisMaxLocation.y
            << ")\n";


    std::cout
            << "RMSE normalizado:    "
            << harrisRmse
            << "\n";


    // ========================================================
    // 12. MAPA HARRIS OPENCV
    // ========================================================

    cv::Mat opencvHarrisVisualization;


    cv::normalize(
            opencvHarrisResponse,
            opencvHarrisVisualization,
            0,
            255,
            cv::NORM_MINMAX,
            CV_8UC1
    );


    cv::imwrite(
            "results/harris_opencv.jpg",
            opencvHarrisVisualization
    );


    // ========================================================
    // 13. VISUALIZACIÓN DE ESQUINAS CON NMS
    // ========================================================

    std::cout
            << "\n========================================\n"
            << " ESQUINAS HARRIS + NMS\n"
            << "========================================\n";


    std::cout
            << "\nHarris propio:\n";


    cv::Mat customHarrisCorners =
                    drawHarrisCorners(
                            image,
                            harrisResponse,
                            harrisThresholdRatio,
                            harrisNmsKernel,
                            cv::Scalar(0, 0, 255)
                    );


    cv::imwrite(
            "results/harris_custom_corners.jpg",
            customHarrisCorners
    );


    std::cout
            << "\nHarris OpenCV:\n";


    cv::Mat opencvHarrisCorners =
                    drawHarrisCorners(
                            image,
                            opencvHarrisResponse,
                            harrisThresholdRatio,
                            harrisNmsKernel,
                            cv::Scalar(0, 255, 0)
                    );


    cv::imwrite(
            "results/harris_opencv_corners.jpg",
            opencvHarrisCorners
    );


    // ========================================================
    // 14. CONFIGURACIÓN DEL BENCHMARK
    // ========================================================

    constexpr int BENCHMARK_WARMUP =
            5;

    constexpr int BENCHMARK_ITERATIONS =
            100;


    // ========================================================
    // 15. BENCHMARK HOUGH
    // ========================================================

    std::cout
            << "\n========================================\n"
            << " BENCHMARK HOUGH\n"
            << "========================================\n";


    std::cout
            << "Warm-up:               "
            << BENCHMARK_WARMUP
            << "\n";


    std::cout
            << "Iteraciones medidas:   "
            << BENCHMARK_ITERATIONS
            << "\n";


    std::cout
            << "Resolucion:            "
            << edges.cols
            << " x "
            << edges.rows
            << "\n";


    std::cout
            << "Pixeles de borde:      "
            << edgePixels
            << "\n";


    std::cout
            << "Bins theta:            "
            << THETA_BINS
            << "\n\n";


    // --------------------------------------------------------
    // Hough propio
    // --------------------------------------------------------

    HoughResult benchmarkHoughResult;


    BenchmarkStats customHoughBenchmark =
            benchmark(
                    "Hough Eigen",
                    BENCHMARK_WARMUP,
                    BENCHMARK_ITERATIONS,

                    [&]()
                    {
                        benchmarkHoughResult =
                                hough.compute(
                                        edges
                                );
                    }
            );


    const int benchmarkMaxVote =
            benchmarkHoughResult
                    .accumulator
                    .maxCoeff();


    std::cout
            << "Max vote validation:  "
            << benchmarkMaxVote
            << "\n";


    // --------------------------------------------------------
    // OpenCV HoughLines
    // --------------------------------------------------------

    std::vector<cv::Vec2f> benchmarkOpenCVLines;


    BenchmarkStats opencvHoughBenchmark =
            benchmark(
                    "OpenCV HoughLines",
                    BENCHMARK_WARMUP,
                    BENCHMARK_ITERATIONS,

                    [&]()
                    {
                        benchmarkOpenCVLines.clear();


                        cv::HoughLines(
                                edges,
                                benchmarkOpenCVLines,
                                1.0,
                                CV_PI / 180.0,
                                HOUGH_THRESHOLD
                        );
                    }
            );


    std::cout
            << "\n========================================\n"
            << " RESULTADOS HOUGH\n"
            << "========================================\n\n";


    printBenchmarkStats(
            customHoughBenchmark
    );


    printBenchmarkStats(
            opencvHoughBenchmark
    );


    // ========================================================
    // 16. BENCHMARK HARRIS
    // ========================================================

    std::cout
            << "\n========================================\n"
            << " BENCHMARK HARRIS\n"
            << "========================================\n";


    std::cout
            << "Warm-up:               "
            << BENCHMARK_WARMUP
            << "\n";


    std::cout
            << "Iteraciones medidas:   "
            << BENCHMARK_ITERATIONS
            << "\n";


    std::cout
            << "Resolucion:            "
            << gray32f.cols
            << " x "
            << gray32f.rows
            << "\n";


    std::cout
            << "Block size:            "
            << harrisBlockSize
            << "\n";


    std::cout
            << "Sobel kernel:          "
            << harrisSobelKernel
            << "\n";


    std::cout
            << "Harris k:              "
            << HARRIS_K
            << "\n\n";


    // --------------------------------------------------------
    // Harris propio
    // --------------------------------------------------------

    cv::Mat benchmarkCustomHarris;


    BenchmarkStats customHarrisBenchmark =
            benchmark(
                    "Harris Eigen",
                    BENCHMARK_WARMUP,
                    BENCHMARK_ITERATIONS,

                    [&]()
                    {
                        benchmarkCustomHarris =
                                harris.compute(
                                        gray32f
                                );
                    }
            );


    // --------------------------------------------------------
    // OpenCV cornerHarris
    // --------------------------------------------------------

    cv::Mat benchmarkOpenCVHarris;


    BenchmarkStats opencvHarrisBenchmark =
            benchmark(
                    "OpenCV cornerHarris",
                    BENCHMARK_WARMUP,
                    BENCHMARK_ITERATIONS,

                    [&]()
                    {
                        cv::cornerHarris(
                                gray32f,
                                benchmarkOpenCVHarris,
                                harrisBlockSize,
                                harrisSobelKernel,
                                HARRIS_K
                        );
                    }
            );


    // --------------------------------------------------------
    // OpenCV minimum eigenvalue
    // --------------------------------------------------------

    cv::Mat benchmarkMinEigen;


    BenchmarkStats minEigenBenchmark =
            benchmark(
                    "OpenCV minEigenVal",
                    BENCHMARK_WARMUP,
                    BENCHMARK_ITERATIONS,

                    [&]()
                    {
                        cv::cornerMinEigenVal(
                                gray32f,
                                benchmarkMinEigen,
                                harrisBlockSize,
                                harrisSobelKernel
                        );
                    }
            );


    // ========================================================
    // 17. VALIDACIÓN POST-BENCHMARK HARRIS
    // ========================================================

    double benchmarkHarrisMin;
    double benchmarkHarrisMax;


    cv::Point benchmarkHarrisMaxLocation;


    cv::minMaxLoc(
            benchmarkCustomHarris,
            &benchmarkHarrisMin,
            &benchmarkHarrisMax,
            nullptr,
            &benchmarkHarrisMaxLocation
    );


    std::cout
            << "Max Harris validation: "
            << benchmarkHarrisMax
            << "\n";


    std::cout
            << "Max location:          ("
            << benchmarkHarrisMaxLocation.x
            << ", "
            << benchmarkHarrisMaxLocation.y
            << ")\n";


    // ========================================================
    // 18. RESULTADOS HARRIS
    // ========================================================

    std::cout
            << "\n========================================\n"
            << " RESULTADOS HARRIS\n"
            << "========================================\n\n";


    printBenchmarkStats(
            customHarrisBenchmark
    );


    printBenchmarkStats(
            opencvHarrisBenchmark
    );


    printBenchmarkStats(
            minEigenBenchmark
    );


    // ========================================================
    // 19. ARCHIVOS GENERADOS
    // ========================================================

    std::cout
            << "\n========================================\n"
            << " ARCHIVOS GENERADOS\n"
            << "========================================\n";


    std::cout
            << "results/edges.png\n"
            << "results/hough_custom.jpg\n"
            << "results/hough_opencv.jpg\n"
            << "results/harris_custom.jpg\n"
            << "results/harris_opencv.jpg\n"
            << "results/harris_custom_corners.jpg\n"
            << "results/harris_opencv_corners.jpg\n";


    std::cout
            << "\n========================================\n"
            << " EXPERIMENTO FINALIZADO\n"
            << "========================================\n";


    return 0;
}