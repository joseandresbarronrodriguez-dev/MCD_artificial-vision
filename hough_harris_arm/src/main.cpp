/**
 * ============================================================
 * TAREA 3 - HOUGH TRANSFORM + HARRIS CORNER SCORE
 * ============================================================
 *
 * Archivo:
 *      main.cpp
 *
 * Objetivo de esta etapa:
 *
 *      1. Cargar una imagen.
 *      2. Convertirla a escala de grises.
 *      3. Detectar bordes utilizando Canny.
 *      4. Ejecutar nuestro acumulador de Hough.
 *      5. Obtener la línea con mayor número de votos.
 *      6. Comparar contra cv::HoughLines().
 *      7. Guardar resultados visuales.
 *
 * Todavía NO realizamos:
 *
 *      - benchmark de tiempos
 *      - Harris
 *
 * ============================================================
 */

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Implementación propia
#include "HoughAccumulator.h"
#include "HarrisScore.h"


// ============================================================
// FUNCIÓN AUXILIAR PARA DIBUJAR UNA LÍNEA DE HOUGH
// ============================================================

/**
 * @brief Dibuja una línea expresada en coordenadas polares.
 *
 * La Transformada de Hough representa una línea mediante:
 *
 *      rho
 *      theta
 *
 * utilizando:
 *
 *      rho = x*cos(theta) + y*sin(theta)
 *
 * Para visualizarla, convertimos esta representación en
 * dos puntos suficientemente alejados.
 *
 * @param image Imagen donde se dibujará la línea.
 * @param rho Distancia de la línea al origen.
 * @param theta Ángulo en radianes.
 * @param color Color BGR utilizado para dibujar.
 */
void drawHoughLine(
        cv::Mat& image,
        float rho,
        float theta,
        const cv::Scalar& color
)
{
    // Componentes de la normal de la línea.
    const double a =
            std::cos(theta);

    const double b =
            std::sin(theta);


    // Punto base de la línea.
    const double x0 =
            a * rho;

    const double y0 =
            b * rho;


    // Creamos dos puntos muy alejados para representar
    // visualmente la línea completa sobre la imagen.
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
// MAIN
// ============================================================

int main(int argc, char* argv[])
{
    std::cout
            << "\n========================================\n"
            << " HOUGH + HARRIS ARM BENCHMARK\n"
            << "========================================\n";


    // ========================================================
    // 1. RUTA DE LA IMAGEN
    // ========================================================
    //
    // Ruta por defecto:
    //
    //      data/input/test.jpg
    //
    // También podemos proporcionar otra imagen como argumento:
    //
    //      hough_harris_arm otra_imagen.jpg
    //
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


    // ========================================================
    // 2. CARGAR LA IMAGEN
    // ========================================================

    cv::Mat image =
            cv::imread(
                    imagePath,
                    cv::IMREAD_COLOR
            );


    if (image.empty())
    {
        std::cerr
                << "\nERROR: No se pudo cargar la imagen.\n"
                << "Ruta utilizada: "
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
    // 3. CONVERSIÓN A ESCALA DE GRISES
    // ========================================================

    cv::Mat gray;


    cv::cvtColor(
            image,
            gray,
            cv::COLOR_BGR2GRAY
    );


    // ========================================================
    // 4. DETECCIÓN DE BORDES CON CANNY
    // ========================================================
    //
    // Canny se utiliza únicamente como preprocesamiento.
    //
    // Para que posteriormente el benchmark sea justo,
    // tanto nuestro Hough como OpenCV recibirán exactamente
    // la misma imagen de bordes.
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


    // Número de píxeles que participarán en la votación.
    const int edgePixels =
            cv::countNonZero(edges);


    std::cout
            << "Pixeles de borde: "
            << edgePixels
            << "\n";


    // ========================================================
    // 5. EJECUTAR NUESTRO ACUMULADOR DE HOUGH
    // ========================================================

    constexpr int THETA_BINS =
            180;


    HoughAccumulator hough(
            THETA_BINS
    );


    HoughResult result =
            hough.compute(
                    edges
            );


    // ========================================================
    // 6. INFORMACIÓN DEL ACUMULADOR
    // ========================================================

    std::cout
            << "\n========================================\n"
            << " ACUMULADOR DE HOUGH\n"
            << "========================================\n";


    std::cout
            << "Rho maximo:          "
            << result.rhoMax
            << "\n";


    std::cout
            << "Bins de rho:         "
            << result.accumulator.rows()
            << "\n";


    std::cout
            << "Bins de theta:       "
            << result.accumulator.cols()
            << "\n";


    std::cout
            << "Tamano acumulador:   "
            << result.accumulator.rows()
            << " x "
            << result.accumulator.cols()
            << "\n";


    // ========================================================
    // 7. BUSCAR EL MÁXIMO DEL ACUMULADOR
    // ========================================================
    //
    // Eigen permite recuperar:
    //
    //      valor máximo
    //      fila del máximo
    //      columna del máximo
    //
    // La celda con más votos corresponde a la línea
    // dominante detectada por nuestro acumulador.
    // ========================================================

    Eigen::Index maxRhoIndex;
    Eigen::Index maxThetaIndex;


    const int maxVotes =
            result.accumulator.maxCoeff(
                    &maxRhoIndex,
                    &maxThetaIndex
            );


    // ========================================================
    // 8. CONVERTIR ÍNDICES A rho Y theta
    // ========================================================

    const int detectedRho =
            static_cast<int>(
                    maxRhoIndex
            )
            -
            result.rhoMax;


    const float detectedThetaDegrees =
            180.0f
            *
            static_cast<float>(
                    maxThetaIndex
            )
            /
            static_cast<float>(
                    result.thetaBins
            );


    const float detectedThetaRadians =
            detectedThetaDegrees
            *
            static_cast<float>(CV_PI)
            /
            180.0f;


    std::cout
            << "\nVoto maximo:         "
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
    // 9. HOUGH DE REFERENCIA UTILIZANDO OPENCV
    // ========================================================
    //
    // Utilizamos cv::HoughLines() para validar nuestra
    // implementación.
    //
    // Ambos algoritmos reciben la MISMA imagen de bordes.
    //
    // Resolución:
    //
    //      rho   = 1 pixel
    //      theta = 1 grado
    //
    // ========================================================

    constexpr int HOUGH_THRESHOLD =
            100;


    std::vector<cv::Vec2f> opencvLines;


    cv::HoughLines(
            edges,
            opencvLines,

            // Resolución de rho
            1.0,

            // Resolución angular
            CV_PI / 180.0,

            // Número mínimo de votos
            HOUGH_THRESHOLD
    );


    // ========================================================
    // 10. MOSTRAR RESULTADO DE OPENCV
    // ========================================================

    std::cout
            << "\n========================================\n"
            << " COMPARACION CON OPENCV\n"
            << "========================================\n";


    std::cout
            << "Lineas detectadas por OpenCV: "
            << opencvLines.size()
            << "\n";


    if (!opencvLines.empty())
    {
        const float opencvRho =
                opencvLines[0][0];


        const float opencvThetaRadians =
                opencvLines[0][1];


        const float opencvThetaDegrees =
                opencvThetaRadians
                *
                180.0f
                /
                static_cast<float>(CV_PI);


        std::cout
                << "\nLinea principal OpenCV:\n";


        std::cout
                << "Rho:                  "
                << opencvRho
                << "\n";


        std::cout
                << "Theta:                "
                << opencvThetaDegrees
                << " grados\n";
    }


    // ========================================================
    // 11. GUARDAR IMAGEN DE BORDES
    // ========================================================

    cv::imwrite(
            "results/edges.png",
            edges
    );


    // ========================================================
    // 12. DIBUJAR NUESTRA LÍNEA
    // ========================================================

    cv::Mat customResult =
            image.clone();


    drawHoughLine(
            customResult,

            static_cast<float>(
                    detectedRho
            ),

            detectedThetaRadians,

            // Rojo en formato BGR
            cv::Scalar(
                    0,
                    0,
                    255
            )
    );


    cv::imwrite(
            "results/hough_custom.jpg",
            customResult
    );


    // ========================================================
    // 13. DIBUJAR LA LÍNEA DE OPENCV
    // ========================================================

    if (!opencvLines.empty())
    {
        cv::Mat opencvResult =
                image.clone();


        drawHoughLine(
                opencvResult,

                opencvLines[0][0],

                opencvLines[0][1],

                // Verde en formato BGR
                cv::Scalar(
                        0,
                        255,
                        0
                )
        );


        cv::imwrite(
                "results/hough_opencv.jpg",
                opencvResult
        );
    }


    // ========================================================
    // 14. RESULTADOS GENERADOS
    // ========================================================

    std::cout
            << "\n========================================\n"
            << " ARCHIVOS GENERADOS\n"
            << "========================================\n";


    std::cout
            << "results/edges.png\n"
            << "results/hough_custom.jpg\n";


    if (!opencvLines.empty())
    {
        std::cout
                << "results/hough_opencv.jpg\n";
    }


    std::cout
            << "\n========================================\n"
            << " HOUGH EJECUTADO CORRECTAMENTE\n"
            << "========================================\n";

    // ========================================================
    // 15. EJECUTAR NUESTRO SCORE DE HARRIS
    // ========================================================
    //
    // En esta primera prueba solamente validamos:
    //
    //      - dimensiones del mapa
    //      - valor mínimo
    //      - valor máximo
    //
    // Todavía NO comparamos contra cv::cornerHarris().
    // ========================================================

    std::cout
                    << "\n========================================\n"
                    << " HARRIS SCORE PROPIO\n"
                    << "========================================\n";


    // Parámetros que mantendremos fijos posteriormente
    // durante el benchmark PC vs ARM.
    constexpr int HARRIS_BLOCK_SIZE =
                    3;

    constexpr int HARRIS_SOBEL_KERNEL =
                    3;

    constexpr float HARRIS_K =
                    0.04f;


    // Creamos nuestro detector.
    HarrisScore harris(
            HARRIS_BLOCK_SIZE,
            HARRIS_SOBEL_KERNEL,
            HARRIS_K
    );


    // Calculamos el mapa completo de respuesta.
    cv::Mat harrisResponse =
                    harris.compute(
                            gray
                    );


    // ========================================================
    // 16. VALIDAR DIMENSIONES
    // ========================================================

    std::cout
                    << "Tamano respuesta:    "
                    << harrisResponse.cols
                    << " x "
                    << harrisResponse.rows
                    << "\n";


    std::cout
                    << "Tipo OpenCV:         "
                    << harrisResponse.type()
                    << "\n";


    // ========================================================
    // 17. OBTENER MÍNIMO Y MÁXIMO
    // ========================================================
    //
    // Harris produce:
    //
    //      R grande positivo  -> posible esquina
    //
    //      R negativo         -> normalmente borde
    //
    //      R cercano a cero   -> región plana
    //
    // ========================================================

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
    // 18. NORMALIZAR PARA VISUALIZACIÓN
    // ========================================================
    //
    // El mapa Harris utiliza float y puede contener números
    // positivos y negativos.
    //
    // Para guardarlo como imagen convertimos el rango:
    //
    //      min ... max
    //
    // a:
    //
    //      0 ... 255
    //
    // Esta normalización se utiliza SOLO para visualización.
    // No modifica los datos que posteriormente compararemos.
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


    std::cout
                    << "Mapa guardado:       "
                    << "results/harris_custom.jpg\n";

    // ========================================================
    // 19. HARRIS DE REFERENCIA CON OPENCV
    // ========================================================
    //
    // Utilizamos cv::cornerHarris() para validar nuestra
    // implementación.
    //
    // IMPORTANTE:
    //
    // Ambos métodos utilizan:
    //
    //      blockSize = 3
    //      Sobel      = 3
    //      k          = 0.04
    //
    // Para que la comparación sea válida.
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
            HARRIS_BLOCK_SIZE,
            HARRIS_SOBEL_KERNEL,
            HARRIS_K
    );

    // ========================================================
    // 20. ESTADÍSTICAS DE OPENCV
    // ========================================================

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

    // ========================================================
    // 21. COMPARACIÓN NUMÉRICA DE LOS MAPAS
    // ========================================================
    //
    // cornerHarris() puede utilizar escalas internas diferentes.
    //
    // Por eso normalizamos ambos mapas antes de calcular el error.
    // Esto nos permite comparar principalmente la estructura
    // espacial de la respuesta.
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
                    << "RMSE normalizado:    "
                    << harrisRmse
                    << "\n";

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

    return 0;

}