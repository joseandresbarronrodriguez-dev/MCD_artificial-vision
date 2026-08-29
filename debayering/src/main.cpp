/*
 * Práctica Debayering
 * Visión Artificial
 * Andrés Barrón
 * 2026-08-22
 */

/**
 * @file main.cpp
 * @brief Aplicación principal para reconstrucción Bayer RAW -> RGB.
 *
 * Flujo general:
 *
 * 1. Selección automática/manual de una imagen RAW.
 * 2. Lectura del mosaico Bayer mediante LibRaw.
 * 3. Extracción de metadatos del sensor.
 * 4. Validación del RAW.
 * 5. Generación de referencia LibRaw Linear.
 * 6. Evaluación automática de los patrones:
 *
 *      RGGB
 *      BGGR
 *      GRBG
 *      GBRG
 *
 * 7. Selección del patrón Bayer con mayor PSNR respecto
 *    a la referencia LibRaw Linear.
 * 8. Debayering bilineal propio usando el patrón seleccionado.
 * 9. Benchmark de cinco iteraciones.
 * 10. Generación RGB lineal de 16 bits.
 * 11. Generación de preview sRGB de 8 bits.
 * 12. Generación de referencia LibRaw AHD.
 * 13. Comparaciones visuales.
 *
 * La orientación de la imagen se omite temporalmente.
 */

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "RawLoader.h"
#include "ImageUtils.h"
#include "Benchmark.h"
#include "LibRawReference.h"
#include "BayerSelector.h"

namespace fs = std::filesystem;


/**
 * @brief Determina si una extensión corresponde a un archivo RAW.
 *
 * @param extension Extensión del archivo incluyendo el punto.
 *
 * @return true si la extensión corresponde a un RAW soportado.
 */
bool isRawExtension(std::string extension)
{
    /*
     * Convertimos la extensión a minúsculas.
     *
     * Esto permite reconocer:
     *
     * .CR2
     * .cr2
     * .Cr2
     */
    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char c)
        {
            return static_cast<char>(
                std::tolower(c)
            );
        }
    );

    return
        extension == ".cr2" ||
        extension == ".cr3" ||
        extension == ".nef" ||
        extension == ".dng" ||
        extension == ".arw" ||
        extension == ".raf";
}


/**
 * @brief Busca archivos RAW dentro de un directorio.
 *
 * Si solamente existe un archivo, este se selecciona
 * automáticamente.
 *
 * Si existen varios archivos, el usuario puede elegir
 * uno mediante consola.
 *
 * @param directory Directorio donde se encuentran los RAW.
 *
 * @return Ruta del RAW seleccionado.
 */
fs::path selectRawFile(
    const fs::path& directory)
{
    // ========================================================
    // Verificar existencia del directorio
    // ========================================================

    if (!fs::exists(directory))
    {
        throw std::runtime_error(
            "No existe el directorio RAW: " +
            directory.string()
        );
    }


    // ========================================================
    // Buscar archivos RAW
    // ========================================================

    std::vector<fs::path> rawFiles;

    for (const auto& entry :
         fs::directory_iterator(directory))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        if (
            isRawExtension(
                entry.path().extension().string()
            )
        )
        {
            rawFiles.push_back(
                entry.path()
            );
        }
    }


    // ========================================================
    // Validar resultados
    // ========================================================

    if (rawFiles.empty())
    {
        throw std::runtime_error(
            "No se encontraron imagenes RAW en: " +
            directory.string()
        );
    }


    // ========================================================
    // Ordenar archivos
    // ========================================================

    std::sort(
        rawFiles.begin(),
        rawFiles.end()
    );


    // ========================================================
    // Mostrar archivos disponibles
    // ========================================================

    std::cout
        << "------------------------------------------\n"
        << " ARCHIVOS RAW DISPONIBLES\n"
        << "------------------------------------------\n";

    for (std::size_t i = 0;
         i < rawFiles.size();
         ++i)
    {
        std::cout
            << "["
            << i + 1
            << "] "
            << rawFiles[i].filename().string()
            << '\n';
    }


    // ========================================================
    // Selección automática si solamente existe uno
    // ========================================================

    if (rawFiles.size() == 1)
    {
        std::cout
            << "\nUnico RAW encontrado.\n"
            << "Seleccionado automaticamente: "
            << rawFiles.front().filename().string()
            << "\n\n";

        return rawFiles.front();
    }


    // ========================================================
    // Selección manual
    // ========================================================

    std::size_t selection = 0;

    std::cout
        << "\nSeleccione una imagen [1-"
        << rawFiles.size()
        << "]: ";

    std::cin >> selection;

    if (
        selection < 1 ||
        selection > rawFiles.size()
    )
    {
        throw std::runtime_error(
            "Seleccion RAW invalida."
        );
    }

    return rawFiles[selection - 1];
}


/**
 * @brief Punto de entrada principal.
 *
 * @param argc Número de argumentos.
 * @param argv Argumentos de línea de comandos.
 *
 * @return 0 si el programa finaliza correctamente.
 * @return 1 si ocurre algún error.
 */
int main(int argc, char* argv[])
{
    // ========================================================
    // Encabezado
    // ========================================================

    std::cout
        << "==========================================\n"
        << "        DEBAYERING - RAW TO RGB\n"
        << "==========================================\n\n";

    std::cout
        << "OpenCV version: "
        << CV_VERSION
        << "\n\n";


    try
    {
        // ====================================================
        // 1. Seleccionar archivo RAW
        // ====================================================

        fs::path rawPath;

        /*
         * Si se proporciona una ruta como argumento,
         * se utiliza directamente.
         *
         * Si no existe argumento se utiliza el selector.
         */
        if (argc >= 2)
        {
            rawPath = argv[1];
        }
        else
        {
            rawPath =
                selectRawFile(
                    "data/raw"
                );
        }

        std::cout
            << "Archivo seleccionado:\n"
            << rawPath.string()
            << "\n\n";


        // ====================================================
        // 2. Cargar RAW
        // ====================================================

        RawImageData raw =
            RawLoader::load(
                rawPath.string()
            );


        // ====================================================
        // 3. Estadísticas básicas del mosaico
        // ====================================================

        double rawMin = 0.0;
        double rawMax = 0.0;

        cv::minMaxLoc(
            raw.rawImage,
            &rawMin,
            &rawMax
        );


        // ====================================================
        // 4. Información del RAW
        // ====================================================

        std::cout
            << "------------------------------------------\n"
            << " INFORMACION DEL RAW\n"
            << "------------------------------------------\n";

        std::cout
            << "Camara: "
            << raw.cameraMake
            << " "
            << raw.cameraModel
            << '\n';

        std::cout
            << "Sensor RAW completo: "
            << raw.rawWidth
            << " x "
            << raw.rawHeight
            << '\n';

        std::cout
            << "Area visible: "
            << raw.width
            << " x "
            << raw.height
            << '\n';

        std::cout
            << "Margen superior: "
            << raw.topMargin
            << '\n';

        std::cout
            << "Margen izquierdo: "
            << raw.leftMargin
            << '\n';

        std::cout
            << "Patron Bayer reportado por LibRaw: "
            << raw.bayerPattern
            << '\n';

        std::cout
            << "Nivel negro: "
            << raw.blackLevel
            << '\n';

        std::cout
            << "Maximo reportado: "
            << raw.maximum
            << '\n';

        std::cout
            << "Minimo observado: "
            << rawMin
            << '\n';

        std::cout
            << "Maximo observado: "
            << rawMax
            << '\n';

        std::cout
            << "Formato matriz: CV_16UC1\n";


        const double megaPixels =
            (
                static_cast<double>(raw.width) *
                static_cast<double>(raw.height)
            ) /
            1'000'000.0;

        std::cout
            << "Resolucion: "
            << megaPixels
            << " MP\n";


        // ====================================================
        // 5. Balance de blancos
        // ====================================================

        std::cout
            << "\nBalance de blancos de camara:\n";

        std::cout
            << "  R: "
            << raw.whiteBalanceR
            << '\n';

        std::cout
            << "  G: "
            << raw.whiteBalanceG
            << '\n';

        std::cout
            << "  B: "
            << raw.whiteBalanceB
            << '\n';


        // ====================================================
        // 6. Matriz de corrección de color
        // ====================================================

        std::cout
            << "\nMatriz de correccion de color:\n";

        if (raw.hasColorMatrix)
        {
            for (int i = 0; i < 3; ++i)
            {
                std::cout
                    << "  [ "
                    << raw.cameraToRgb(i, 0)
                    << "  "
                    << raw.cameraToRgb(i, 1)
                    << "  "
                    << raw.cameraToRgb(i, 2)
                    << " ]\n";
            }
        }
        else
        {
            std::cout
                << "  No disponible. "
                << "Se utilizara matriz identidad.\n";
        }


        // ====================================================
        // 7. Validaciones
        // ====================================================

        std::cout
            << "\n------------------------------------------\n"
            << " VALIDACION DEL RAW\n"
            << "------------------------------------------\n";


        const bool validRaw =
            !raw.rawImage.empty();

        const bool validType =
            raw.rawImage.type() == CV_16UC1;

        const bool validMetadataPattern =
            raw.bayerPattern == "RGGB" ||
            raw.bayerPattern == "BGGR" ||
            raw.bayerPattern == "GRBG" ||
            raw.bayerPattern == "GBRG";


        std::cout
            << "Matriz RAW: "
            << (
                validRaw
                ? "OK"
                : "ERROR"
            )
            << '\n';

        std::cout
            << "Formato 16 bits / 1 canal: "
            << (
                validType
                ? "OK"
                : "ERROR"
            )
            << '\n';

        std::cout
            << "Patron Bayer metadata valido: "
            << (
                validMetadataPattern
                ? "OK"
                : "ADVERTENCIA"
            )
            << '\n';


        if (!validRaw)
        {
            throw std::runtime_error(
                "La matriz RAW esta vacia."
            );
        }

        if (!validType)
        {
            throw std::runtime_error(
                "El RAW no tiene formato CV_16UC1."
            );
        }


        // ====================================================
        // 8. Crear directorio de resultados
        // ====================================================

        fs::create_directories(
            "results"
        );


        // ====================================================
        // 9. Generar referencia LibRaw Linear
        // ====================================================

        /*
         * Esta referencia tendrá dos funciones:
         *
         * 1. Servir para seleccionar automáticamente el CFA.
         * 2. Servir posteriormente para comparación visual.
         */
        std::cout
            << "\n------------------------------------------\n"
            << " REFERENCIA LIBRAW LINEAR\n"
            << "------------------------------------------\n";

        std::cout
            << "Generando referencia LibRaw Linear...\n";


        cv::Mat librawLinear =
            LibRawReference::render(
                rawPath.string(),
                0
            );


        if (librawLinear.empty())
        {
            throw std::runtime_error(
                "LibRaw Linear produjo una imagen vacia."
            );
        }


        const std::string librawLinearPath =
            "results/libraw_linear_reference.png";


        if (!cv::imwrite(
                librawLinearPath,
                librawLinear
            ))
        {
            throw std::runtime_error(
                "No se pudo guardar LibRaw Linear."
            );
        }


        std::cout
            << "Referencia LibRaw Linear: OK\n";


        // ====================================================
        // 10. Selección automática del patrón Bayer
        // ====================================================

        std::cout
            << "\n------------------------------------------\n"
            << " SELECCION AUTOMATICA DEL PATRON BAYER\n"
            << "------------------------------------------\n";

        /*
         * Se prueban automáticamente:
         *
         * RGGB
         * BGGR
         * GRBG
         * GBRG
         *
         * sobre un ROI central.
         *
         * El patrón con mayor PSNR respecto a LibRaw Linear
         * será seleccionado.
         */
        BayerSelectionResult selection =
            BayerSelector::select(
                raw.rawImage,
                librawLinear,
                raw.bayerPattern,
                raw.blackLevel,
                raw.maximum,
                raw.whiteBalanceR,
                raw.whiteBalanceG,
                raw.whiteBalanceB,
                raw.cameraToRgb
            );


        // ====================================================
        // 11. Mostrar resultados de selección
        // ====================================================

        std::cout
            << "Patron reportado por LibRaw: "
            << raw.bayerPattern
            << "\n\n";


        for (const BayerScore& score :
     selection.scores)
        {
            std::cout
                << score.pattern
                << '\n';

            std::cout
                << "  PSNR:          "
                << score.psnr
                << " dB\n";

            std::cout
                << "  MSE:           "
                << score.mse
                << '\n';

            std::cout
                << "  Error borde:   "
                << score.gradientError
                << '\n';

            std::cout
                << "  Error chroma:  "
                << score.chromaError
                << '\n';

            std::cout
                << "  Score final:   "
                << score.finalScore
                << "\n\n";
        }


        std::cout
            << "Ganador experimental: "
            << selection.empiricalWinner
            << '\n';

        std::cout
            << "Margen de confianza: "
            << selection.confidenceMargin
            << '\n';

        std::cout
            << "Patron metadata: "
            << raw.bayerPattern
            << '\n';

        std::cout
            << "Patron seleccionado finalmente: "
            << selection.selectedPattern
            << '\n';


        if (selection.usedMetadataFallback)
        {
            std::cout
                << "Seleccion final: metadata utilizada "
                << "por baja confianza del selector.\n";
        }
        else
        {
            std::cout
                << "Seleccion final: resultado experimental.\n";
        }


        // ====================================================
        // 12. Benchmark + Debayering final
        // ====================================================

        std::cout
            << "\n------------------------------------------\n"
            << " DEBAYERING BILINEAL FINAL\n"
            << "------------------------------------------\n";

        std::cout
            << "Resolucion: "
            << raw.width
            << " x "
            << raw.height
            << '\n';

        std::cout
            << "Patron utilizado: "
            << selection.selectedPattern
            << '\n';


        /*
         * Conservamos cinco iteraciones para nuestra
         * evaluación experimental.
         */
        constexpr int benchmarkIterations = 5;


        BenchmarkResult benchmark =
            Benchmark::runDebayer(
                raw.rawImage,
                selection.selectedPattern,
                benchmarkIterations
            );


        Benchmark::print(
            benchmark
        );


        /*
         * Reutilizamos la última reconstrucción producida
         * durante el benchmark.
         */
        cv::Mat rgb =
            benchmark.output;


        if (rgb.empty())
        {
            throw std::runtime_error(
                "El Debayering produjo una imagen vacia."
            );
        }


        std::cout
            << "\nReconstruccion completada.\n"
            << "Salida: CV_16UC3\n"
            << "Canales internos: RGB\n";


        // ====================================================
        // 13. Guardar benchmark
        // ====================================================

        Benchmark::appendCSV(
            benchmark,
            "results/benchmark.csv",
            rawPath.filename().string()
        );


        // ====================================================
        // 14. Guardar RGB lineal de 16 bits
        // ====================================================

        const std::string linearPath =
            "results/custom_bilinear_linear16.png";


        ImageUtils::saveLinear16(
            rgb,
            linearPath
        );


        // ====================================================
        // 15. Crear preview propio
        // ====================================================

        /*
         * createPreview8 realiza:
         *
         * - eliminación de black level
         * - normalización
         * - white balance
         * - corrección de color
         * - transformación sRGB
         *
         * Devuelve BGR CV_8UC3.
         */
        cv::Mat customPreview =
            ImageUtils::createPreview8(
                rgb,
                raw.blackLevel,
                raw.maximum,
                raw.whiteBalanceR,
                raw.whiteBalanceG,
                raw.whiteBalanceB,
                raw.cameraToRgb
            );


        if (customPreview.empty())
        {
            throw std::runtime_error(
                "No se pudo generar el preview propio."
            );
        }


        const std::string previewPath =
            "results/custom_bilinear_preview.png";


        if (!cv::imwrite(
                previewPath,
                customPreview
            ))
        {
            throw std::runtime_error(
                "No se pudo guardar el preview propio."
            );
        }


        // ====================================================
        // 16. Generar referencia LibRaw AHD
        // ====================================================

        std::cout
            << "\n------------------------------------------\n"
            << " REFERENCIA LIBRAW AHD\n"
            << "------------------------------------------\n";

        std::cout
            << "Generando referencia LibRaw AHD...\n";


        cv::Mat librawAHD =
            LibRawReference::render(
                rawPath.string(),
                3
            );


        if (librawAHD.empty())
        {
            throw std::runtime_error(
                "LibRaw AHD produjo una imagen vacia."
            );
        }


        const std::string librawAHDPath =
            "results/libraw_ahd_reference.png";


        if (!cv::imwrite(
                librawAHDPath,
                librawAHD
            ))
        {
            throw std::runtime_error(
                "No se pudo guardar LibRaw AHD."
            );
        }


        std::cout
            << "Referencia LibRaw AHD: OK\n";


        // ====================================================
        // 17. Comparación con LibRaw Linear
        // ====================================================

        const std::string comparisonLinearPath =
            "results/comparison_bilinear_vs_libraw_linear.png";


        ImageUtils::saveComparison(
            customPreview,
            librawLinear,
            "Nuestro Bilinear",
            "LibRaw Linear",
            comparisonLinearPath
        );


        // ====================================================
        // 18. Comparación con LibRaw AHD
        // ====================================================

        const std::string comparisonAHDPath =
            "results/comparison_bilinear_vs_libraw_ahd.png";


        ImageUtils::saveComparison(
            customPreview,
            librawAHD,
            "Nuestro Bilinear",
            "LibRaw AHD",
            comparisonAHDPath
        );


        // ====================================================
        // 19. Resumen final
        // ====================================================

        std::cout
            << "\n==========================================\n"
            << " RESULTADOS GENERADOS\n"
            << "==========================================\n";


        std::cout
            << "\nPatron Bayer metadata:\n"
            << raw.bayerPattern
            << '\n';


        std::cout
            << "\nPatron Bayer seleccionado:\n"
            << selection.selectedPattern
            << '\n';


        std::cout
            << "\n[Debayer propio]\n"
            << linearPath
            << '\n'
            << previewPath
            << '\n';


        std::cout
            << "\n[Referencias LibRaw]\n"
            << librawLinearPath
            << '\n'
            << librawAHDPath
            << '\n';


        std::cout
            << "\n[Comparaciones]\n"
            << comparisonLinearPath
            << '\n'
            << comparisonAHDPath
            << '\n';


        std::cout
            << "\n[Benchmark]\n"
            << "results/benchmark.csv\n";


        std::cout
            << "\nProceso terminado correctamente.\n";
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "\n==========================================\n"
            << " ERROR\n"
            << "==========================================\n"
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}