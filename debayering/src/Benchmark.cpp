//
// Created by andru on 8/24/2026.
//
/**
 * @file Benchmark.cpp
 * @brief Implementación de las mediciones de rendimiento.
 */

#include "Benchmark.h"
#include "Debayer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace fs = std::filesystem;


BenchmarkResult Benchmark::runDebayer(
    const cv::Mat& raw,
    const std::string& bayerPattern,
    int iterations)
{
    if (raw.empty())
    {
        throw std::invalid_argument(
            "Benchmark: la imagen RAW esta vacia."
        );
    }

    if (iterations <= 0)
    {
        throw std::invalid_argument(
            "Benchmark: iterations debe ser mayor que cero."
        );
    }

    BenchmarkResult result;

    result.iterations = iterations;

    const long long totalPixels =
        static_cast<long long>(raw.rows) *
        static_cast<long long>(raw.cols);

    result.megaPixels =
        static_cast<double>(totalPixels) /
        1'000'000.0;


    // ========================================================
    // 1. Warm-up
    // ========================================================

    /*
     * La primera ejecución puede incluir efectos adicionales:
     *
     * - páginas de memoria todavía no utilizadas;
     * - cachés frías;
     * - inicializaciones internas.
     *
     * Por ello se realiza una ejecución previa que no se mide.
     */
    std::cout
        << "Warm-up del algoritmo...\n";

    result.output =
        Debayer::bilinear(
            raw,
            bayerPattern
        );


    // ========================================================
    // 2. Mediciones
    // ========================================================

    result.samplesMs.reserve(iterations);

    std::cout
        << "Ejecutando "
        << iterations
        << " iteraciones medidas...\n\n";

    for (int i = 0; i < iterations; ++i)
    {
        /*
         * steady_clock es apropiado para benchmarks porque
         * es monotónico y no depende de cambios en el reloj
         * del sistema operativo.
         */
        const auto start =
            std::chrono::steady_clock::now();

        result.output =
            Debayer::bilinear(
                raw,
                bayerPattern
            );

        const auto end =
            std::chrono::steady_clock::now();

        const double elapsedMs =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        result.samplesMs.push_back(
            elapsedMs
        );

        std::cout
            << "Iteracion "
            << std::setw(2)
            << i + 1
            << ": "
            << std::fixed
            << std::setprecision(3)
            << elapsedMs
            << " ms\n";
    }


    // ========================================================
    // 3. Estadísticas básicas
    // ========================================================

    result.minMs =
        *std::min_element(
            result.samplesMs.begin(),
            result.samplesMs.end()
        );

    result.maxMs =
        *std::max_element(
            result.samplesMs.begin(),
            result.samplesMs.end()
        );

    const double sum =
        std::accumulate(
            result.samplesMs.begin(),
            result.samplesMs.end(),
            0.0
        );

    result.meanMs =
        sum /
        static_cast<double>(iterations);


    // ========================================================
    // 4. Desviación estándar
    // ========================================================

    double variance = 0.0;

    for (const double sample :
         result.samplesMs)
    {
        const double difference =
            sample - result.meanMs;

        variance +=
            difference * difference;
    }

    variance /=
        static_cast<double>(iterations);

    result.stdDevMs =
        std::sqrt(variance);


    // ========================================================
    // 5. Métricas derivadas
    // ========================================================

    const double meanSeconds =
        result.meanMs / 1000.0;

    result.megaPixelsPerSecond =
        result.megaPixels /
        meanSeconds;

    result.millisecondsPerMegaPixel =
        result.meanMs /
        result.megaPixels;

    result.nanoSecondsPerPixel =
        (
            result.meanMs *
            1'000'000.0
        ) /
        static_cast<double>(totalPixels);

    result.equivalentFPS =
        1000.0 /
        result.meanMs;

    return result;
}


void Benchmark::print(
    const BenchmarkResult& result)
{
    std::cout
        << "\n------------------------------------------\n"
        << " BENCHMARK DEBAYERING\n"
        << "------------------------------------------\n";

#ifdef NDEBUG
    std::cout
        << "Modo compilacion: RELEASE\n";
#else
    std::cout
        << "Modo compilacion: DEBUG\n";
#endif

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "Iteraciones: "
        << result.iterations
        << '\n';

    std::cout
        << "Resolucion: "
        << result.megaPixels
        << " MP\n\n";

    std::cout
        << "Tiempo medio: "
        << result.meanMs
        << " ms\n";

    std::cout
        << "Tiempo minimo: "
        << result.minMs
        << " ms\n";

    std::cout
        << "Tiempo maximo: "
        << result.maxMs
        << " ms\n";

    std::cout
        << "Desviacion estandar: "
        << result.stdDevMs
        << " ms\n\n";

    std::cout
        << "Throughput: "
        << result.megaPixelsPerSecond
        << " MP/s\n";

    std::cout
        << "Costo por megapixel: "
        << result.millisecondsPerMegaPixel
        << " ms/MP\n";

    std::cout
        << "Costo por pixel: "
        << result.nanoSecondsPerPixel
        << " ns/pixel\n";

    std::cout
        << "FPS equivalentes: "
        << result.equivalentFPS
        << " FPS\n";
}


void Benchmark::appendCSV(
    const BenchmarkResult& result,
    const std::string& outputPath,
    const std::string& imageName)
{
    const bool exists =
        fs::exists(outputPath);

    std::ofstream file(
        outputPath,
        std::ios::app
    );

    if (!file.is_open())
    {
        throw std::runtime_error(
            "No se pudo abrir benchmark CSV."
        );
    }

    /*
     * Si el archivo todavía no existe, escribimos
     * primero los nombres de las columnas.
     */
    if (!exists)
    {
        file
            << "image,"
            << "build,"
            << "iterations,"
            << "megapixels,"
            << "mean_ms,"
            << "min_ms,"
            << "max_ms,"
            << "stddev_ms,"
            << "mp_per_s,"
            << "ms_per_mp,"
            << "ns_per_pixel,"
            << "fps\n";
    }

#ifdef NDEBUG
    const std::string buildMode = "Release";
#else
    const std::string buildMode = "Debug";
#endif

    file
        << imageName << ','
        << buildMode << ','
        << result.iterations << ','
        << result.megaPixels << ','
        << result.meanMs << ','
        << result.minMs << ','
        << result.maxMs << ','
        << result.stdDevMs << ','
        << result.megaPixelsPerSecond << ','
        << result.millisecondsPerMegaPixel << ','
        << result.nanoSecondsPerPixel << ','
        << result.equivalentFPS
        << '\n';
}