//
// Created by andru on 8/24/2026.
//
/**
 * @file Benchmark.h
 * @brief Herramientas para evaluar el rendimiento del debayering.
 *
 * Este módulo mide exclusivamente el tiempo de ejecución del
 * algoritmo Debayer::bilinear().
 *
 * No incluye:
 *
 * - Lectura del archivo RAW.
 * - Decodificación mediante LibRaw.
 * - Balance de blancos.
 * - Corrección de color.
 * - Gamma / sRGB.
 * - Escritura de imágenes.
 *
 * De esta manera los resultados pueden compararse posteriormente
 * entre plataformas x86 y ARM.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <opencv2/core.hpp>

#include <string>
#include <vector>


/**
 * @struct BenchmarkResult
 * @brief Resultados estadísticos del benchmark.
 */
struct BenchmarkResult
{
    // Número de repeticiones medidas.
    int iterations = 0;

    // Resolución.
    double megaPixels = 0.0;

    // Tiempos en milisegundos.
    double meanMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
    double stdDevMs = 0.0;

    // Métricas derivadas.
    double megaPixelsPerSecond = 0.0;
    double millisecondsPerMegaPixel = 0.0;
    double nanoSecondsPerPixel = 0.0;
    double equivalentFPS = 0.0;

    // Tiempos individuales.
    std::vector<double> samplesMs;

    /**
     * @brief Última imagen reconstruida.
     *
     * Permite reutilizar el resultado del benchmark sin tener
     * que volver a ejecutar Debayer::bilinear().
     */
    cv::Mat output;
};


class Benchmark
{
public:

    /**
     * @brief Ejecuta múltiples veces el debayering y calcula estadísticas.
     *
     * Antes de iniciar las mediciones se realiza una ejecución de
     * calentamiento que NO se incluye en los resultados.
     *
     * @param raw Imagen Bayer CV_16UC1.
     * @param bayerPattern Patrón Bayer RGGB/BGGR/GRBG/GBRG.
     * @param iterations Número de ejecuciones medidas.
     *
     * @return Estructura con métricas y última imagen reconstruida.
     */
    static BenchmarkResult runDebayer(
        const cv::Mat& raw,
        const std::string& bayerPattern,
        int iterations
    );


    /**
     * @brief Imprime las estadísticas del benchmark.
     *
     * @param result Resultados calculados previamente.
     */
    static void print(
        const BenchmarkResult& result
    );


    /**
     * @brief Guarda los resultados en formato CSV.
     *
     * Si el archivo no existe, crea también el encabezado.
     *
     * @param result Resultados del benchmark.
     * @param outputPath Archivo CSV de destino.
     * @param imageName Nombre de la imagen utilizada.
     */
    static void appendCSV(
        const BenchmarkResult& result,
        const std::string& outputPath,
        const std::string& imageName
    );
};

#endif