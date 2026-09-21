/**
 * ============================================================
 * TAREA 3 - BENCHMARK DE ALGORITMOS
 * ============================================================
 *
 * Archivo:
 *      Benchmark.cpp
 *
 * Objetivo:
 *      Implementar la medición de tiempo y el cálculo de
 *      estadísticas para comparar algoritmos en PC y ARM.
 *
 * Métricas calculadas:
 *
 *      - media
 *      - desviación estándar
 *      - mínimo
 *      - mediana
 *      - P95
 *      - máximo
 *
 * ============================================================
 */

#include "Benchmark.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>


// ============================================================
// FUNCIÓN AUXILIAR PARA PERCENTILES
// ============================================================

/**
 * @brief Calcula un percentil sobre un vector ordenado.
 *
 * @param sortedTimes Vector de tiempos ya ordenado.
 * @param percentile Percentil en rango 0.0 - 1.0.
 *
 * Ejemplos:
 *
 *      0.50 -> mediana
 *      0.95 -> P95
 */
static double calculatePercentile(
        const std::vector<double>& sortedTimes,
        double percentile
)
{
    if (sortedTimes.empty())
    {
        throw std::invalid_argument(
                "No se puede calcular percentil de un vector vacio."
        );
    }


    // Posición real dentro del vector.
    const double position =
            percentile
            *
            static_cast<double>(
                    sortedTimes.size() - 1
            );


    const std::size_t lowerIndex =
            static_cast<std::size_t>(
                    std::floor(position)
            );


    const std::size_t upperIndex =
            static_cast<std::size_t>(
                    std::ceil(position)
            );


    // Si el percentil cae exactamente sobre una posición,
    // no necesitamos interpolar.
    if (lowerIndex == upperIndex)
    {
        return sortedTimes[lowerIndex];
    }


    // Interpolación lineal entre los dos valores vecinos.
    const double fraction =
            position
            -
            static_cast<double>(
                    lowerIndex
            );


    return
            sortedTimes[lowerIndex]
            +
            fraction
            *
            (
                    sortedTimes[upperIndex]
                    -
                    sortedTimes[lowerIndex]
            );
}


// ============================================================
// FUNCIÓN PRINCIPAL DE BENCHMARK
// ============================================================

BenchmarkStats benchmark(
        const std::string& name,
        int warmupIterations,
        int measuredIterations,
        const std::function<void()>& function
)
{
    // --------------------------------------------------------
    // 1. Validar parámetros
    // --------------------------------------------------------

    if (warmupIterations < 0)
    {
        throw std::invalid_argument(
                "warmupIterations no puede ser negativo."
        );
    }


    if (measuredIterations <= 0)
    {
        throw std::invalid_argument(
                "measuredIterations debe ser mayor que cero."
        );
    }


    // --------------------------------------------------------
    // 2. Warm-up
    // --------------------------------------------------------
    //
    // Estas ejecuciones NO forman parte de las estadísticas.
    //
    // Su objetivo es reducir efectos como:
    //
    //      - inicialización de memoria
    //      - carga de código
    //      - cachés
    //      - inicialización interna de librerías
    //
    // --------------------------------------------------------

    for (int iteration = 0;
         iteration < warmupIterations;
         ++iteration)
    {
        function();
    }


    // --------------------------------------------------------
    // 3. Vector donde almacenaremos cada medición
    // --------------------------------------------------------

    std::vector<double> timesMs;

    timesMs.reserve(
            measuredIterations
    );


    // --------------------------------------------------------
    // 4. Ejecutar iteraciones medidas
    // --------------------------------------------------------

    for (int iteration = 0;
         iteration < measuredIterations;
         ++iteration)
    {
        // steady_clock es apropiado para benchmarks porque
        // representa un reloj monotónico.
        const auto start =
                std::chrono::steady_clock::now();


        function();


        const auto end =
                std::chrono::steady_clock::now();


        // Convertimos el tiempo transcurrido a milisegundos.
        const double elapsedMs =
                std::chrono::duration<
                        double,
                        std::milli
                >(
                        end - start
                ).count();


        timesMs.push_back(
                elapsedMs
        );
    }


    // ========================================================
    // 5. MEDIA
    // ========================================================

    const double sum =
            std::accumulate(
                    timesMs.begin(),
                    timesMs.end(),
                    0.0
            );


    const double mean =
            sum
            /
            static_cast<double>(
                    timesMs.size()
            );


    // ========================================================
    // 6. DESVIACIÓN ESTÁNDAR
    // ========================================================

    double squaredDifferenceSum =
            0.0;


    for (const double value : timesMs)
    {
        const double difference =
                value - mean;


        squaredDifferenceSum +=
                difference
                *
                difference;
    }


    const double standardDeviation =
            std::sqrt(
                    squaredDifferenceSum
                    /
                    static_cast<double>(
                            timesMs.size()
                    )
            );


    // ========================================================
    // 7. ORDENAR COPIA PARA MEDIANA Y P95
    // ========================================================

    std::vector<double> sortedTimes =
            timesMs;


    std::sort(
            sortedTimes.begin(),
            sortedTimes.end()
    );


    // ========================================================
    // 8. CALCULAR ESTADÍSTICAS
    // ========================================================

    const double minimum =
            sortedTimes.front();


    const double maximum =
            sortedTimes.back();


    const double median =
            calculatePercentile(
                    sortedTimes,
                    0.50
            );


    const double p95 =
            calculatePercentile(
                    sortedTimes,
                    0.95
            );


    // ========================================================
    // 9. CONSTRUIR RESULTADO
    // ========================================================

    BenchmarkStats stats;

    stats.name =
            name;

    stats.meanMs =
            mean;

    stats.stdDevMs =
            standardDeviation;

    stats.minMs =
            minimum;

    stats.medianMs =
            median;

    stats.p95Ms =
            p95;

    stats.maxMs =
            maximum;


    return stats;
}