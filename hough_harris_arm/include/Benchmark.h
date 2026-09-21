//
// Created by andru on 9/20/2026.
//
/**
 * ============================================================
 * TAREA 3 - BENCHMARK DE ALGORITMOS
 * ============================================================
 *
 * Archivo:
 *      Benchmark.h
 *
 * Objetivo:
 *      Definir estructuras y funciones para medir el tiempo
 *      de ejecución de los algoritmos implementados.
 *
 * Estrategia:
 *
 *      1. Ejecutar varias iteraciones de warm-up.
 *      2. Ejecutar N iteraciones medidas.
 *      3. Registrar cada tiempo individual.
 *      4. Calcular estadísticas:
 *
 *              - media
 *              - desviación estándar
 *              - mínimo
 *              - mediana
 *              - P95
 *              - máximo
 *
 * Esto nos permitirá posteriormente comparar:
 *
 *      PC x86
 *      Raspberry Pi
 *      Orange Pi
 *      RB3 Gen 2
 *
 * utilizando exactamente la misma metodología.
 *
 * ============================================================
 */

#pragma once


#include <functional>
#include <string>
#include <vector>


// ============================================================
// ESTRUCTURA DE RESULTADOS
// ============================================================

/**
 * @brief Estadísticas generadas por una prueba de rendimiento.
 */
struct BenchmarkStats
{
    /**
     * Nombre del algoritmo probado.
     *
     * Ejemplos:
     *
     *      "Hough Eigen"
     *      "OpenCV HoughLines"
     */
    std::string name;


    /**
     * Tiempo promedio de ejecución.
     */
    double meanMs = 0.0;


    /**
     * Desviación estándar.
     *
     * Nos permite conocer la variabilidad de los tiempos.
     */
    double stdDevMs = 0.0;


    /**
     * Tiempo mínimo observado.
     */
    double minMs = 0.0;


    /**
     * Mediana.
     *
     * 50 % de las ejecuciones están por debajo de este valor.
     */
    double medianMs = 0.0;


    /**
     * Percentil 95.
     *
     * El 95 % de las ejecuciones tardaron menos que este valor.
     *
     * Esta métrica será especialmente interesante en ARM,
     * donde pueden aparecer variaciones por:
     *
     *      - sistema operativo
     *      - temperatura
     *      - frecuencia de CPU
     *      - procesos en segundo plano
     */
    double p95Ms = 0.0;


    /**
     * Tiempo máximo observado.
     */
    double maxMs = 0.0;
};


// ============================================================
// FUNCIÓN DE BENCHMARK
// ============================================================

/**
 * @brief Ejecuta y mide una función varias veces.
 *
 * @param name Nombre del algoritmo.
 *
 * @param warmupIterations Número de ejecuciones que no se
 *                         incluirán en las estadísticas.
 *
 * @param measuredIterations Número de ejecuciones medidas.
 *
 * @param function Función o algoritmo que será ejecutado.
 *
 * @return BenchmarkStats con los resultados.
 *
 *
 * Ejemplo de uso:
 *
 *      BenchmarkStats result =
 *          benchmark(
 *              "Hough Eigen",
 *              5,
 *              100,
 *              [&]()
 *              {
 *                  hough.compute(edges);
 *              }
 *          );
 */
BenchmarkStats benchmark(
        const std::string& name,
        int warmupIterations,
        int measuredIterations,
        const std::function<void()>& function
);