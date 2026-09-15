//
// Created by andru on 9/7/2026.
//
#ifndef BENCHMARK_H
#define BENCHMARK_H

/*
 * ============================================================
 * Archivo: benchmark.h
 * Proyecto: Convolución separable y fusión de kernels en ARM
 *
 * Descripción:
 *   Define las estructuras y funciones utilizadas para medir
 *   el tiempo de ejecución de distintas implementaciones.
 *
 * Objetivo:
 *   Obtener estadísticas reproducibles para comparar:
 *
 *      V1 - Gaussiano 2D + Sobel
 *      V2 - Gaussiano separable + Sobel
 *      V3 - Kernels fusionados
 *
 * Las mediciones serán reutilizadas posteriormente en:
 *
 *      PC x86-64
 *      Raspberry Pi 5 ARM64
 *
 * Métricas:
 *   - media;
 *   - mediana;
 *   - desviación estándar;
 *   - mínimo;
 *   - máximo.
 * ============================================================
 */

#ifdef __cplusplus
extern "C" {
#endif


    /**
     * @brief Función genérica que puede ser ejecutada por benchmark.
     *
     * El parámetro context permite pasar cualquier estructura
     * necesaria para ejecutar una versión específica del algoritmo.
     */
    typedef void (*BenchmarkFunction)(void *context);


    /**
     * @brief Estadísticas obtenidas durante el benchmark.
     */
    typedef struct
    {
        double mean_ms;
        double median_ms;
        double stddev_ms;
        double min_ms;
        double max_ms;

    } BenchmarkStats;


    /**
     * @brief Ejecuta una función múltiples veces y mide su duración.
     *
     * Primero se realizan iteraciones de calentamiento que no se
     * incluyen en las estadísticas.
     *
     * @param function   Función que será evaluada.
     * @param context    Datos utilizados por la función.
     * @param warmups    Número de ejecuciones de calentamiento.
     * @param iterations Número de ejecuciones medidas.
     *
     * @return Estadísticas del tiempo de ejecución en milisegundos.
     */
    BenchmarkStats benchmark_run(
            BenchmarkFunction function,
            void *context,
            int warmups,
            int iterations);


#ifdef __cplusplus
}
#endif

#endif