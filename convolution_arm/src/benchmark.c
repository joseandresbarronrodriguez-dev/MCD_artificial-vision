//
// Created by andru on 9/7/2026.
//
/*
 * ============================================================
 * Archivo: benchmark.c
 *
 * Descripción:
 *   Implementa medición de tiempo de alta resolución para PC
 *   Windows y plataformas POSIX/ARM.
 *
 * Portabilidad:
 *
 *   Windows:
 *      QueryPerformanceCounter
 *
 *   Linux / Raspberry Pi:
 *      clock_gettime
 *
 * Esto permite utilizar el mismo código de benchmark en
 * x86-64 y ARM64.
 * ============================================================
 */

#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "benchmark.h"

#include <math.h>
#include <stdlib.h>


#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#else

#include <time.h>

#endif


/*
 * ============================================================
 * TEMPORIZADOR DE ALTA RESOLUCION
 * ============================================================
 */

/**
 * @brief Obtiene tiempo monotónico en segundos.
 */
static double benchmark_now_seconds(void)
{
#ifdef _WIN32

    static LARGE_INTEGER frequency;
    static int initialized = 0;

    LARGE_INTEGER counter;


    if (!initialized)
    {
        QueryPerformanceFrequency(&frequency);

        initialized = 1;
    }


    QueryPerformanceCounter(&counter);


    return
            (double) counter.QuadPart
            /
            (double) frequency.QuadPart;

#else

    struct timespec time_value;


#ifdef CLOCK_MONOTONIC_RAW

    clock_gettime(
            CLOCK_MONOTONIC_RAW,
            &time_value
    );

#else

    clock_gettime(
            CLOCK_MONOTONIC,
            &time_value
    );

#endif


    return
            (double) time_value.tv_sec
            +
            (double) time_value.tv_nsec
            *
            1e-9;

#endif
}


/*
 * Función auxiliar utilizada por qsort().
 */
static int compare_double(
        const void *a,
        const void *b)
{
    const double value_a =
            *(const double *) a;

    const double value_b =
            *(const double *) b;


    if (value_a < value_b)
    {
        return -1;
    }


    if (value_a > value_b)
    {
        return 1;
    }


    return 0;
}


/*
 * ============================================================
 * BENCHMARK PRINCIPAL
 * ============================================================
 */

BenchmarkStats benchmark_run(
        BenchmarkFunction function,
        void *context,
        int warmups,
        int iterations)
{
    BenchmarkStats statistics = {0};

    double *samples;

    double sum = 0.0;
    double sum_squared = 0.0;

    int i;


    if (function == NULL ||
        iterations <= 0)
    {
        return statistics;
    }


    if (warmups < 0)
    {
        warmups = 0;
    }


    /*
     * ========================================================
     * WARM-UP
     * ========================================================
     *
     * Estas ejecuciones permiten:
     *
     *   - calentar caches;
     *   - estabilizar frecuencia de CPU;
     *   - reducir ruido de la primera ejecución.
     *
     * NO forman parte de las estadísticas.
     */
    for (i = 0;
         i < warmups;
         ++i)
    {
        function(context);
    }


    /*
     * Reservamos memoria para almacenar todas las mediciones.
     */
    samples =
            (double *) malloc(
                    (size_t) iterations *
                    sizeof(double)
            );


    if (samples == NULL)
    {
        return statistics;
    }


    statistics.min_ms =
            1e300;

    statistics.max_ms =
            0.0;


    /*
     * ========================================================
     * ITERACIONES MEDIDAS
     * ========================================================
     */
    for (i = 0;
         i < iterations;
         ++i)
    {
        const double start =
                benchmark_now_seconds();


        function(context);


        const double end =
                benchmark_now_seconds();


        const double elapsed_ms =
                (end - start)
                *
                1000.0;


        samples[i] =
                elapsed_ms;


        sum +=
                elapsed_ms;


        sum_squared +=
                elapsed_ms *
                elapsed_ms;


        if (elapsed_ms <
            statistics.min_ms)
        {
            statistics.min_ms =
                    elapsed_ms;
        }


        if (elapsed_ms >
            statistics.max_ms)
        {
            statistics.max_ms =
                    elapsed_ms;
        }
    }


    /*
     * ========================================================
     * MEDIA
     * ========================================================
     */
    statistics.mean_ms =
            sum /
            (double) iterations;


    /*
     * ========================================================
     * DESVIACION ESTANDAR MUESTRAL
     * ========================================================
     */
    if (iterations > 1)
    {
        double variance =
                (
                    sum_squared
                    -
                    (double) iterations
                    *
                    statistics.mean_ms
                    *
                    statistics.mean_ms
                )
                /
                (double) (iterations - 1);


        /*
         * Protección contra pequeños errores numéricos.
         */
        if (variance < 0.0)
        {
            variance = 0.0;
        }


        statistics.stddev_ms =
                sqrt(variance);
    }


    /*
     * ========================================================
     * MEDIANA
     * ========================================================
     */
    qsort(
            samples,
            (size_t) iterations,
            sizeof(double),
            compare_double
    );


    if ((iterations % 2) == 0)
    {
        statistics.median_ms =
                (
                    samples[
                        iterations / 2 - 1
                    ]
                    +
                    samples[
                        iterations / 2
                    ]
                )
                /
                2.0;
    }
    else
    {
        statistics.median_ms =
                samples[
                    iterations / 2
                ];
    }


    free(samples);


    return statistics;
}