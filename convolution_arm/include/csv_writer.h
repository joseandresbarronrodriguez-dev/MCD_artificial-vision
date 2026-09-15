//
// Created by andru on 9/14/2026.
//
#ifndef CSV_WRITER_H
#define CSV_WRITER_H

#include "benchmark.h"
#include "energy.h"

#ifdef __cplusplus
extern "C" {
#endif


    /**
     * Guarda una fila con los resultados del benchmark.
     *
     * Si el archivo no existe, crea también el encabezado.
     */
    int benchmark_csv_append(
        const char *filename,
        const char *platform,
        const char *build_type,
        const char *benchmark_type,
        int width,
        int height,
        int kernel_size,
        double sigma,
        const char *version,
        BenchmarkStats stats,
        double speedup,
        EnergyMetrics energy);


#ifdef __cplusplus
}
#endif

#endif