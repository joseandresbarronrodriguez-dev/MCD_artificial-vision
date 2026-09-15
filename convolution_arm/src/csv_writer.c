//
// Created by andru on 9/14/2026.
//
#include "csv_writer.h"

#include <stdio.h>


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
        EnergyMetrics energy)
{
    FILE *checkFile;
    FILE *file;

    int fileExists = 0;


    /*
     * Revisamos si el archivo ya existe.
     */
    checkFile =
            fopen(filename, "r");


    if (checkFile != NULL)
    {
        fileExists = 1;

        fclose(checkFile);
    }


    /*
     * Abrimos en modo append.
     */
    file =
            fopen(filename, "a");


    if (file == NULL)
    {
        return 0;
    }


    /*
     * Si es un archivo nuevo, escribimos encabezado.
     */
    if (!fileExists)
    {
        fprintf(
                file,
                "platform,"
                "build,"
                "benchmark_type,"
                "width,"
                "height,"
                "pixels,"
                "kernel,"
                "sigma,"
                "version,"
                "mean_ms,"
                "median_ms,"
                "stddev_ms,"
                "min_ms,"
                "max_ms,"
                "speedup,"
                "power_w,"
                "energy_frame_j,"
                "energy_pixel_uj,"
                "pixels_per_joule\n"
        );
    }


    const long long pixels =
            (long long) width *
            (long long) height;


    fprintf(
            file,
            "%s,"
            "%s,"
            "%s,"
            "%d,"
            "%d,"
            "%lld,"
            "%d,"
            "%.6f,"
            "%s,"
            "%.9f,"
            "%.9f,"
            "%.9f,"
            "%.9f,"
            "%.9f,"
            "%.9f,"
            "%.9f,"
            "%.12f,"
            "%.12f,"
            "%.6f\n",

            platform,
            build_type,
            benchmark_type,

            width,
            height,

            pixels,

            kernel_size,
            sigma,

            version,

            stats.mean_ms,
            stats.median_ms,
            stats.stddev_ms,
            stats.min_ms,
            stats.max_ms,

            speedup,

            energy.average_power_w,
            energy.energy_frame_j,
            energy.energy_pixel_uj,
            energy.pixels_per_joule
    );


    fclose(file);


    return 1;
}