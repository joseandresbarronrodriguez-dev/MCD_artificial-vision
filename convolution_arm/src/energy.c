//
// Created by andru on 9/14/2026.
//
#include "energy.h"


EnergyMetrics energy_calculate(
        double power_w,
        double time_ms,
        long long total_pixels)
{
    EnergyMetrics result = {0};

    if (power_w <= 0.0 ||
        time_ms <= 0.0 ||
        total_pixels <= 0)
    {
        return result;
    }


    /*
     * Convertimos milisegundos a segundos.
     */
    const double time_seconds =
            time_ms / 1000.0;


    /*
     * Energía:
     *
     *      E = P * t
     */
    result.average_power_w =
            power_w;


    result.energy_frame_j =
            power_w *
            time_seconds;


    /*
     * Joules por pixel -> microjoules por pixel
     */
    result.energy_pixel_uj =
            (
                result.energy_frame_j
                /
                (double) total_pixels
            )
            *
            1e6;


    /*
     * Eficiencia energética.
     */
    result.pixels_per_joule =
            (double) total_pixels
            /
            result.energy_frame_j;


    return result;
}