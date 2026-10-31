//
// Created by andru on 9/14/2026.
//
#ifndef ENERGY_H
#define ENERGY_H

#ifdef __cplusplus
extern "C" {
#endif

    /*
     * ============================================================
     * Métricas energéticas
     * ============================================================
     *
     * average_power_w:
     *      Potencia promedio durante el procesamiento [W].
     *
     * energy_frame_j:
     *      Energía utilizada para procesar un frame [J].
     *
     * energy_pixel_uj:
     *      Energía utilizada por píxel [microjoules/pixel].
     *
     * pixels_per_joule:
     *      Cantidad de píxeles procesados por joule.
     */

    typedef struct
    {
        double average_power_w;

        double energy_frame_j;

        double energy_pixel_uj;

        double pixels_per_joule;

    } EnergyMetrics;


    /**
     * Calcula métricas energéticas a partir de:
     *
     * power_w        -> potencia promedio [W]
     * time_ms        -> tiempo de procesamiento [ms]
     * total_pixels   -> número de píxeles
     */
    EnergyMetrics energy_calculate(
            double power_w,
            double time_ms,
            long long total_pixels);


#ifdef __cplusplus
}
#endif

#endif