//
// Created by andru on 9/5/2026.
//
#ifndef METRICS_H
#define METRICS_H

#include "convolution.h"

/*
 * ============================================================
 * Archivo: metrics.h
 *
 * Descripción:
 *   Define métricas numéricas para comparar resultados de
 *   distintas implementaciones de convolución.
 *
 * Objetivo:
 *   Verificar que una optimización mantiene un resultado
 *   numéricamente equivalente a la implementación de referencia.
 * ============================================================
 */

typedef struct
{
    double mae;
    double rmse;
    double max_abs_error;

} ErrorMetrics;


#ifdef __cplusplus
extern "C" {
#endif


    /**
     * @brief Compara dos imágenes ImageF32.
     *
     * Se evalúan únicamente los píxeles internos, excluyendo un
     * borde configurable.
     *
     * @param reference Imagen de referencia.
     * @param candidate Imagen a comparar.
     * @param border    Número de píxeles ignorados en cada borde.
     *
     * @return Estructura con MAE, RMSE y error máximo.
     */
    ErrorMetrics compare_images_f32(
            const ImageF32 *reference,
            const ImageF32 *candidate,
            int border);


#ifdef __cplusplus
}
#endif

#endif