//
// Created by andru on 9/5/2026.
//
#include "metrics.h"

#include <math.h>
#include <stddef.h>

/*
 * ============================================================
 * Archivo: metrics.c
 *
 * Descripción:
 *   Implementa métricas de error para comparar resultados de
 *   filtrado.
 *
 * Métricas:
 *   - MAE
 *   - RMSE
 *   - Error absoluto máximo
 * ============================================================
 */


ErrorMetrics compare_images_f32(
        const ImageF32 *reference,
        const ImageF32 *candidate,
        int border)
{
    ErrorMetrics result = {0};

    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    double max_error = 0.0;

    size_t count = 0;

    int x;
    int y;


    /*
     * Validación de entrada.
     */
    if (reference == NULL ||
        candidate == NULL ||
        reference->data == NULL ||
        candidate->data == NULL)
    {
        return result;
    }


    /*
     * Ambas imágenes deben tener las mismas dimensiones.
     */
    if (reference->width != candidate->width ||
        reference->height != candidate->height)
    {
        return result;
    }


    if (border < 0)
    {
        border = 0;
    }


    /*
     * Recorremos solamente la región válida.
     *
     * Esto evita que las políticas de borde afecten la
     * comparación.
     */
    for (y = border;
         y < reference->height - border;
         ++y)
    {
        for (x = border;
             x < reference->width - border;
             ++x)
        {
            const size_t index =
                    (size_t) y *
                    reference->width +
                    x;


            const double difference =
                    (double) reference->data[index] -
                    (double) candidate->data[index];


            const double absolute_error =
                    fabs(difference);


            absolute_sum += absolute_error;

            squared_sum +=
                    difference *
                    difference;


            if (absolute_error > max_error)
            {
                max_error = absolute_error;
            }


            ++count;
        }
    }


    /*
     * Calculamos métricas finales.
     */
    if (count > 0)
    {
        result.mae =
                absolute_sum /
                (double) count;


        result.rmse =
                sqrt(
                    squared_sum /
                    (double) count
                );


        result.max_abs_error =
                max_error;
    }


    return result;
}