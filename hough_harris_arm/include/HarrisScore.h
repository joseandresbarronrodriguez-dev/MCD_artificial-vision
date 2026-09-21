//
// Created by andru on 9/20/2026.
//
/**
 * ============================================================
 * TAREA 3 - HARRIS CORNER SCORE
 * ============================================================
 *
 * Archivo:
 *      HarrisScore.h
 *
 * Objetivo:
 *      Definir la clase encargada de calcular el score de
 *      Harris para cada píxel de una imagen.
 *
 * El detector de Harris utiliza la matriz:
 *
 *              [ Sxx  Sxy ]
 *          M = [          ]
 *              [ Sxy  Syy ]
 *
 * donde:
 *
 *      Sxx = suma de Ix^2
 *      Syy = suma de Iy^2
 *      Sxy = suma de Ix*Iy
 *
 * Ix e Iy representan los gradientes de la imagen.
 *
 * El score de Harris se calcula como:
 *
 *      R = det(M) - k * trace(M)^2
 *
 * o equivalentemente:
 *
 *      R =
 *      (Sxx*Syy - Sxy^2)
 *      -
 *      k*(Sxx + Syy)^2
 *
 * ============================================================
 */

#pragma once


// ============================================================
// DEPENDENCIAS
// ============================================================

// OpenCV será utilizado para recibir la imagen y calcular
// operaciones de preprocesamiento como los gradientes Sobel.
#include <opencv2/core.hpp>


// Eigen será utilizado para realizar operaciones matriciales
// sobre los mapas Sxx, Syy y Sxy.
#include <Eigen/Dense>


// ============================================================
// CLASE HARRIS SCORE
// ============================================================

/**
 * @brief Implementación propia del score de Harris.
 *
 * La responsabilidad de esta clase será:
 *
 *      imagen grayscale
 *              ↓
 *      gradientes Ix, Iy
 *              ↓
 *      productos Ix², Iy², IxIy
 *              ↓
 *      integración espacial
 *              ↓
 *      score Harris R
 *
 * NO realizará:
 *
 *      - lectura de imágenes
 *      - visualización
 *      - detección final de puntos
 *      - benchmark
 *
 * Esto permite medir únicamente el costo asociado
 * al cálculo del score.
 */
class HarrisScore
{

public:

    /**
     * @brief Constructor.
     *
     * @param blockSize Tamaño de la ventana utilizada para
     *                  acumular la información de gradientes.
     *
     * @param sobelKernel Tamaño del kernel utilizado para
     *                    calcular los gradientes con Sobel.
     *
     * @param k Parámetro empírico de Harris.
     *
     * Valores típicos:
     *
     *      k = 0.04 ... 0.06
     *
     * Para nuestro experimento utilizaremos:
     *
     *      k = 0.04
     */
    HarrisScore(
            int blockSize = 3,
            int sobelKernel = 3,
            float k = 0.04f
    );


    /**
     * @brief Calcula el mapa completo del score de Harris.
     *
     * @param gray Imagen grayscale de un solo canal.
     *
     * La imagen podrá ser:
     *
     *      CV_8UC1
     *
     * o:
     *
     *      CV_32FC1
     *
     * Internamente trabajaremos con valores float.
     *
     * @return cv::Mat de tipo CV_32FC1 con el score R
     *         calculado para cada píxel.
     */
    cv::Mat compute(
            const cv::Mat& gray
    ) const;


private:

    /**
     * Tamaño de la ventana utilizada para integrar
     * la información de gradientes.
     *
     * Ejemplo:
     *
     *      blockSize = 3
     *
     * corresponde a una ventana:
     *
     *      3 x 3
     */
    int blockSize_;


    /**
     * Tamaño del kernel de Sobel.
     *
     * Inicialmente utilizaremos:
     *
     *      3 x 3
     */
    int sobelKernel_;


    /**
     * Constante utilizada en la respuesta Harris:
     *
     *      R = det(M) - k * trace(M)^2
     */
    float k_;
};