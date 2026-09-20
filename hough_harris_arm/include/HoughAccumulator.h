//
// Created by andru on 9/20/2026.
//
/**
 * ============================================================
 * TAREA 3 - TRANSFORMADA DE HOUGH
 * ============================================================
 *
 * Archivo:
 *      HoughAccumulator.h
 *
 * Objetivo:
 *      Definir la estructura encargada de construir el
 *      acumulador de la Transformada de Hough para detección
 *      de líneas.
 *
 * La Transformada de Hough utiliza la ecuación:
 *
 *      rho = x*cos(theta) + y*sin(theta)
 *
 * donde:
 *
 *      x, y  -> coordenadas de un píxel de borde
 *      theta -> ángulo de la línea
 *      rho   -> distancia de la línea al origen
 *
 * Cada píxel de borde "vota" por diferentes combinaciones
 * (rho, theta).
 *
 * Los votos se almacenarán en una matriz Eigen:
 *
 *      Eigen::MatrixXi accumulator
 *
 * donde:
 *
 *      filas    -> valores discretizados de rho
 *      columnas -> valores discretizados de theta
 *
 * ============================================================
 */

#pragma once


// ============================================================
// DEPENDENCIAS
// ============================================================

// OpenCV será utilizado para recibir la imagen de bordes.
//
// Esperamos una imagen binaria de un solo canal, normalmente
// obtenida mediante cv::Canny().
#include <opencv2/core.hpp>


// Eigen será utilizado para almacenar el acumulador.
//
// MatrixXi:
//      Matrix -> matriz
//      X      -> tamaño dinámico
//      i      -> integer
#include <Eigen/Dense>


// ============================================================
// RESULTADO DEL ACUMULADOR
// ============================================================

/**
 * @brief Contiene el resultado de la Transformada de Hough.
 *
 * Además de la matriz de votos necesitamos conocer algunos
 * parámetros para poder interpretar posteriormente los índices
 * de la matriz como valores físicos de rho y theta.
 */
struct HoughResult
{
    /**
     * Matriz de votos.
     *
     * Filas:
     *      rho
     *
     * Columnas:
     *      theta
     */
    Eigen::MatrixXi accumulator;


    /**
     * Valor máximo considerado para rho.
     *
     * rho puede ser positivo o negativo:
     *
     *      -rhoMax <= rho <= rhoMax
     *
     * Como una matriz no puede tener índices negativos,
     * posteriormente utilizaremos un desplazamiento:
     *
     *      rhoIndex = rho + rhoMax
     */
    int rhoMax = 0;


    /**
     * Número de divisiones utilizadas para theta.
     *
     * Inicialmente utilizaremos:
     *
     *      180 bins
     *
     * equivalente aproximadamente a:
     *
     *      1 grado por columna
     */
    int thetaBins = 180;
};


// ============================================================
// CLASE HOUGH ACCUMULATOR
// ============================================================

/**
 * @brief Implementación propia del acumulador de Hough.
 *
 * La responsabilidad de esta clase será exclusivamente:
 *
 *      imagen de bordes
 *              ↓
 *      votación de Hough
 *              ↓
 *      matriz acumuladora
 *
 * NO realizará:
 *
 *      - lectura de imágenes
 *      - Canny
 *      - dibujo de líneas
 *      - benchmark
 *
 * Esto mantiene cada etapa separada para poder medir
 * correctamente el costo computacional del algoritmo.
 */
class HoughAccumulator
{

public:

    /**
     * @brief Constructor.
     *
     * @param thetaBins Número de divisiones del espacio angular.
     *
     * Usaremos por defecto 180:
     *
     *      theta = 0° ... 179°
     */
    explicit HoughAccumulator(
            int thetaBins = 180
    );


    /**
     * @brief Construye el acumulador de Hough.
     *
     * @param edges Imagen binaria de bordes.
     *
     * Formato esperado:
     *
     *      CV_8UC1
     *
     * donde:
     *
     *      0   -> no existe borde
     *      255 -> píxel de borde
     *
     * @return HoughResult con la matriz acumuladora y
     *         parámetros necesarios para interpretarla.
     */
    HoughResult compute(
            const cv::Mat& edges
    ) const;


private:

    /**
     * Número de valores discretos de theta.
     */
    int thetaBins_;


    /**
     * Tabla precalculada de cos(theta).
     *
     * Evitaremos calcular:
     *
     *      cos(theta)
     *
     * para cada píxel y cada iteración.
     *
     * Esto será especialmente importante en ARM.
     */
    Eigen::VectorXf cosTable_;


    /**
     * Tabla precalculada de sin(theta).
     *
     * Igual que cosTable_, se calcula una sola vez en el
     * constructor.
     */
    Eigen::VectorXf sinTable_;
};