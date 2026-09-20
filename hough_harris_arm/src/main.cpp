/**
 * ============================================================
 * TAREA 3 - HOUGH TRANSFORM + HARRIS CORNER SCORE
 * ============================================================
 *
 * Archivo:
 *      main.cpp
 *
 * Objetivo actual:
 *
 *      Verificar la integración de OpenCV y Eigen antes
 *      de implementar los algoritmos de Hough y Harris.
 *
 * Futuras etapas:
 *
 *      1. Acumulador de Hough usando Eigen.
 *      2. Comparación con cv::HoughLines().
 *      3. Score de Harris usando Eigen.
 *      4. Comparación con cv::cornerHarris().
 *      5. Benchmark PC vs ARM.
 *
 * ============================================================
 */

#include <iostream>

// OpenCV
#include <opencv2/core.hpp>

// Eigen
#include <Eigen/Dense>


int main()
{
    std::cout
            << "\n========================================\n"
            << " HOUGH + HARRIS ARM BENCHMARK\n"
            << "========================================\n";


    // ========================================================
    // PRUEBA 1 - OpenCV
    // ========================================================

    std::cout
            << "\n[OpenCV]\n"
            << "Version: "
            << CV_VERSION
            << "\n";


    // Creamos una matriz pequeña para comprobar que
    // OpenCV se encuentra correctamente enlazado.
    cv::Mat cvMatrix =
            cv::Mat::zeros(
                    3,
                    3,
                    CV_8UC1
            );


    std::cout
            << "Matriz de prueba: "
            << cvMatrix.rows
            << "x"
            << cvMatrix.cols
            << "\n";


    // ========================================================
    // PRUEBA 2 - Eigen
    // ========================================================

    std::cout
            << "\n[Eigen]\n"
            << "Version: "
            << EIGEN_WORLD_VERSION
            << "."
            << EIGEN_MAJOR_VERSION
            << "."
            << EIGEN_MINOR_VERSION
            << "\n";


    /*
     * MatrixXf:
     *
     * Matrix -> matriz
     * X      -> tamaño dinámico
     * f      -> float
     *
     * Más adelante utilizaremos estructuras similares para
     * almacenar matrices relacionadas con Harris.
     */
    Eigen::MatrixXf eigenMatrix =
            Eigen::MatrixXf::Identity(
                    3,
                    3
            );


    std::cout
            << "\nMatriz Eigen de prueba:\n"
            << eigenMatrix
            << "\n";


    // ========================================================
    // Resultado
    // ========================================================

    std::cout
            << "\n========================================\n"
            << " DEPENDENCIAS CONFIGURADAS CORRECTAMENTE\n"
            << "========================================\n";


    return 0;
}