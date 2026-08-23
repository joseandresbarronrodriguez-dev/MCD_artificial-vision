/*
 * Práctica debuyering
 * Visión artificial
 * Andrés Barrón
 * 2026-08-23
 */
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    std::cout << "======================================" << std::endl;
    std::cout << " Practica: Debayering RAW -> RGB" << std::endl;
    std::cout << "======================================" << std::endl;

    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);

    std::cout << "Imagen creada correctamente" << std::endl;
    std::cout << "Resolucion: "
              << image.cols << " x "
              << image.rows << std::endl;

    std::cout << "Canales: "
              << image.channels() << std::endl;

    return 0;
}