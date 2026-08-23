/*
 * Práctica debuyering
 * Visión artificial
 * Andrés Barrón
 * 2026-08-23
 */
#include <iostream>

#include <opencv2/opencv.hpp>
#include <libraw/libraw.h>

int main()
{
    std::cout << "======================================\n";
    std::cout << " Practica: Debayering RAW -> RGB\n";
    std::cout << "======================================\n\n";

    // OpenCV
    std::cout << "OpenCV version: "
              << CV_VERSION
              << '\n';

    cv::Mat image = cv::Mat::zeros(
        480,
        640,
        CV_8UC3
    );

    std::cout << "OpenCV funcionando correctamente\n";

    // LibRaw
    LibRaw rawProcessor;

    std::cout << "LibRaw inicializado correctamente\n";

    std::cout << "\nEntorno listo para procesar RAW.\n";

    return 0;
}