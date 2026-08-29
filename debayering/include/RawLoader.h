#ifndef RAW_LOADER_HPP
#define RAW_LOADER_HPP

#include <opencv2/core.hpp>

#include <string>

struct RawImageData
{
    // Mosaico Bayer original de la región visible.
    // 1 canal, 16 bits.
    cv::Mat rawImage;

    // Información de la cámara.
    std::string cameraMake;
    std::string cameraModel;

    // Información del CFA/Bayer.
    std::string bayerPattern;

    // Dimensiones del sensor RAW completo.
    int rawWidth = 0;
    int rawHeight = 0;

    // Dimensiones del área visible.
    int width = 0;
    int height = 0;

    // Márgenes existentes dentro del RAW.
    int topMargin = 0;
    int leftMargin = 0;

    // Valor máximo reportado por LibRaw.
    unsigned int maximum = 0;
    unsigned int blackLevel = 0;

    // Balance de blancos de cámara.
    float whiteBalanceR = 1.0f;
    float whiteBalanceG = 1.0f;
    float whiteBalanceB = 1.0f;

    // Matriz de corrección de color.
    cv::Matx33f cameraToRgb = cv::Matx33f::eye();

    // Indica si la matriz de corrección fue obtenida.
    bool hasColorMatrix = false;

    int orientation = 0;
};

class RawLoader
{
public:
    static RawImageData load(const std::string& filePath);

private:
    static std::string detectBayerPattern(class LibRaw& processor);
};

#endif //DEBAYERING_RAWLOADER_H