//
// Created by andru on 9/20/2026.
//
/**
 * ============================================================
 * TAREA 3 - HARRIS CORNER SCORE
 * ============================================================
 *
 * Archivo:
 *      HarrisScore.cpp
 *
 * Objetivo de esta etapa:
 *
 *      Implementar el constructor de HarrisScore y validar
 *      los parámetros utilizados por el algoritmo.
 *
 * Todavía NO se calcula:
 *
 *      - gradiente Ix
 *      - gradiente Iy
 *      - matriz de estructura
 *      - score R
 *
 * Esas operaciones se implementarán posteriormente dentro
 * del método compute().
 *
 * ============================================================
 */


#include "HarrisScore.h"

#include <stdexcept>
// OpenCV - procesamiento de imagen
#include <opencv2/imgproc.hpp>

// ============================================================
// CONSTRUCTOR
// ============================================================

HarrisScore::HarrisScore(
        int blockSize,
        int sobelKernel,
        float k
)
        :
        blockSize_(blockSize),
        sobelKernel_(sobelKernel),
        k_(k)
{
    // --------------------------------------------------------
    // 1. Validar blockSize
    // --------------------------------------------------------
    //
    // blockSize determina el tamaño de la ventana local
    // utilizada para acumular:
    //
    //      Ix^2
    //      Iy^2
    //      Ix*Iy
    //
    // Utilizaremos normalmente:
    //
    //      blockSize = 3
    //
    // --------------------------------------------------------

    if (blockSize_ < 2)
    {
        throw std::invalid_argument(
                "blockSize debe ser mayor o igual a 2."
        );
    }


    // --------------------------------------------------------
    // 2. Validar kernel de Sobel
    // --------------------------------------------------------
    //
    // Para esta práctica permitiremos:
    //
    //      3 x 3
    //      5 x 5
    //      7 x 7
    //
    // Inicialmente utilizaremos 3 x 3 para reducir costo
    // computacional y mantener una comparación reproducible.
    // --------------------------------------------------------

    if (sobelKernel_ != 3 &&
        sobelKernel_ != 5 &&
        sobelKernel_ != 7)
    {
        throw std::invalid_argument(
                "sobelKernel debe ser 3, 5 o 7."
        );
    }


    // --------------------------------------------------------
    // 3. Validar parámetro k
    // --------------------------------------------------------
    //
    // Harris utiliza:
    //
    //      R = det(M) - k * trace(M)^2
    //
    // Normalmente:
    //
    //      k = 0.04 ... 0.06
    //
    // Permitimos un rango ligeramente mayor para facilitar
    // experimentación posterior.
    // --------------------------------------------------------

    if (k_ <= 0.0f ||
        k_ >= 0.25f)
    {
        throw std::invalid_argument(
                "El parametro k debe estar entre 0 y 0.25."
        );
    }
}

// ============================================================
// CÁLCULO DEL SCORE DE HARRIS
// ============================================================

cv::Mat HarrisScore::compute(
        const cv::Mat& gray
) const
{
    // --------------------------------------------------------
    // 1. Validar imagen
    // --------------------------------------------------------

    if (gray.empty())
    {
        throw std::invalid_argument(
                "La imagen de entrada esta vacia."
        );
    }

    if (gray.channels() != 1)
    {
        throw std::invalid_argument(
                "HarrisScore requiere una imagen grayscale."
        );
    }


    // --------------------------------------------------------
    // 2. Convertir imagen a float
    // --------------------------------------------------------
    //
    // Harris trabaja con derivadas y productos que pueden
    // superar fácilmente el rango de uint8.
    //
    // Por eso convertimos:
    //
    //      CV_8U  -> CV_32F
    //
    // y normalizamos:
    //
    //      0 ... 255
    //
    // a:
    //
    //      0 ... 1
    //
    // Si la imagen ya es CV_32F, no repetimos la conversión.
    // --------------------------------------------------------

    cv::Mat gray32f;

    if (gray.type() == CV_8UC1)
    {
        gray.convertTo(
                gray32f,
                CV_32F,
                1.0 / 255.0
        );
    }
    else if (gray.type() == CV_32FC1)
    {
        gray32f = gray;
    }
    else
    {
        throw std::invalid_argument(
                "HarrisScore acepta CV_8UC1 o CV_32FC1."
        );
    }


    // ========================================================
    // 3. CALCULAR GRADIENTES
    // ========================================================
    //
    // Sobel aproxima:
    //
    //      Ix = dI/dx
    //      Iy = dI/dy
    //
    // Estos gradientes nos indican cómo cambia la intensidad
    // de la imagen horizontal y verticalmente.
    // ========================================================

    cv::Mat ix;
    cv::Mat iy;


    cv::Sobel(
            gray32f,
            ix,
            CV_32F,

            // dx
            1,

            // dy
            0,

            sobelKernel_
    );


    cv::Sobel(
            gray32f,
            iy,
            CV_32F,

            // dx
            0,

            // dy
            1,

            sobelKernel_
    );


    // ========================================================
    // 4. PRODUCTOS DE LOS GRADIENTES
    // ========================================================
    //
    // Construimos los tres términos necesarios para la
    // matriz de estructura:
    //
    //      Ix²
    //      Iy²
    //      IxIy
    //
    // ========================================================

    cv::Mat ix2 =
            ix.mul(ix);

    cv::Mat iy2 =
            iy.mul(iy);

    cv::Mat ixy =
            ix.mul(iy);


    // ========================================================
    // 5. INTEGRACIÓN ESPACIAL
    // ========================================================
    //
    // Harris no analiza un píxel aislado.
    //
    // Integra la información de una región local:
    //
    //      Sxx = suma Ix²
    //      Syy = suma Iy²
    //      Sxy = suma IxIy
    //
    // Utilizamos boxFilter porque realiza eficientemente
    // esta acumulación sobre una ventana.
    //
    // normalize = false
    //
    // porque necesitamos la SUMA, no el promedio.
    // ========================================================

    cv::Mat sxx;
    cv::Mat syy;
    cv::Mat sxy;


    cv::boxFilter(
            ix2,
            sxx,
            CV_32F,
            cv::Size(
                    blockSize_,
                    blockSize_
            ),
            cv::Point(-1, -1),

            // normalize
            false
    );


    cv::boxFilter(
            iy2,
            syy,
            CV_32F,
            cv::Size(
                    blockSize_,
                    blockSize_
            ),
            cv::Point(-1, -1),
            false
    );


    cv::boxFilter(
            ixy,
            sxy,
            CV_32F,
            cv::Size(
                    blockSize_,
                    blockSize_
            ),
            cv::Point(-1, -1),
            false
    );


    // ========================================================
    // 6. PREPARAR MAPA DE RESPUESTA
    // ========================================================

    cv::Mat response(
            gray.rows,
            gray.cols,
            CV_32FC1
    );


    // --------------------------------------------------------
    // Eigen::Map requiere memoria continua para utilizar esta
    // representación sencilla.
    //
    // Normalmente estas matrices ya son continuas, pero lo
    // comprobamos para mantener el código robusto.
    // --------------------------------------------------------

    if (!sxx.isContinuous())
    {
        sxx = sxx.clone();
    }

    if (!syy.isContinuous())
    {
        syy = syy.clone();
    }

    if (!sxy.isContinuous())
    {
        sxy = sxy.clone();
    }

    if (!response.isContinuous())
    {
        response = response.clone();
    }


    // ========================================================
    // 7. MAPEAR OpenCV -> Eigen
    // ========================================================
    //
    // OpenCV almacena las imágenes por filas.
    //
    // Por ello utilizamos RowMajor para que Eigen interprete
    // la memoria exactamente en el mismo orden.
    // ========================================================

    using EigenArray =
            Eigen::Array<
                    float,
                    Eigen::Dynamic,
                    Eigen::Dynamic,
                    Eigen::RowMajor
            >;


    Eigen::Map<const EigenArray> eigenSxx(
            sxx.ptr<float>(),
            sxx.rows,
            sxx.cols
    );


    Eigen::Map<const EigenArray> eigenSyy(
            syy.ptr<float>(),
            syy.rows,
            syy.cols
    );


    Eigen::Map<const EigenArray> eigenSxy(
            sxy.ptr<float>(),
            sxy.rows,
            sxy.cols
    );


    Eigen::Map<EigenArray> eigenResponse(
            response.ptr<float>(),
            response.rows,
            response.cols
    );


    // ========================================================
    // 8. CALCULAR SCORE DE HARRIS CON EIGEN
    // ========================================================
    //
    // Para:
    //
    //          [ Sxx  Sxy ]
    //      M = [          ]
    //          [ Sxy  Syy ]
    //
    // tenemos:
    //
    //      det(M) =
    //      Sxx*Syy - Sxy²
    //
    // y:
    //
    //      trace(M) =
    //      Sxx + Syy
    //
    // Harris:
    //
    //      R =
    //      det(M)
    //      -
    //      k * trace(M)²
    //
    // Eigen ejecuta esta operación sobre el mapa completo,
    // evitando escribir nosotros un doble for por píxel.
    // ========================================================

    const auto determinant =
            eigenSxx * eigenSyy
            -
            eigenSxy.square();


    const auto trace =
            eigenSxx
            +
            eigenSyy;


    eigenResponse =
            determinant
            -
            k_ * trace.square();


    // ========================================================
    // 9. DEVOLVER MAPA DE SCORE
    // ========================================================

    return response;
}