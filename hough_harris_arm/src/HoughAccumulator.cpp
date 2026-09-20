/**
 * ============================================================
 * TAREA 3 - TRANSFORMADA DE HOUGH
 * ============================================================
 *
 * Archivo:
 *      HoughAccumulator.cpp
 *
 * Objetivo de esta etapa:
 *
 *      Implementar la inicialización del acumulador de Hough.
 *
 * En particular:
 *
 *      1. Validar el número de bins de theta.
 *      2. Crear las tablas de seno y coseno.
 *      3. Precalcular los valores trigonométricos.
 *
 * La votación del acumulador se implementará en el siguiente
 * paso dentro del método compute().
 *
 * ============================================================
 */


#include "HoughAccumulator.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>


// ============================================================
// CONSTRUCTOR
// ============================================================

HoughAccumulator::HoughAccumulator(
        int thetaBins
)
        : thetaBins_(thetaBins)
{
    // --------------------------------------------------------
    // Validación
    // --------------------------------------------------------
    //
    // El espacio angular debe tener al menos una división.
    //
    // Para nuestro experimento utilizaremos normalmente:
    //
    //      thetaBins = 180
    //
    // lo que representa aproximadamente un grado por bin.
    // --------------------------------------------------------

    if (thetaBins_ <= 0)
    {
        throw std::invalid_argument(
                "thetaBins debe ser mayor que cero."
        );
    }


    // --------------------------------------------------------
    // Reservar memoria para las tablas trigonométricas
    // --------------------------------------------------------
    //
    // Eigen::VectorXf es un vector dinámico de valores float.
    //
    // Tendremos:
    //
    //      cosTable_[theta]
    //      sinTable_[theta]
    //
    // para evitar ejecutar std::cos() y std::sin() durante
    // cada voto del acumulador.
    // --------------------------------------------------------

    cosTable_.resize(thetaBins_);
    sinTable_.resize(thetaBins_);


    // --------------------------------------------------------
    // Precalcular seno y coseno
    // --------------------------------------------------------
    //
    // La Transformada de Hough para líneas utiliza:
    //
    //      rho = x*cos(theta) + y*sin(theta)
    //
    // Trabajamos en el intervalo:
    //
    //      0 <= theta < PI
    //
    // equivalente a:
    //
    //      0° <= theta < 180°
    //
    // No necesitamos llegar a 360°, ya que una línea puede
    // representarse completamente usando rho positivo y
    // negativo en un intervalo angular de 180 grados.
    // --------------------------------------------------------

    for (int thetaIndex = 0;
         thetaIndex < thetaBins_;
         ++thetaIndex)
    {
        // Convertimos el índice discreto a un ángulo real.
        //
        // Para 180 bins:
        //
        //      thetaIndex = 0   ->   0°
        //      thetaIndex = 90  ->  90°
        //      thetaIndex = 179 -> 179°
        //
        constexpr float PI =
        3.14159265358979323846f;

        const float theta =
                PI
                * static_cast<float>(thetaIndex)
                / static_cast<float>(thetaBins_);


        // Guardamos los resultados para reutilizarlos
        // posteriormente durante la votación.
        cosTable_(thetaIndex) = std::cos(theta);

        sinTable_(thetaIndex) = std::sin(theta);
    }
}

// ============================================================
// CONSTRUCCIÓN DEL ACUMULADOR DE HOUGH
// ============================================================

HoughResult HoughAccumulator::compute(
        const cv::Mat& edges
) const
{
    // --------------------------------------------------------
    // 1. Validar la imagen de entrada
    // --------------------------------------------------------
    //
    // El acumulador espera una imagen binaria proveniente,
    // normalmente, de cv::Canny().
    //
    // Formato esperado:
    //
    //      CV_8UC1
    //
    // donde:
    //
    //      0   -> no existe borde
    //      255 -> píxel de borde
    //
    // --------------------------------------------------------

    if (edges.empty())
    {
        throw std::invalid_argument(
                "La imagen de bordes esta vacia."
        );
    }


    if (edges.type() != CV_8UC1)
    {
        throw std::invalid_argument(
                "HoughAccumulator requiere una imagen CV_8UC1."
        );
    }


    // --------------------------------------------------------
    // 2. Calcular el rango máximo de rho
    // --------------------------------------------------------
    //
    // La ecuación de Hough es:
    //
    //      rho = x*cos(theta) + y*sin(theta)
    //
    // El valor máximo posible de rho está limitado por
    // aproximadamente la diagonal de la imagen:
    //
    //      rhoMax = sqrt(width^2 + height^2)
    //
    // rho puede ser:
    //
    //      -rhoMax ... 0 ... +rhoMax
    //
    // --------------------------------------------------------

    const int rhoMax =
            static_cast<int>(
                    std::ceil(
                            std::hypot(
                                    static_cast<double>(edges.cols),
                                    static_cast<double>(edges.rows)
                            )
                    )
            );


    // --------------------------------------------------------
    // 3. Número total de bins para rho
    // --------------------------------------------------------
    //
    // Necesitamos representar:
    //
    //      -rhoMax ... +rhoMax
    //
    // Ejemplo:
    //
    //      rhoMax = 3
    //
    // valores:
    //
    //      -3 -2 -1 0 1 2 3
    //
    // total:
    //
    //      2 * rhoMax + 1
    //
    // --------------------------------------------------------

    const int rhoBins =
            2 * rhoMax + 1;


    // --------------------------------------------------------
    // 4. Crear el acumulador usando Eigen
    // --------------------------------------------------------
    //
    // Dimensiones:
    //
    //      filas    -> rho
    //      columnas -> theta
    //
    // MatrixXi significa una matriz dinámica de enteros.
    //
    // Inicialmente todos los votos son cero.
    // --------------------------------------------------------

    Eigen::MatrixXi accumulator =
            Eigen::MatrixXi::Zero(
                    rhoBins,
                    thetaBins_
            );


    // --------------------------------------------------------
    // 5. Obtener solamente los píxeles de borde
    // --------------------------------------------------------
    //
    // En lugar de recorrer TODOS los píxeles de la imagen
    // dentro del algoritmo de Hough, usamos OpenCV para obtener
    // únicamente las posiciones donde existe un borde.
    //
    // Esto reduce el trabajo innecesario y mantiene el foco
    // del benchmark en la votación de Hough.
    // --------------------------------------------------------

    std::vector<cv::Point> edgePoints;

    cv::findNonZero(
            edges,
            edgePoints
    );


    // --------------------------------------------------------
    // 6. Votación de Hough
    // --------------------------------------------------------
    //
    // Cada píxel de borde votará para todos los valores
    // discretos de theta.
    //
    // Para cada punto:
    //
    //      (x, y)
    //
    // calculamos:
    //
    //      rho = x*cos(theta) + y*sin(theta)
    //
    // --------------------------------------------------------

    for (const cv::Point& point : edgePoints)
    {
        const float x =
                static_cast<float>(point.x);

        const float y =
                static_cast<float>(point.y);


        // ----------------------------------------------------
        // Recorrer el espacio angular
        // ----------------------------------------------------

        for (int thetaIndex = 0;
             thetaIndex < thetaBins_;
             ++thetaIndex)
        {
            // Utilizamos las tablas previamente calculadas
            // en el constructor.
            //
            // Esto evita ejecutar std::sin() y std::cos()
            // millones de veces.
            const float rho =
                    x * cosTable_(thetaIndex)
                    +
                    y * sinTable_(thetaIndex);


            // ------------------------------------------------
            // Convertir rho real a índice de matriz
            // ------------------------------------------------
            //
            // rho puede ser negativo:
            //
            //      -rhoMax ... +rhoMax
            //
            // Eigen requiere índices:
            //
            //      0 ... rhoBins-1
            //
            // Por eso desplazamos:
            //
            //      rhoIndex = rho + rhoMax
            //
            // Ejemplo:
            //
            //      rho = -10
            //      rhoMax = 500
            //
            //      rhoIndex = 490
            //
            // ------------------------------------------------

            const int rhoIndex =
                    static_cast<int>(
                            std::lround(rho)
                    )
                    +
                    rhoMax;


            // ------------------------------------------------
            // Validación de seguridad
            // ------------------------------------------------
            //
            // Matemáticamente debería encontrarse siempre
            // dentro del rango, pero mantenemos la validación
            // para evitar accesos inválidos por redondeo.
            // ------------------------------------------------

            if (rhoIndex >= 0 &&
                rhoIndex < rhoBins)
            {
                // Incrementamos el voto correspondiente.
                accumulator(
                        rhoIndex,
                        thetaIndex
                )++;
            }
        }
    }


    // --------------------------------------------------------
    // 7. Construir el resultado
    // --------------------------------------------------------

    HoughResult result;

    result.accumulator =
            std::move(accumulator);

    result.rhoMax =
            rhoMax;

    result.thetaBins =
            thetaBins_;


    return result;
}