/*
 * ============================================================
 * Archivo: main.cpp
 * Proyecto: Convolución separable y fusión de kernels en ARM
 *
 * Objetivo:
 *   Comparar experimentalmente tres implementaciones de un
 *   detector de bordes basado en filtro Gaussiano + Sobel.
 *
 * Versiones:
 *
 *   V1 - Directa
 *        Gaussiano 2D
 *            ↓
 *        Sobel X / Sobel Y
 *            ↓
 *        Magnitud
 *
 *   V2 - Separable
 *        Gaussiano 1D horizontal
 *            ↓
 *        Gaussiano 1D vertical
 *            ↓
 *        Sobel X / Sobel Y
 *            ↓
 *        Magnitud
 *
 *   V3 - Fusionada
 *        Kernel Gaussiano + Sobel X
 *        Kernel Gaussiano + Sobel Y
 *            ↓
 *        Magnitud
 *
 * Benchmark:
 *
 *   A) Pipeline completo
 *      Incluye reserva/liberación de buffers.
 *
 *   B) Kernel-only
 *      Buffers preasignados.
 *
 * Parámetros dinámicos:
 *
 *   - Tamaño de kernel seleccionado por el usuario.
 *   - Sigma automático:
 *
 *          sigma = kernelSize / 6
 *
 * Energía:
 *   Pendiente para etapa posterior en ARM.
 *
 * Plataforma actual:
 *   PC x86-64 / Windows / MSVC / Release
 *
 * Plataforma objetivo:
 *   Raspberry Pi 5 / ARM64 / Cortex-A76
 * ============================================================
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <limits>

#include <opencv2/opencv.hpp>

#include "image.h"
#include "kernels.h"
#include "convolution.h"
#include "metrics.h"
#include "benchmark.h"


/*
 * ============================================================
 * CONTEXTOS PIPELINE COMPLETO
 * ============================================================
 */

struct BenchmarkContextV1
{
    const ImageU8 *input;

    const Kernel2D *gaussian;

    const Kernel2D *sobelX;
    const Kernel2D *sobelY;
};


struct BenchmarkContextV2
{
    const ImageU8 *input;

    const Kernel1D *gaussian;

    const Kernel2D *sobelX;
    const Kernel2D *sobelY;
};


struct BenchmarkContextV3
{
    const ImageU8 *input;

    const Kernel2D *fusedX;
    const Kernel2D *fusedY;
};


/*
 * ============================================================
 * CONTEXTOS KERNEL-ONLY
 * ============================================================
 */

struct KernelOnlyContextV1
{
    const ImageU8 *input;

    const Kernel2D *gaussian;

    const Kernel2D *sobelX;
    const Kernel2D *sobelY;

    ImageF32 *gaussianBuffer;

    ImageF32 *gx;
    ImageF32 *gy;

    ImageF32 *magnitude;
};


struct KernelOnlyContextV2
{
    const ImageU8 *input;

    const Kernel1D *gaussian;

    const Kernel2D *sobelX;
    const Kernel2D *sobelY;

    ImageF32 *intermediate;
    ImageF32 *gaussianBuffer;

    ImageF32 *gx;
    ImageF32 *gy;

    ImageF32 *magnitude;
};


struct KernelOnlyContextV3
{
    const ImageU8 *input;

    const Kernel2D *fusedX;
    const Kernel2D *fusedY;

    ImageF32 *gx;
    ImageF32 *gy;

    ImageF32 *magnitude;
};


/*
 * ============================================================
 * PIPELINE COMPLETO V1
 * ============================================================
 */

static void benchmarkPipelineV1(
        void *contextPointer)
{
    auto *context =
            static_cast<BenchmarkContextV1 *>(
                    contextPointer
            );


    ImageF32 gaussian = {0};

    ImageF32 gx = {0};
    ImageF32 gy = {0};

    ImageF32 magnitude = {0};


    if (!convolution_2d_direct_u8_f32(
            context->input,
            context->gaussian,
            &gaussian))
    {
        return;
    }


    if (!convolution_2d_f32_f32(
            &gaussian,
            context->sobelX,
            &gx))
    {
        image_f32_free(&gaussian);

        return;
    }


    if (!convolution_2d_f32_f32(
            &gaussian,
            context->sobelY,
            &gy))
    {
        image_f32_free(&gaussian);
        image_f32_free(&gx);

        return;
    }


    if (!gradient_magnitude_f32(
            &gx,
            &gy,
            &magnitude))
    {
        image_f32_free(&gaussian);

        image_f32_free(&gx);
        image_f32_free(&gy);

        return;
    }


    image_f32_free(&gaussian);

    image_f32_free(&gx);
    image_f32_free(&gy);

    image_f32_free(&magnitude);
}


/*
 * ============================================================
 * PIPELINE COMPLETO V2
 * ============================================================
 */

static void benchmarkPipelineV2(
        void *contextPointer)
{
    auto *context =
            static_cast<BenchmarkContextV2 *>(
                    contextPointer
            );


    ImageF32 intermediate = {0};
    ImageF32 gaussian = {0};

    ImageF32 gx = {0};
    ImageF32 gy = {0};

    ImageF32 magnitude = {0};


    if (!convolution_gaussian_separable(
            context->input,
            context->gaussian,
            &intermediate,
            &gaussian))
    {
        return;
    }


    if (!convolution_2d_f32_f32(
            &gaussian,
            context->sobelX,
            &gx))
    {
        image_f32_free(&intermediate);
        image_f32_free(&gaussian);

        return;
    }


    if (!convolution_2d_f32_f32(
            &gaussian,
            context->sobelY,
            &gy))
    {
        image_f32_free(&intermediate);
        image_f32_free(&gaussian);
        image_f32_free(&gx);

        return;
    }


    if (!gradient_magnitude_f32(
            &gx,
            &gy,
            &magnitude))
    {
        image_f32_free(&intermediate);
        image_f32_free(&gaussian);

        image_f32_free(&gx);
        image_f32_free(&gy);

        return;
    }


    image_f32_free(&intermediate);
    image_f32_free(&gaussian);

    image_f32_free(&gx);
    image_f32_free(&gy);

    image_f32_free(&magnitude);
}


/*
 * ============================================================
 * PIPELINE COMPLETO V3
 * ============================================================
 */

static void benchmarkPipelineV3(
        void *contextPointer)
{
    auto *context =
            static_cast<BenchmarkContextV3 *>(
                    contextPointer
            );


    ImageF32 gx = {0};
    ImageF32 gy = {0};

    ImageF32 magnitude = {0};


    if (!convolution_2d_direct_u8_f32(
            context->input,
            context->fusedX,
            &gx))
    {
        return;
    }


    if (!convolution_2d_direct_u8_f32(
            context->input,
            context->fusedY,
            &gy))
    {
        image_f32_free(&gx);

        return;
    }


    if (!gradient_magnitude_f32(
            &gx,
            &gy,
            &magnitude))
    {
        image_f32_free(&gx);
        image_f32_free(&gy);

        return;
    }


    image_f32_free(&gx);
    image_f32_free(&gy);

    image_f32_free(&magnitude);
}


/*
 * ============================================================
 * KERNEL-ONLY V1
 * ============================================================
 */

static void benchmarkKernelV1(
        void *contextPointer)
{
    auto *context =
            static_cast<KernelOnlyContextV1 *>(
                    contextPointer
            );


    convolution_2d_direct_u8_f32_into(
            context->input,
            context->gaussian,
            context->gaussianBuffer
    );


    convolution_2d_f32_f32_into(
            context->gaussianBuffer,
            context->sobelX,
            context->gx
    );


    convolution_2d_f32_f32_into(
            context->gaussianBuffer,
            context->sobelY,
            context->gy
    );


    gradient_magnitude_f32_into(
            context->gx,
            context->gy,
            context->magnitude
    );
}


/*
 * ============================================================
 * KERNEL-ONLY V2
 * ============================================================
 */

static void benchmarkKernelV2(
        void *contextPointer)
{
    auto *context =
            static_cast<KernelOnlyContextV2 *>(
                    contextPointer
            );


    convolution_gaussian_separable_into(
            context->input,
            context->gaussian,
            context->intermediate,
            context->gaussianBuffer
    );


    convolution_2d_f32_f32_into(
            context->gaussianBuffer,
            context->sobelX,
            context->gx
    );


    convolution_2d_f32_f32_into(
            context->gaussianBuffer,
            context->sobelY,
            context->gy
    );


    gradient_magnitude_f32_into(
            context->gx,
            context->gy,
            context->magnitude
    );
}


/*
 * ============================================================
 * KERNEL-ONLY V3
 * ============================================================
 */

static void benchmarkKernelV3(
        void *contextPointer)
{
    auto *context =
            static_cast<KernelOnlyContextV3 *>(
                    contextPointer
            );


    convolution_2d_direct_u8_f32_into(
            context->input,
            context->fusedX,
            context->gx
    );


    convolution_2d_direct_u8_f32_into(
            context->input,
            context->fusedY,
            context->gy
    );


    gradient_magnitude_f32_into(
            context->gx,
            context->gy,
            context->magnitude
    );
}


/*
 * ============================================================
 * MOSTRAR ESTADISTICAS
 * ============================================================
 */

static void printStats(
        const char *version,
        const BenchmarkStats &stats)
{
    std::cout
            << "\n"
            << version
            << "\n";

    std::cout
            << "----------------------------------------\n";


    std::cout
            << "Media:        "
            << stats.mean_ms
            << " ms\n";


    std::cout
            << "Mediana:      "
            << stats.median_ms
            << " ms\n";


    std::cout
            << "Desv. Std:    "
            << stats.stddev_ms
            << " ms\n";


    std::cout
            << "Minimo:       "
            << stats.min_ms
            << " ms\n";


    std::cout
            << "Maximo:       "
            << stats.max_ms
            << " ms\n";
}


/*
 * ============================================================
 * GUARDAR RESULTADOS VISUALES
 * ============================================================
 */

static bool saveEdgeImage(
        const ImageF32 &source,
        const std::string &path)
{
    if (source.data == nullptr)
    {
        return false;
    }


    cv::Mat sourceMat(
            source.height,
            source.width,
            CV_32FC1,
            source.data
    );


    cv::Mat normalized;


    cv::normalize(
            sourceMat,
            normalized,
            0.0,
            255.0,
            cv::NORM_MINMAX
    );


    cv::Mat output;


    normalized.convertTo(
            output,
            CV_8UC1
    );


    return cv::imwrite(
            path,
            output
    );
}


/*
 * ============================================================
 * SELECCION DE KERNEL
 * ============================================================
 */

static int selectKernelSize()
{
    int kernelSize = 0;


    while (true)
    {
        std::cout
                << "\nSeleccione tamano del kernel Gaussiano\n"
                << "(numero impar >= 3, por ejemplo 3, 5, 7, 9, 11): ";


        if (!(std::cin >> kernelSize))
        {
            std::cout
                    << "\nEntrada invalida. Introduzca un numero entero.\n";


            std::cin.clear();


            std::cin.ignore(
                    std::numeric_limits<std::streamsize>::max(),
                    '\n'
            );


            continue;
        }


        if (kernelSize >= 3 &&
            (kernelSize % 2) != 0)
        {
            return kernelSize;
        }


        std::cout
                << "\nKernel invalido.\n"
                << "El valor debe ser impar y mayor o igual a 3.\n";
    }
}


/*
 * ============================================================
 * MAIN
 * ============================================================
 */

int main()
{
    /*
     * ========================================================
     * DIRECTORIO DE RESULTADOS
     * ========================================================
     */

    std::filesystem::create_directories(
            "results"
    );


    /*
     * ========================================================
     * ENCABEZADO
     * ========================================================
     */

    std::cout
            << "\n========================================\n"
            << " CONVOLUTION ARM BENCHMARK\n"
            << "========================================\n";


    std::cout
            << "\nComparacion de:\n"
            << "  V1 - Convolucion 2D directa + Sobel\n"
            << "  V2 - Convolucion separable + Sobel\n"
            << "  V3 - Kernels fusionados\n";


    /*
     * ========================================================
     * CONFIGURACION DINAMICA
     * ========================================================
     */

    const int kernelSize =
            selectKernelSize();


    /*
     * Sigma automatico.
     *
     * Aproximamos que la ventana cubra aproximadamente
     * +/- 3 sigma:
     *
     *      sigma = k / 6
     */

    const float sigma =
            static_cast<float>(
                    kernelSize)
            /
            6.0f;


    const int warmupIterations =
            5;


    const int benchmarkIterations =
            100;


    /*
     * ========================================================
     * RUTAS
     * ========================================================
     */

    const std::string inputPath =
            "data/test.jpg";


    const std::string outputV1Path =
            "results/edges_v1_direct.png";


    const std::string outputV2Path =
            "results/edges_v2_separable.png";


    const std::string outputV3Path =
            "results/edges_v3_fused.png";


    /*
     * ========================================================
     * CARGA DE IMAGEN
     * ========================================================
     */

    cv::Mat image =
            cv::imread(
                    inputPath,
                    cv::IMREAD_COLOR
            );


    if (image.empty())
    {
        std::cerr
                << "\nERROR: no se pudo cargar:\n"
                << inputPath
                << "\n";

        return 1;
    }


    cv::Mat gray;


    cv::cvtColor(
            image,
            gray,
            cv::COLOR_BGR2GRAY
    );


    ImageU8 inputImage = {0};


    if (!image_u8_copy_from_buffer(
            &inputImage,
            gray.data,
            gray.cols,
            gray.rows,
            gray.step))
    {
        std::cerr
                << "ERROR: no se pudo crear ImageU8\n";

        return 1;
    }


    const long long totalPixels =
            static_cast<long long>(
                    inputImage.width)
            *
            static_cast<long long>(
                    inputImage.height);


    /*
     * ========================================================
     * CREACION DE KERNELS
     * ========================================================
     */

    Kernel1D gaussian1D = {0};


    if (!gaussian_kernel_1d_create(
            &gaussian1D,
            kernelSize,
            sigma))
    {
        std::cerr
                << "ERROR creando Gaussiano 1D\n";

        image_u8_free(&inputImage);

        return 1;
    }


    Kernel2D gaussian2D = {0};


    if (!kernel_outer_product(
            &gaussian1D,
            &gaussian1D,
            &gaussian2D))
    {
        std::cerr
                << "ERROR creando Gaussiano 2D\n";

        kernel1d_free(&gaussian1D);
        image_u8_free(&inputImage);

        return 1;
    }


    Kernel2D sobelX =
            sobel_x_kernel();


    Kernel2D sobelY =
            sobel_y_kernel();


    Kernel2D fusedX = {0};
    Kernel2D fusedY = {0};


    if (!kernel_compose_2d(
            &gaussian2D,
            &sobelX,
            &fusedX))
    {
        std::cerr
                << "ERROR creando kernel fusionado X\n";

        return 1;
    }


    if (!kernel_compose_2d(
            &gaussian2D,
            &sobelY,
            &fusedY))
    {
        std::cerr
                << "ERROR creando kernel fusionado Y\n";

        kernel2d_free(&fusedX);

        return 1;
    }


    /*
     * ========================================================
     * CONFIGURACION DEL EXPERIMENTO
     * ========================================================
     */

    std::cout
            << "\n========================================\n"
            << " CONFIGURACION DEL EXPERIMENTO\n"
            << "========================================\n";


    std::cout
            << "Build:                 Release\n";


    std::cout
            << "Resolucion:            "
            << inputImage.width
            << "x"
            << inputImage.height
            << "\n";


    std::cout
            << "Pixeles totales:       "
            << totalPixels
            << "\n";


    std::cout
            << "Kernel Gaussiano:      "
            << kernelSize
            << "x"
            << kernelSize
            << "\n";


    std::cout
            << std::fixed
            << std::setprecision(6);


    std::cout
            << "Sigma automatico:       "
            << sigma
            << "\n";


    std::cout
            << "Kernel Sobel:          "
            << sobelX.size
            << "x"
            << sobelX.size
            << "\n";


    std::cout
            << "Kernel fusionado:      "
            << fusedX.size
            << "x"
            << fusedX.size
            << "\n";


    std::cout
            << "Warm-up:               "
            << warmupIterations
            << "\n";


    std::cout
            << "Iteraciones medidas:   "
            << benchmarkIterations
            << "\n";


    /*
     * ========================================================
     * EJECUCION DE REFERENCIA V1
     * ========================================================
     */

    ImageF32 gaussianV1 = {0};

    ImageF32 gxV1 = {0};
    ImageF32 gyV1 = {0};

    ImageF32 edgesV1 = {0};


    if (!convolution_2d_direct_u8_f32(
            &inputImage,
            &gaussian2D,
            &gaussianV1) ||

        !convolution_2d_f32_f32(
            &gaussianV1,
            &sobelX,
            &gxV1) ||

        !convolution_2d_f32_f32(
            &gaussianV1,
            &sobelY,
            &gyV1) ||

        !gradient_magnitude_f32(
            &gxV1,
            &gyV1,
            &edgesV1))
    {
        std::cerr
                << "ERROR ejecutando V1\n";

        return 1;
    }


    /*
     * ========================================================
     * EJECUCION DE REFERENCIA V2
     * ========================================================
     */

    ImageF32 intermediateV2 = {0};
    ImageF32 gaussianV2 = {0};

    ImageF32 gxV2 = {0};
    ImageF32 gyV2 = {0};

    ImageF32 edgesV2 = {0};


    if (!convolution_gaussian_separable(
            &inputImage,
            &gaussian1D,
            &intermediateV2,
            &gaussianV2) ||

        !convolution_2d_f32_f32(
            &gaussianV2,
            &sobelX,
            &gxV2) ||

        !convolution_2d_f32_f32(
            &gaussianV2,
            &sobelY,
            &gyV2) ||

        !gradient_magnitude_f32(
            &gxV2,
            &gyV2,
            &edgesV2))
    {
        std::cerr
                << "ERROR ejecutando V2\n";

        return 1;
    }


    /*
     * ========================================================
     * EJECUCION DE REFERENCIA V3
     * ========================================================
     */

    ImageF32 gxV3 = {0};
    ImageF32 gyV3 = {0};

    ImageF32 edgesV3 = {0};


    if (!convolution_2d_direct_u8_f32(
            &inputImage,
            &fusedX,
            &gxV3) ||

        !convolution_2d_direct_u8_f32(
            &inputImage,
            &fusedY,
            &gyV3) ||

        !gradient_magnitude_f32(
            &gxV3,
            &gyV3,
            &edgesV3))
    {
        std::cerr
                << "ERROR ejecutando V3\n";

        return 1;
    }


    /*
     * ========================================================
     * VALIDACION NUMERICA
     * ========================================================
     */

    const int validBorder =
            fusedX.size / 2;


    const ErrorMetrics errorV1V2 =
            compare_images_f32(
                    &edgesV1,
                    &edgesV2,
                    validBorder
            );


    const ErrorMetrics errorV1V3 =
            compare_images_f32(
                    &edgesV1,
                    &edgesV3,
                    validBorder
            );


    const double validationTolerance =
            1e-3;


    std::cout
            << "\n========================================\n"
            << " 1. VALIDACION NUMERICA\n"
            << "========================================\n";


    std::cout
            << std::scientific
            << std::setprecision(8);


    std::cout
            << "\nV1 vs V2\n"
            << "MAE:       "
            << errorV1V2.mae
            << "\n"
            << "RMSE:      "
            << errorV1V2.rmse
            << "\n"
            << "MAX ERROR: "
            << errorV1V2.max_abs_error
            << "\n";


    std::cout
            << "Estado:    "
            << (
                errorV1V2.max_abs_error
                < validationTolerance
                ? "OK"
                : "NO EQUIVALENTE"
               )
            << "\n";


    std::cout
            << "\nV1 vs V3\n"
            << "MAE:       "
            << errorV1V3.mae
            << "\n"
            << "RMSE:      "
            << errorV1V3.rmse
            << "\n"
            << "MAX ERROR: "
            << errorV1V3.max_abs_error
            << "\n";


    std::cout
            << "Estado:    "
            << (
                errorV1V3.max_abs_error
                < validationTolerance
                ? "OK"
                : "NO EQUIVALENTE"
               )
            << "\n";


    std::cout
            << std::fixed
            << std::setprecision(6);


    /*
     * ========================================================
     * GUARDAR RESULTADOS
     * ========================================================
     */

    if (!saveEdgeImage(
            edgesV1,
            outputV1Path))
    {
        std::cout
                << "Advertencia guardando V1\n";
    }


    if (!saveEdgeImage(
            edgesV2,
            outputV2Path))
    {
        std::cout
                << "Advertencia guardando V2\n";
    }


    if (!saveEdgeImage(
            edgesV3,
            outputV3Path))
    {
        std::cout
                << "Advertencia guardando V3\n";
    }


    /*
     * ========================================================
     * BENCHMARK PIPELINE COMPLETO
     * ========================================================
     */

    BenchmarkContextV1 pipelineContextV1 =
    {
        &inputImage,
        &gaussian2D,
        &sobelX,
        &sobelY
    };


    BenchmarkContextV2 pipelineContextV2 =
    {
        &inputImage,
        &gaussian1D,
        &sobelX,
        &sobelY
    };


    BenchmarkContextV3 pipelineContextV3 =
    {
        &inputImage,
        &fusedX,
        &fusedY
    };


    const BenchmarkStats pipelineV1 =
            benchmark_run(
                    benchmarkPipelineV1,
                    &pipelineContextV1,
                    warmupIterations,
                    benchmarkIterations
            );


    const BenchmarkStats pipelineV2 =
            benchmark_run(
                    benchmarkPipelineV2,
                    &pipelineContextV2,
                    warmupIterations,
                    benchmarkIterations
            );


    const BenchmarkStats pipelineV3 =
            benchmark_run(
                    benchmarkPipelineV3,
                    &pipelineContextV3,
                    warmupIterations,
                    benchmarkIterations
            );


    std::cout
            << "\n========================================\n"
            << " 2. BENCHMARK PIPELINE COMPLETO\n"
            << "========================================\n";


    printStats(
            "V1 - Directa + Sobel",
            pipelineV1
    );


    printStats(
            "V2 - Separable + Sobel",
            pipelineV2
    );


    printStats(
            "V3 - Fusionada",
            pipelineV3
    );


    /*
     * ========================================================
     * PREASIGNACION KERNEL-ONLY
     * ========================================================
     */

    ImageF32 koV1Gaussian = {0};
    ImageF32 koV1Gx = {0};
    ImageF32 koV1Gy = {0};
    ImageF32 koV1Magnitude = {0};


    ImageF32 koV2Intermediate = {0};
    ImageF32 koV2Gaussian = {0};
    ImageF32 koV2Gx = {0};
    ImageF32 koV2Gy = {0};
    ImageF32 koV2Magnitude = {0};


    ImageF32 koV3Gx = {0};
    ImageF32 koV3Gy = {0};
    ImageF32 koV3Magnitude = {0};


    auto createBuffer =
            [&inputImage](
                    ImageF32 *buffer)
    {
        return image_f32_create(
                buffer,
                inputImage.width,
                inputImage.height
        );
    };


    if (!createBuffer(&koV1Gaussian) ||
        !createBuffer(&koV1Gx) ||
        !createBuffer(&koV1Gy) ||
        !createBuffer(&koV1Magnitude) ||

        !createBuffer(&koV2Intermediate) ||
        !createBuffer(&koV2Gaussian) ||
        !createBuffer(&koV2Gx) ||
        !createBuffer(&koV2Gy) ||
        !createBuffer(&koV2Magnitude) ||

        !createBuffer(&koV3Gx) ||
        !createBuffer(&koV3Gy) ||
        !createBuffer(&koV3Magnitude))
    {
        std::cerr
                << "ERROR reservando buffers\n";

        return 1;
    }


    /*
     * ========================================================
     * CONTEXTOS KERNEL-ONLY
     * ========================================================
     */

    KernelOnlyContextV1 kernelContextV1 =
    {
        &inputImage,

        &gaussian2D,

        &sobelX,
        &sobelY,

        &koV1Gaussian,

        &koV1Gx,
        &koV1Gy,

        &koV1Magnitude
    };


    KernelOnlyContextV2 kernelContextV2 =
    {
        &inputImage,

        &gaussian1D,

        &sobelX,
        &sobelY,

        &koV2Intermediate,
        &koV2Gaussian,

        &koV2Gx,
        &koV2Gy,

        &koV2Magnitude
    };


    KernelOnlyContextV3 kernelContextV3 =
    {
        &inputImage,

        &fusedX,
        &fusedY,

        &koV3Gx,
        &koV3Gy,

        &koV3Magnitude
    };


    /*
     * ========================================================
     * BENCHMARK KERNEL-ONLY
     * ========================================================
     */

    const BenchmarkStats kernelV1 =
            benchmark_run(
                    benchmarkKernelV1,
                    &kernelContextV1,
                    warmupIterations,
                    benchmarkIterations
            );


    const BenchmarkStats kernelV2 =
            benchmark_run(
                    benchmarkKernelV2,
                    &kernelContextV2,
                    warmupIterations,
                    benchmarkIterations
            );


    const BenchmarkStats kernelV3 =
            benchmark_run(
                    benchmarkKernelV3,
                    &kernelContextV3,
                    warmupIterations,
                    benchmarkIterations
            );


    std::cout
            << "\n========================================\n"
            << " 3. BENCHMARK KERNEL-ONLY\n"
            << "========================================\n";


    printStats(
            "V1 - Directa + Sobel",
            kernelV1
    );


    printStats(
            "V2 - Separable + Sobel",
            kernelV2
    );


    printStats(
            "V3 - Fusionada",
            kernelV3
    );


    /*
     * ========================================================
     * SPEEDUPS
     * ========================================================
     */

    const double pipelineMeanSpeedupV2 =
            pipelineV1.mean_ms /
            pipelineV2.mean_ms;


    const double pipelineMeanSpeedupV3 =
            pipelineV1.mean_ms /
            pipelineV3.mean_ms;


    const double pipelineMedianSpeedupV2 =
            pipelineV1.median_ms /
            pipelineV2.median_ms;


    const double pipelineMedianSpeedupV3 =
            pipelineV1.median_ms /
            pipelineV3.median_ms;


    const double kernelMeanSpeedupV2 =
            kernelV1.mean_ms /
            kernelV2.mean_ms;


    const double kernelMeanSpeedupV3 =
            kernelV1.mean_ms /
            kernelV3.mean_ms;


    const double kernelMedianSpeedupV2 =
            kernelV1.median_ms /
            kernelV2.median_ms;


    const double kernelMedianSpeedupV3 =
            kernelV1.median_ms /
            kernelV3.median_ms;


    /*
     * ========================================================
     * OVERHEAD
     * ========================================================
     */

    const double overheadMedianV1 =
            pipelineV1.median_ms -
            kernelV1.median_ms;


    const double overheadMedianV2 =
            pipelineV2.median_ms -
            kernelV2.median_ms;


    const double overheadMedianV3 =
            pipelineV3.median_ms -
            kernelV3.median_ms;


    /*
     * ========================================================
     * COSTO TEORICO
     * ========================================================
     */

    const long long v1Operations =
            static_cast<long long>(
                    kernelSize)
            *
            kernelSize
            +
            18;


    const long long v2Operations =
            static_cast<long long>(2)
            *
            kernelSize
            +
            18;


    const long long v3Operations =
            static_cast<long long>(2)
            *
            fusedX.size
            *
            fusedX.size;


    /*
     * ========================================================
     * SPEEDUP
     * ========================================================
     */

    std::cout
            << "\n========================================\n"
            << " 4. SPEEDUP\n"
            << "========================================\n";


    std::cout
            << "\nPipeline completo\n";


    std::cout
            << "V1 baseline:        1.000000x\n";


    std::cout
            << "V2 media:           "
            << pipelineMeanSpeedupV2
            << "x\n";


    std::cout
            << "V2 mediana:         "
            << pipelineMedianSpeedupV2
            << "x\n";


    std::cout
            << "V3 media:           "
            << pipelineMeanSpeedupV3
            << "x\n";


    std::cout
            << "V3 mediana:         "
            << pipelineMedianSpeedupV3
            << "x\n";


    std::cout
            << "\nKernel-only\n";


    std::cout
            << "V1 baseline:        1.000000x\n";


    std::cout
            << "V2 media:           "
            << kernelMeanSpeedupV2
            << "x\n";


    std::cout
            << "V2 mediana:         "
            << kernelMedianSpeedupV2
            << "x\n";


    std::cout
            << "V3 media:           "
            << kernelMeanSpeedupV3
            << "x\n";


    std::cout
            << "V3 mediana:         "
            << kernelMedianSpeedupV3
            << "x\n";


    /*
     * ========================================================
     * OVERHEAD DE MEMORIA
     * ========================================================
     */

    std::cout
            << "\n========================================\n"
            << " 5. OVERHEAD DE GESTION DE BUFFERS\n"
            << "========================================\n";


    std::cout
            << "Estimacion usando medianas.\n\n";


    std::cout
            << "V1: "
            << overheadMedianV1
            << " ms\n";


    std::cout
            << "V2: "
            << overheadMedianV2
            << " ms\n";


    std::cout
            << "V3: "
            << overheadMedianV3
            << " ms\n";


    /*
     * ========================================================
     * COSTO TEORICO
     * ========================================================
     */

    std::cout
            << "\n========================================\n"
            << " 6. COSTO TEORICO\n"
            << "========================================\n";


    std::cout
            << "V1: "
            << v1Operations
            << " mult/pixel\n";


    std::cout
            << "V2: "
            << v2Operations
            << " mult/pixel\n";


    std::cout
            << "V3: "
            << v3Operations
            << " mult/pixel\n";


    /*
     * ========================================================
     * VERSION GANADORA
     * ========================================================
     */

    const char *bestVersion =
            "V1";


    double bestMedian =
            kernelV1.median_ms;


    if (kernelV2.median_ms <
        bestMedian)
    {
        bestVersion =
                "V2";

        bestMedian =
                kernelV2.median_ms;
    }


    if (kernelV3.median_ms <
        bestMedian)
    {
        bestVersion =
                "V3";

        bestMedian =
                kernelV3.median_ms;
    }


    /*
     * ========================================================
     * RESUMEN FINAL
     * ========================================================
     */

    std::cout
            << "\n========================================\n"
            << " RESULTADO DEL EXPERIMENTO\n"
            << "========================================\n";


    std::cout
            << "Kernel seleccionado:      "
            << kernelSize
            << "x"
            << kernelSize
            << "\n";


    std::cout
            << "Sigma automatico:         "
            << sigma
            << "\n";


    std::cout
            << "Kernel fusionado:         "
            << fusedX.size
            << "x"
            << fusedX.size
            << "\n";


    std::cout
            << "Mejor implementacion:     "
            << bestVersion
            << "\n";


    std::cout
            << "Mejor mediana kernel:     "
            << bestMedian
            << " ms\n";


    if (std::string(bestVersion) == "V2")
    {
        std::cout
                << "\nConclusion:\n"
                << "La convolucion separable presenta el mejor\n"
                << "rendimiento para esta configuracion.\n";
    }
    else if (std::string(bestVersion) == "V3")
    {
        std::cout
                << "\nConclusion:\n"
                << "La fusion de kernels presenta el mejor\n"
                << "rendimiento para esta configuracion.\n";
    }
    else
    {
        std::cout
                << "\nConclusion:\n"
                << "La convolucion directa presenta el mejor\n"
                << "rendimiento para esta configuracion.\n";
    }


    std::cout
            << "\nEnergia: PENDIENTE\n";


    std::cout
            << "\nResultados visuales:\n"
            << outputV1Path
            << "\n"
            << outputV2Path
            << "\n"
            << outputV3Path
            << "\n";


    /*
     * ========================================================
     * LIBERACION KERNEL-ONLY
     * ========================================================
     */

    image_f32_free(
            &koV1Gaussian
    );

    image_f32_free(
            &koV1Gx
    );

    image_f32_free(
            &koV1Gy
    );

    image_f32_free(
            &koV1Magnitude
    );


    image_f32_free(
            &koV2Intermediate
    );

    image_f32_free(
            &koV2Gaussian
    );

    image_f32_free(
            &koV2Gx
    );

    image_f32_free(
            &koV2Gy
    );

    image_f32_free(
            &koV2Magnitude
    );


    image_f32_free(
            &koV3Gx
    );

    image_f32_free(
            &koV3Gy
    );

    image_f32_free(
            &koV3Magnitude
    );


    /*
     * ========================================================
     * LIBERACION REFERENCIAS
     * ========================================================
     */

    image_f32_free(
            &gaussianV1
    );

    image_f32_free(
            &gxV1
    );

    image_f32_free(
            &gyV1
    );

    image_f32_free(
            &edgesV1
    );


    image_f32_free(
            &intermediateV2
    );

    image_f32_free(
            &gaussianV2
    );

    image_f32_free(
            &gxV2
    );

    image_f32_free(
            &gyV2
    );

    image_f32_free(
            &edgesV2
    );


    image_f32_free(
            &gxV3
    );

    image_f32_free(
            &gyV3
    );

    image_f32_free(
            &edgesV3
    );


    /*
     * ========================================================
     * LIBERACION DE KERNELS
     * ========================================================
     */

    kernel1d_free(
            &gaussian1D
    );


    kernel2d_free(
            &gaussian2D
    );


    kernel2d_free(
            &fusedX
    );


    kernel2d_free(
            &fusedY
    );


    /*
     * Sobel X/Y utilizan memoria estática.
     * No se liberan.
     */


    image_u8_free(
            &inputImage
    );


    std::cout
            << "\n========================================\n"
            << " BENCHMARK FINALIZADO CORRECTAMENTE\n"
            << "========================================\n";


    return 0;
}