# ============================================================
# TAREA 3 - HOUGH TRANSFORM + HARRIS CORNER SCORE
# ============================================================
#
# Objetivo:
#   Preparar un benchmark para comparar el costo computacional
#   de Hough y Harris en plataformas x86 y ARM.
#
# Dependencias:
#   OpenCV -> procesamiento de imagen y algoritmos de referencia.
#   Eigen  -> operaciones matriciales y almacenamiento eficiente.
#
# En esta primera etapa únicamente verificamos que ambas
# librerías estén correctamente configuradas.
# ============================================================

cmake_minimum_required(VERSION 3.21)

project(hough_harris_arm LANGUAGES CXX)

# ------------------------------------------------------------
# Estándar de C++
# ------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ------------------------------------------------------------
# Dependencias
# ------------------------------------------------------------

# OpenCV será utilizado para:
#   - lectura de imágenes
#   - conversión a escala de grises
#   - Canny / Sobel
#   - algoritmos de referencia
find_package(OpenCV CONFIG REQUIRED)

# Eigen será utilizado para:
#   - acumulador de Hough
#   - matrices del detector Harris
#   - operaciones matriciales optimizadas
find_package(Eigen3 CONFIG REQUIRED)


# ------------------------------------------------------------
# Ejecutable
# ------------------------------------------------------------

add_executable(hough_harris_arm

        src/main.cpp
        include/HoughAccumulator.h
        src/HoughAccumulator.cpp
        include/HarrisScore.h
        src/HarrisScore.cpp
        include/Benchmark.h
        src/Benchmark.cpp

)


# ------------------------------------------------------------
# Headers propios
# ------------------------------------------------------------

target_include_directories(hough_harris_arm
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)


# ------------------------------------------------------------
# Librerías
# ------------------------------------------------------------

target_link_libraries(hough_harris_arm
        PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
)


# ------------------------------------------------------------
# Optimización
#
# Windows -> MSVC
# ARM     -> GCC / Clang
# ------------------------------------------------------------

# ------------------------------------------------------------
# Optimización
#
# Debug:
#   Se conservan las opciones de depuración del compilador.
#
# Release:
#   Se habilita optimización para que posteriormente los
#   benchmarks representen el rendimiento real del algoritmo.
#
# Esto evita conflictos entre:
#
#   MSVC Debug   -> /RTC1 /Od
#   MSVC Release -> /O2
#
# ------------------------------------------------------------

if(MSVC)

    target_compile_options(
            hough_harris_arm
            PRIVATE

            $<$<CONFIG:Release>:/O2>
    )

else()

    target_compile_options(
            hough_harris_arm
            PRIVATE

            $<$<CONFIG:Release>:-O3>
    )

endif()