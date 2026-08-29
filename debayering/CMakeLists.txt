cmake_minimum_required(VERSION 3.20)

project(Debayering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==========================
# Dependencias
# ==========================

find_package(OpenCV CONFIG REQUIRED)
find_package(libraw CONFIG REQUIRED)

# ==========================
# Ejecutable
# ==========================

add_executable(debayering
        src/main.cpp
        include/RawLoader.h
        src/RawLoader.cpp
        include/Debayer.h
        src/Debayer.cpp
        include/ImageUtils.h
        src/ImageUtils.cpp
        include/Benchmark.h
        src/Benchmark.cpp
        include/LibRawReference.h
        src/LibRawReference.cpp
        include/BayerSelector.h
        src/BayerSelector.cpp
)

# ==========================
# Includes propios
# ==========================

target_include_directories(debayering
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

# ==========================
# Librerías
# ==========================

target_link_libraries(debayering
        PRIVATE
        ${OpenCV_LIBS}
        libraw::raw
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")