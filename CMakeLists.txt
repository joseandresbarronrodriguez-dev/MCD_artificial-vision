cmake_minimum_required(VERSION 3.20)

project(Vision_Artificial LANGUAGES C CXX)

# ============================
# Configuración C++
# ============================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================
# Configuración C
# ============================
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ============================
# Módulos del repositorio
# ============================
add_subdirectory(debayering)
add_subdirectory(convolution_arm)