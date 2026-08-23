cmake_minimum_required(VERSION 3.20)

project(Debayering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV CONFIG REQUIRED)

add_executable(debayering
        src/main.cpp
)

target_include_directories(debayering
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(debayering
        PRIVATE
        ${OpenCV_LIBS}
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")