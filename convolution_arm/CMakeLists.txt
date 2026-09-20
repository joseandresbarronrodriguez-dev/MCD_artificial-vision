find_package(OpenCV REQUIRED)

add_executable(convolution_arm
        src/main.cpp
        include/image.h
        src/image.c
        include/kernels.h
        src/kernels.c
        include/convolution.h
        src/convolution.c
        include/metrics.h
        src/metrics.c
        include/benchmark.h
        src/benchmark.c
        include/energy.h
        src/energy.c
        include/csv_writer.h
        src/csv_writer.c
        ../hough_harris_arm/src/main.cpp
)

target_include_directories(convolution_arm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(convolution_arm PRIVATE
        ${OpenCV_LIBS}
)