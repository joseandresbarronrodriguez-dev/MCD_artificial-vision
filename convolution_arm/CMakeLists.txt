add_executable(convolution_arm
        src/main.c
)

target_include_directories(convolution_arm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)