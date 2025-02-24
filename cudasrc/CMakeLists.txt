cmake_minimum_required(VERSION 3.17) 
project(cuda_matmul CUDA CXX)

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_matmul src/CUDASolMain.cu src/CUDASol.cu)

target_compile_features(cuda_matmul PRIVATE cuda_std_17)

target_link_libraries(cuda_matmul CUDA::cudart_static)

set_target_properties(cuda_matmul PROPERTIES
    CUDA_ARCHITECTURES "61"  
    COMPILE_OPTIONS "-G"
)

add_custom_target(run
    COMMAND $<TARGET_FILE:cuda_matmul> > output.txt 2>&1
    DEPENDS cuda_matmul
)