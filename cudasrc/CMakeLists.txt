cmake_minimum_required(VERSION 3.17) 
project(cuda_matmul CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cuda_matmul src/CUDASolMain.cu src/CUDASol.cu)

target_compile_features(cuda_matmul PRIVATE cuda_std_17)

target_link_libraries(cuda_matmul 
    CUDA::cudart_static
    OpenMP::OpenMP_CXX
)

set_target_properties(cuda_matmul PROPERTIES
    CUDA_ARCHITECTURES "61"  
    COMPILE_OPTIONS "-G"
)

target_compile_options(cuda_matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
)

add_custom_target(run
    COMMAND $<TARGET_FILE:cuda_matmul> > output.txt 2>&1
    DEPENDS cuda_matmul
)
