cmake_minimum_required(VERSION 3.18) 
project(cuda_matmul LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_matmul 
    src/CUDASolMain.cu 
    src/CUDASol.cu
)

target_compile_features(cuda_matmul PRIVATE 
    cuda_std_17
)

target_link_libraries(cuda_matmul 
    OpenMP::OpenMP_CXX
)

set_target_properties(cuda_matmul PROPERTIES
    CUDA_ARCHITECTURES "70"
)

target_compile_options(cuda_matmul PRIVATE
    $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-G -lineinfo -O0>
    $<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O4>
)

# add_custom_target(run
#     COMMAND $<TARGET_FILE:cuda_matmul> > output.txt 2>&1
#     DEPENDS cuda_matmul
# )
