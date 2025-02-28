cmake_minimum_required(VERSION 3.18)
project(cuda_matmul LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)

# use c++17 for host
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# use c++17 for CUDA device code
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# create target
add_executable(cuda_matmul src/CUDASolMain.cu src/CUDASol.cu)

# if the above lines fail, use older approach:
# target_compile_options(cuda_matmul PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>
# )

target_link_libraries(cuda_matmul
    OpenMP::OpenMP_CXX
)

set_target_properties(cuda_matmul PROPERTIES
    CUDA_ARCHITECTURES "70"
)

target_compile_options(cuda_matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
