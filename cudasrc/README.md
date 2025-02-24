# CUDASol Matrix Multiplication

This repository implements matrix multiplication using CUDA with three different approaches:
- **Naive Implementation**
- **Shared Memory Optimization**
- **Padded Shared Memory Optimization**

## Repository Structure

```
.
├── CMakeLists.txt
├── cudatest.sub
├── Makefile
└── src
    ├── CUDASol.cu
    ├── CUDASol.cuh
    └── CUDASolMain.cu
```

## How to Build and Run

### On HPC (High Performance Computing)

1. **Create a Build Directory and Build:**
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

2. **Prepare and Submit Your Job:**
   ```bash
   cp ../cudatest.sub .
   ```
   Edit `cudatest.sub` to suit your specific environment or requirements. Then submit your job:
   ```bash
   qsub cudatest.sub
   ```

### On a Personal PC

1. **Build the Project:**
   ```bash
   make
   ```

2. **Run the Executable:**
   ```bash
   make run
   ```

## Implementation Details

- **Naive Matrix Multiplication:**  
  Implemented in the `matmul_naive` function within `CUDASol.cu`.

- **Shared Memory Optimization:**  
  Implemented in the `matmul_shared` function, which uses CUDA shared memory to speed up computations.

- **Padded Shared Memory Optimization:**  
  Implemented in the `matmul_shared_padded` function, which adds padding to avoid shared memory bank conflicts.

- **Reference Implementation:**  
  A serial version (`matmul_ref`) is provided for validation purposes.


