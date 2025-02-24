# OpenMP Matrix Multiplication

This repository implements matrix multiplication using OpenMP for parallel processing.

## Repository Structure

```
.
├── Makefile
├── matmul_perf
│   ├── matmul
│   ├── openmp.sub
│   └── openMPTest.o86932
├── openmp.sub
└── src
    ├── OpenMPSol.h
    ├── OpenMPSoll.cc
    ├── OpenMPSoll.o
    ├── OpenMPSolMain.cc
    └── OpenMPSolMain.o
```

## How to Build and Run

### On HPC

1. **Build the Project:**
   ```bash
   make
   ```

2. **Submit Your Job:**
   - **Option 1:** Change to the `matmul_perf` directory and run the executable:
     ```bash
     cd matmul_perf
     ./matmul
     ```
   - **Option 2:** Copy the submission script into the appropriate directory and submit with `qsub`:
     ```bash
     cp openmp.sub matmul_perf/
     cd matmul_perf
     qsub openmp.sub
     ```

### On a Personal PC

1. **Build the Project:**
   ```bash
   make
   ```

2. **Run the Executable:**
   ```bash
   cd matmul_perf
   ./matmul
   ```

