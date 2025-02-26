# CUDA Kernel for Non-Square Matrix Multiplication

My solution to _redacted_ University's homework (Sorry can't leak). The computation considered is

$$
C = A \cdot B,
$$

where $A$ is an $n \times k$ matrix, $B$ is a $k \times n$ matrix, and $C$ is an $n \times n$ matrix, with $k < n$. 

![ezgif-66599899878f5](https://github.com/user-attachments/assets/7fc2f895-b4c9-4be4-835e-0fa9a80a4f18)


## CUDA Implementations

All CUDA kernels utilize floating-point numbers (`floats`) _(you can use double if you care about precision, i personally care about speed)_ and employ row-major matrix storage, following standard C/C++ conventions.

### 1. Naive CUDA Kernel

The *(slow)* naive kernel directly translates the serial matrix multiplication algorithm into a parallel CUDA execution. Each thread computes a single element of the output matrix $C$ by fetching data directly from global memory. This kernel is expected to be memory-bandwidth limited, as each thread accesses global memory for every multiplication operation. Note each element of $C$ is given by 

$$ 
C_{ij} = \sum_{p=0}^{k-1} A_{ip} \cdot B_{pj},\quad \text{for } 0 \le i,j < n.
$$

_(Standard Matrix Multiplication)._

### 2. Shared Memory CUDA Kernel

Of course, this implementation comes with global memory bandwidth, which can be solved by giving it more memory. I personally implemented a shared memory kernel to mitigate these limitations. This approach leverages tiling to load sub-matrices of $A$ and $B$ into faster shared memory. In this configuration, each element of matrix $C$ requires data from an entire row of $A$ and a column of $B$, which are processed in tiles corresponding to the block dimensions of $C$.

The procedure is as follows:

1. Threads load a tile from global memory into shared memory.
2. A synchronization step ensures that all threads in the block have completed loading the tile.
3. A dot product is computed for each thread over the elements in the tile, accumulating a partial sum.

```math
\text{partialSum}^{(t)}_{ij} = \sum_{p=0}^{T-1} \text{tile\_A}^{(t)}[i,p] \cdot \text{tile\_B}^{(t)}[p,j].
```

4. A loop, optimized with a `#pragma unroll` directive to reduce overhead, iterates over the necessary tiles.
5. A final synchronization ensures that all computations using the current tile are complete before loading the next tile.
6. After loop completion, the accumulated partial sums represent the final value for the corresponding element in matrix $C$.

```math
C_{ij} = \sum_{t=0}^{\left\lceil \frac{k}{T} \right\rceil - 1} \text{partialSum}^{(t)}_{ij}.
```

### 3. Padding in Shared Memory

Introducing problem 2, shared memory is organized into banks (or stripes), and simultaneous access by multiple threads to the same bank can lead to serialized memory access. Thus, padding is introduced to adjust the memory layout and prevent bank conflicts. This modification is particularly relevant in the computation line:

```cpp
sum += tile_A[threadIdx.x][p] * tile_B[p][threadIdx.y];
```

Without padding, threads accessing elements within the same bank may cause serialized accesses within a warp, thereby degrading performance.

## Conclusion

Cool project, thanks @ _redacted_. An OpenMP implementation is also provided for comparative analysis in a parallel computing environment.
