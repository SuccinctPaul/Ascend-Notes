import numpy as np
import logging
import time
import tools
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)






def gemm_native(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Native GEMM function
    Theory:Python GEMM = Python MatMul

    MatMul: C= A*B

    GEMM = MatMul + bias + scale + transpose + layout + dtype + epilogue
        eg: C= 𝛼AB+𝛽𝐶
    
    Args:
        A (np.ndarray): Input matrix A
        B (np.ndarray): Input matrix B
        
    Returns:
        np.ndarray: Output matrix C = A * B
    """
    # Check input shapes
    if A.shape[1] != B.shape[0]:
        raise ValueError("A must have the same number of columns as B must have rows")
    # Check input types
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise ValueError("A and B must be numpy arrays")

    # Get matrix shape
    M, K = A.shape
    K2, N = B.shape

    logging.debug("Matrix A Size: %d x %d", M, K)
    logging.debug("Matrix B Size: %d x %d", K2, N)

    assert K == K2, "A must have the same number of columns as B must have rows == N"
    C = np.zeros((M, N), dtype=A.dtype)
    logging.debug("Matrix C Size: %d x %d", M, N)


    start = time.perf_counter()

    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i, k] * B[k, j]
            C[i, j] = s

    # 你的代码
    end = time.perf_counter()

    logging.debug("Time: %s \n", tools.format_time_color(end - start))

    return C



# Test
if __name__ == "__main__":
    A = np.random.rand(20, 30)
    B = np.random.rand(30, 40)
    C = gemm_native(A, B)   
    # print(C)clear