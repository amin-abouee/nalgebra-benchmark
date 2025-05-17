# Nalgebra Matrix Decomposition Benchmarks

This project benchmarks various matrix decomposition algorithms available in the [nalgebra](https://nalgebra.org/) Rust library. It evaluates performance for different matrix sizes and shapes, similar to benchmarks available for Eigen.

## Decomposition Algorithms Benchmarked

- Cholesky Decomposition (`.cholesky()`)
- LU Decomposition with partial pivoting (`.lu()`)
- LU Decomposition with full pivoting (`.full_piv_lu()`)
- QR Decomposition (`.qr()`)
- Hessenberg Decomposition (`.hessenberg()`)
- Schur Decomposition (`.schur()`)
- Hermitian Eigendecomposition (`.symmetric_eigen()`)
- SVD (`.svd()`)

## Matrix Sizes

The benchmark tests each algorithm against:

- Square matrices: 8×8, 100×100, 1000×1000, 4000×4000
- Rectangular matrices: 10000×8, 10000×100, 10000×1000, 10000×4000

## Understanding Accuracy Metrics

The benchmark measures both performance (timing) and accuracy for each algorithm:

- **Accuracy**: Measured as the Frobenius norm of the difference between the original matrix and the reconstructed matrix after decomposition. Lower values indicate better accuracy.
- **Scientific Notation**: Accuracy is displayed in scientific notation (e.g., `1.39e-15` means 1.39 × 10^-15).
- **Interpretation**: 
  - Values close to machine epsilon (~1e-15 to 1e-16) indicate excellent numerical stability.
  - Larger values (e.g., 1e0 or greater) indicate significant numerical error or that the algorithm is not designed to reconstruct the original matrix exactly.

## BLAS Implementation

These benchmarks use OpenBLAS as the default BLAS (Basic Linear Algebra Subprograms) implementation. OpenBLAS is an optimized open-source implementation of BLAS that provides efficient matrix operations and significantly impacts the performance of linear algebra algorithms.

The performance results are directly influenced by the OpenBLAS implementation, which provides:

- Multi-threading capabilities for large matrices
- Optimized implementations for different CPU architectures
- Efficient memory access patterns

Different BLAS implementations (like Intel MKL or Apple Accelerate) may yield different performance results.

## Requirements

- Rust (2021 edition)

## Running the Benchmarks

To run the main program with performance measurements:

```sh
cargo run --release
```

## Expected Output

The benchmarks will output a table showing execution time (in milliseconds or nanoseconds), relative performance ratios, and reconstruction accuracy for each algorithm and matrix size. 

- **Baseline**: The Cholesky decomposition is used as the baseline for relative performance ratios (x-factor).
- **Accuracy**: Reported as the Frobenius norm of the difference between the original matrix and the reconstructed matrix.
- **Asterisk `*`**: For Cholesky, LU, and FullPivLU on rectangular matrices, this symbol indicates that the algorithm operates on `A'A`, and the timing/accuracy reflects this.
- **Dash `-`**: Indicates cases where an algorithm is skipped for a particular matrix size (e.g., non-square inputs for square-only algorithms).

### Timing Results Table

```
┌────────────────┬─────────────────┬─────────────────┬────────────────────┬──────────────────────┬────────────────────┬────────────────────┬─────────────────────┬───────────────────────┐
│ solver/size    │ 8x8             │ 100x100         │ 1000x1000          │ 4000x4000            │ 10000x8            │ 10000x100          │ 10000x1000          │ 10000x4000            │
├────────────────┼─────────────────┼─────────────────┼────────────────────┼──────────────────────┼────────────────────┼────────────────────┼─────────────────────┼───────────────────────┤
│ Cholesky       │ 0.0001          │ 0.0280          │ 26.3418            │ 2341.7360            │ 0.000 *            │ 0.028 *            │ 26.440 *            │ 2207.161 *            │
├────────────────┼─────────────────┼─────────────────┼────────────────────┼──────────────────────┼────────────────────┼────────────────────┼─────────────────────┼───────────────────────┤
│ LU             │ 0.0002 (x1.33)  │ 0.0484 (x1.73)  │ 56.4018 (x2.14)    │ 4998.9053 (x2.13)    │ 0.059 (x705.82) *  │ 4.630 (x167.10) *  │ 425.947 (x16.11) *  │ 10351.877 (x4.69) *   │
├────────────────┼─────────────────┼─────────────────┼────────────────────┼──────────────────────┼────────────────────┼────────────────────┼─────────────────────┼───────────────────────┤
│ FullPivLU      │ 0.0005 (x4.00)  │ 0.3582 (x12.77) │ 373.0143 (x14.16)  │ 25397.3867 (x10.85)  │ 0.065 (x788.16) *  │ 5.013 (x180.91) *  │ 743.845 (x28.13) *  │ 31196.648 (x14.13) *  │
├────────────────┼─────────────────┼─────────────────┼────────────────────┼──────────────────────┼────────────────────┼────────────────────┼─────────────────────┼───────────────────────┤
│ QR             │ 0.0003 (x2.33)  │ 0.0780 (x2.78)  │ 89.1844 (x3.39)    │ 6659.8002 (x2.84)    │ 0.133 (x1605.93) * │ 17.466 (x630.33) * │ 1680.134 (x63.55) * │ 24428.374 (x11.07) *  │
├────────────────┼─────────────────┼─────────────────┼────────────────────┼──────────────────────┼────────────────────┼────────────────────┼─────────────────────┼───────────────────────┤
│ Hessenberg     │ 0.0005 (x3.66)  │ 0.1818 (x6.48)  │ 225.2367 (x8.55)   │ 16738.5972 (x7.15)   │ -                  │ -                  │ -                   │ -                     │
├────────────────┼─────────────────┼─────────────────┼────────────────────┼──────────────────────┼────────────────────┼────────────────────┼─────────────────────┼───────────────────────┤
│ Schur          │ 0.0043 (x34.00) │ 1.8235 (x65.03) │ 1596.0479 (x60.59) │ 163522.2493 (x69.83) │ -                  │ -                  │ -                   │ -                     │
├────────────────┼─────────────────┼─────────────────┼────────────────────┼──────────────────────┼────────────────────┼────────────────────┼─────────────────────┼───────────────────────┤
│ HermitianEigen │ 0.0021 (x17.00) │ 0.4520 (x16.12) │ 381.5837 (x14.49)  │ 31564.4653 (x13.48)  │ -                  │ -                  │ -                   │ -                     │
├────────────────┼─────────────────┼─────────────────┼────────────────────┼──────────────────────┼────────────────────┼────────────────────┼─────────────────────┼───────────────────────┤
│ SVD            │ 0.0030 (x24.34) │ 1.0108 (x36.05) │ 934.3805 (x35.47)  │ 163472.4715 (x69.81) │ 0.606 (x7305.72) * │ 81.549 (x2943.06) *│ 8532.376 (x322.71) *│ 266361.922 (x120.68) *│
└────────────────┴─────────────────┴─────────────────┴────────────────────┴──────────────────────┴────────────────────┴────────────────────┴─────────────────────┴───────────────────────┘
```

### Accuracy Results Table

```
┌────────────────┬──────────┬──────────┬───────────┬───────────┬────────────┬────────────┬────────────┬────────────┐
│ solver/size    │ 8x8      │ 100x100  │ 1000x1000 │ 4000x4000 │ 10000x8    │ 10000x100  │ 10000x1000 │ 10000x4000 │
├────────────────┼──────────┼──────────┼───────────┼───────────┼────────────┼────────────┼────────────┼────────────┤
│ Cholesky       │ 1.39e-15 │ 1.32e-13 │ 1.23e-11  │ 1.85e-10  │ 1.52e-15 * │ 1.26e-13 * │ 1.24e-11 * │ 1.87e-10 * │
├────────────────┼──────────┼──────────┼───────────┼───────────┼────────────┼────────────┼────────────┼────────────┤
│ LU             │ 5.85e0   │ 8.02e1   │ 8.15e2    │ 3.26e3    │ 6.43e-13 * │ 1.09e-11 * │ 9.68e-11 * │ 4.30e-10 * │
├────────────────┼──────────┼──────────┼───────────┼───────────┼────────────┼────────────┼────────────┼────────────┤
│ FullPivLU      │ 5.21e0   │ 8.17e1   │ 8.16e2    │ 3.27e3    │ 3.18e2 *   │ 4.71e3 *   │ 4.71e4 *   │ 1.89e5 *   │
├────────────────┼──────────┼──────────┼───────────┼───────────┼────────────┼────────────┼────────────┼────────────┤
│ QR             │ 2.82e-15 │ 4.64e-14 │ 1.01e-12  │ 7.60e-12  │ 3.18e-13 * │ 1.00e-12 * │ 4.43e-12 * │ 1.41e-11 * │
├────────────────┼──────────┼──────────┼───────────┼───────────┼────────────┼────────────┼────────────┼────────────┤
│ Hessenberg     │ 4.04e-15 │ 9.43e-14 │ 1.88e-12  │ 1.44e-11  │ -          │ -          │ -          │ -          │
├────────────────┼──────────┼──────────┼───────────┼───────────┼────────────┼────────────┼────────────┼────────────┤
│ Schur          │ 3.66e-14 │ 4.96e-12 │ 4.29e-10  │ 6.58e-9   │ -          │ -          │ -          │ -          │
├────────────────┼──────────┼──────────┼───────────┼───────────┼────────────┼────────────┼────────────┼────────────┤
│ HermitianEigen │ 1.71e-14 │ 2.24e-12 │ 2.17e-10  │ 3.64e-9   │ -          │ -          │ -          │ -          │
├────────────────┼──────────┼──────────┼───────────┼───────────┼────────────┼────────────┼────────────┼────────────┤
│ SVD            │ 5.91e-15 │ 2.06e-13 │ 6.01e-12  │ 4.68e-11  │ 4.15e-13 * │ 2.13e-12 * │ 1.81e-11 * │ 7.45e-11 * │
└────────────────┴──────────┴──────────┴───────────┴───────────┴────────────┴────────────┴────────────┴────────────┘
```

**Note**: The accuracy results show interesting patterns:
- For square matrices, Cholesky, QR, Hessenberg, and SVD provide excellent numerical stability.
- LU and FullPivLU decompositions show higher reconstruction errors for square matrices, which is expected due to their pivoting strategies.
- For rectangular matrices, the accuracy varies significantly between algorithms, with QR and SVD generally providing better stability.
- The asterisk (*) indicates that for rectangular matrices, some algorithms operate on A'A instead of A directly, which affects both timing and accuracy measurements.