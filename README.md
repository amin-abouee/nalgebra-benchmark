# Nalgebra Matrix Decomposition Benchmarks

This project benchmarks various matrix decomposition algorithms available in the [nalgebra](https://nalgebra.org/) Rust library. It evaluates performance for different matrix sizes and shapes, similar to benchmarks available for Eigen.

## Decomposition Algorithms Benchmarked

- QR Decomposition (`.qr()`)
- LU Decomposition with partial pivoting (`.lu()`)
- LU Decomposition with full pivoting (`.full_piv_lu()`)
- Hessenberg Decomposition (`.hessenberg()`)
- Cholesky Decomposition (`.cholesky()`)
- Schur Decomposition (`.schur()`)
- Symmetric Eigendecomposition (`.symmetric_eigen()`)
- SVD (`.svd()`)

## Matrix Sizes

The benchmark tests each algorithm against:

- Square matrices: 8×8, 100×100, 1000×1000, 4000×4000
- Rectangular matrices: 10000×8, 10000×100, 10000×1000, 10000×4000

## Requirements

- Rust (2021 edition)

## Running the Benchmarks

To run the main program with performance measurements:

```sh
cargo run --release
```

To run the criterion benchmarks (a smaller subset):

```sh
cargo bench
```

## Expected Output

The benchmarks will output a table showing execution time in milliseconds for each algorithm and matrix size, with relative performance ratios compared to the QR algorithm (which is used as the baseline).

Example:

```
╔════════════════╦═══════╦═════════╦══════════╦════════════╦═══════════╦════════════╦═════════════╦══════════════╗
║ solver/size    ║ 8x8   ║ 100x100 ║ 1000x1000 ║ 4000x4000  ║ 10000x8  ║ 10000x100 ║ 10000x1000 ║ 10000x4000   ║
╠════════════════╬═══════╬═════════╬══════════╬════════════╬═══════════╬════════════╬═════════════╬══════════════╣
║ QR             ║ 0.05  ║ 0.42    ║ 5.83     ║ 374.55    ║ 6.79 *    ║ 30.15 *   ║ 236.34 *    ║ 3847.17 *    ║
╠════════════════╬═══════╬═════════╬══════════╬════════════╬═══════════╬════════════╬═════════════╬══════════════╣
║ LU             ║ 0.08  ║ 0.69    ║ 15.63    ║ 709.32    ║ 6.81 *    ║ 31.32 *   ║ 241.68 *    ║ 4270.48 *    ║
╠════════════════╬═══════╬═════════╬══════════╬════════════╬═══════════╬════════════╬═════════════╬══════════════╣
╠════════════════╬═══════╬═════════╬══════════╬════════════╬═══════════╬════════════╬═════════════╬══════════════╣
// ... other algorithms ...
╚════════════════╩═══════╩═════════╩══════════╩════════════╩═══════════╩════════════╩═════════════╩══════════════╝
```

*Note: The "-" symbol indicates cases where the algorithm could not be run for a particular matrix size (typically for large matrices that would take too long to process).* 
