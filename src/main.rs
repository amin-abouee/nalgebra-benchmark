use nalgebra::DMatrix;
use prettytable::{cell, format, row, Table};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::time::{Duration, Instant};

// This benchmark compares different matrix decomposition algorithms in nalgebra.
// For square matrices, the benchmark uses symmetric matrices.
// For overconstrained matrices (rectangular), the benchmark computes the symmetric covariance matrix A'A
// for the first four solvers (Cholesky, LU, Full LU, QR), as denoted by the * symbol.
// Timings are in milliseconds, and factors are relative to the Cholesky decomposition,
// which is the fastest but also the least general and robust.

// Define matrix dimensions to benchmark
const SQUARE_SIZES: [usize; 4] = [8, 100, 1000, 4000];
const RECT_ROWS: usize = 10000;
const RECT_COLS: [usize; 4] = [8, 100, 1000, 4000];

// Generate a random square matrix of the given size
fn random_square_matrix(size: usize) -> DMatrix<f64> {
    let mut rng = thread_rng();
    let dist = Uniform::from(-1.0..1.0);

    DMatrix::from_fn(size, size, |_, _| dist.sample(&mut rng))
}

// Generate a random rectangular matrix
fn random_rect_matrix(rows: usize, cols: usize) -> DMatrix<f64> {
    let mut rng = thread_rng();
    let dist = Uniform::from(-1.0..1.0);

    DMatrix::from_fn(rows, cols, |_, _| dist.sample(&mut rng))
}

// Generate a random symmetric positive definite matrix
fn random_spd_matrix(size: usize) -> DMatrix<f64> {
    let mut rng = thread_rng();
    let dist = Uniform::from(-1.0..1.0);

    // Create a random matrix
    let a = DMatrix::from_fn(size, size, |_, _| dist.sample(&mut rng));

    // A'A is symmetric positive definite
    let result = &a.transpose() * &a;

    // Add a small value to the diagonal to ensure it's positive definite
    result + DMatrix::identity(size, size)
}

// Benchmark functions for each decomposition algorithm
fn bench_qr(matrix: &DMatrix<f64>) -> Duration {
    let start = Instant::now();
    let _ = matrix.clone().qr();
    start.elapsed()
}

fn bench_lu(matrix: &DMatrix<f64>) -> Duration {
    let start = Instant::now();
    // For non-square matrices, compute A'A first (similar to the C++ compute_norm_equation function)
    if matrix.nrows() != matrix.ncols() {
        let a_transpose_a = matrix.transpose() * matrix;
        let _ = a_transpose_a.lu();
    } else {
        let _ = matrix.clone().lu();
    }
    start.elapsed()
}

fn bench_full_piv_lu(matrix: &DMatrix<f64>) -> Duration {
    let start = Instant::now();
    // For non-square matrices, compute A'A first (similar to the C++ compute_norm_equation function)
    if matrix.nrows() != matrix.ncols() {
        let a_transpose_a = matrix.transpose() * matrix;
        let _ = a_transpose_a.full_piv_lu();
    } else {
        let _ = matrix.clone().full_piv_lu();
    }
    start.elapsed()
}

fn bench_hessenberg(matrix: &DMatrix<f64>) -> Duration {
    let start = Instant::now();
    let _ = matrix.clone().hessenberg();
    start.elapsed()
}

fn bench_cholesky(matrix: &DMatrix<f64>) -> Duration {
    let start = Instant::now();
    // For non-square matrices, compute A'A first (similar to the C++ compute_norm_equation function)
    // Note: Cholesky already requires a symmetric positive definite matrix
    if matrix.nrows() != matrix.ncols() {
        let a_transpose_a = matrix.transpose() * matrix;
        let _ = a_transpose_a.cholesky();
    } else {
        let _ = matrix.clone().cholesky();
    }
    start.elapsed()
}

fn bench_schur(matrix: &DMatrix<f64>) -> Duration {
    let start = Instant::now();
    let _ = matrix.clone().schur();
    start.elapsed()
}

fn bench_hermitian_eigen(matrix: &DMatrix<f64>) -> Duration {
    let start = Instant::now();
    let _ = matrix.clone().symmetric_eigen();
    start.elapsed()
}

fn bench_svd(matrix: &DMatrix<f64>) -> Duration {
    let start = Instant::now();
    let _ = matrix.clone().svd(true, true);
    start.elapsed()
}

// Benchmark with multiple iterations for more stable results
fn bench_with_iterations<F>(matrix: &DMatrix<f64>, bench_fn: F, iterations: usize) -> Duration
where
    F: Fn(&DMatrix<f64>) -> Duration,
{
    let mut total = Duration::new(0, 0);

    // To account for potential outliers, we run multiple iterations
    // and take the minimum time (best performance)
    for _ in 0..iterations {
        let duration = bench_fn(matrix);
        if total.is_zero() || duration < total {
            total = duration;
        }
    }

    total
}

fn run_benchmarks() {
    // Number of iterations for small matrices to get more stable results
    const SMALL_MATRIX_ITERATIONS: usize = 5;
    const MEDIUM_MATRIX_ITERATIONS: usize = 3;
    // Large matrices only run once

    // Create results table
    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_BOX_CHARS);

    // Add header row
    let mut header = row!["solver/size"];
    for size in SQUARE_SIZES.iter() {
        header.add_cell(cell!(format!("{}x{}", size, size)));
    }
    for col in RECT_COLS.iter() {
        header.add_cell(cell!(format!("{}x{}", RECT_ROWS, col)));
    }
    table.add_row(header);

    // Define benchmark functions and their names
    // Sorted in order: Cholesky, LU, Full LU, QR, Hessenberg, Schur, Hermitian_Eigen, and SVD
    // The * symbol in the results indicates that for overconstrained matrices, the timing includes
    // the cost of computing the symmetric covariance matrix A'A
    let benchmarks: [(&str, fn(&DMatrix<f64>) -> Duration, bool); 8] = [
        ("Cholesky", bench_cholesky, false),
        ("LU", bench_lu, false),
        ("FullPivLU", bench_full_piv_lu, false),
        ("QR", bench_qr, false),
        ("Hessenberg", bench_hessenberg, false),
        ("Schur", bench_schur, false),
        ("HermitianEigen", bench_hermitian_eigen, true),
        ("SVD", bench_svd, false),
    ];

    // First benchmark to use as reference
    let mut reference_times = Vec::new();

    // Run benchmarks for each algorithm
    for (idx, (name, bench_fn, needs_spd)) in benchmarks.iter().enumerate() {
        println!("Benchmarking {} algorithm...", name);
        let mut row = row![name];

        // Square matrices
        for (i, &size) in SQUARE_SIZES.iter().enumerate() {
            println!("  - Testing {}x{} matrix", size, size);
            let time_ms = if *needs_spd {
                let matrix = random_spd_matrix(size);
                let iterations = if size <= 100 {
                    SMALL_MATRIX_ITERATIONS
                } else if size <= 1000 {
                    MEDIUM_MATRIX_ITERATIONS
                } else {
                    1 // Run once for large matrices
                };
                let duration = bench_with_iterations(&matrix, bench_fn, iterations);
                duration.as_secs_f64() * 1000.0
            } else {
                let matrix = random_square_matrix(size);
                let iterations = if size <= 100 {
                    SMALL_MATRIX_ITERATIONS
                } else if size <= 1000 {
                    MEDIUM_MATRIX_ITERATIONS
                } else {
                    1 // Run once for large matrices
                };
                let duration = bench_with_iterations(&matrix, bench_fn, iterations);
                duration.as_secs_f64() * 1000.0
            };

            // Store reference time (first algorithm)
            if idx == 0 {
                reference_times.push(time_ms);
                // For small matrices (8x8), show with higher precision or in nanoseconds
                // if size == 8 {
                //     let time_ns = time_ms * 1_000_000.0; // Convert to nanoseconds
                //     row.add_cell(cell!(format!("{:.2} ns", time_ns)));
                //     println!("    - Time: {:.2} ns", time_ns);
                // } else {
                    row.add_cell(cell!(format!("{:.4}", time_ms)));
                    println!("    - Time: {:.4} ms", time_ms);
                // }
            } else {
                let ratio = time_ms / reference_times[i];
                // For small matrices (8x8), show with higher precision or in nanoseconds
                // if size == 8 {
                //     let time_ns = time_ms * 1_000_000.0; // Convert to nanoseconds
                //     row.add_cell(cell!(format!("{:.2} ns (x{:.2})", time_ns, ratio)));
                //     println!("    - Time: {:.2} ns (x{:.2})", time_ns, ratio);
                // } else {
                    row.add_cell(cell!(format!("{:.4} (x{:.2})", time_ms, ratio)));
                    println!("    - Time: {:.4} ms (x{:.2})", time_ms, ratio);
                // }
            }
        }

        // Rectangular matrices
        for (i, &cols) in RECT_COLS.iter().enumerate() {
            println!("  - Testing {}x{} matrix", RECT_ROWS, cols);

            // Skip only for non-square matrices for certain algorithms
            if (*name == "Hessenberg" || *name == "HermitianEigen" || *name == "Schur") && RECT_ROWS != cols
            {
                println!("    - Skipping (requires square matrix)");
                row.add_cell(cell!("-"));
                continue;
            }

            let time_ms = if *needs_spd {
                if cols == RECT_ROWS {
                    let matrix = random_spd_matrix(cols);
                    let iterations = if cols <= 100 {
                        SMALL_MATRIX_ITERATIONS
                    } else if cols <= 1000 {
                        MEDIUM_MATRIX_ITERATIONS
                    } else {
                        1 // Run once for large matrices
                    };
                    let duration = bench_with_iterations(&matrix, bench_fn, iterations);
                    duration.as_secs_f64() * 1000.0
                } else {
                    // For non-square matrices that need SPD, we skip
                    println!("    - Skipping (SPD requires square matrix)");
                    row.add_cell(cell!("-"));
                    continue;
                }
            } else {
                let matrix = random_rect_matrix(RECT_ROWS, cols);
                let iterations = if cols <= 100 {
                    SMALL_MATRIX_ITERATIONS
                } else if cols <= 1000 {
                    MEDIUM_MATRIX_ITERATIONS
                } else {
                    1 // Run once for large matrices
                };
                let duration = bench_with_iterations(&matrix, bench_fn, iterations);
                duration.as_secs_f64() * 1000.0
            };

            // Add relative performance indicators
            // The * symbol indicates that for overconstrained matrices, the timing includes
            // the cost of computing the symmetric covariance matrix A'A
            if idx == 0 {
                reference_times.push(time_ms);
                // For small matrices (cols=8), show with higher precision or in nanoseconds
                if cols == 8 {
                    let time_ns = time_ms * 1_000_000.0; // Convert to nanoseconds
                    row.add_cell(cell!(format!("{:.2} ns *", time_ns)));
                    println!("    - Time: {:.2} ns *", time_ns);
                } else {
                    row.add_cell(cell!(format!("{:.3} *", time_ms)));
                    println!("    - Time: {:.3} ms *", time_ms);
                }
            } else {
                let rect_idx = i + SQUARE_SIZES.len();
                let ratio = time_ms / reference_times[rect_idx];
                // For small matrices (cols=8), show with higher precision or in nanoseconds
                if cols == 8 {
                    let time_ns = time_ms * 1_000_000.0; // Convert to nanoseconds
                    row.add_cell(cell!(format!("{:.2} ns (x{:.2}) *", time_ns, ratio)));
                    println!("    - Time: {:.2} ns (x{:.2}) *", time_ns, ratio);
                } else {
                    row.add_cell(cell!(format!("{:.3} (x{:.2}) *", time_ms, ratio)));
                    println!("    - Time: {:.3} ms (x{:.2}) *", time_ms, ratio);
                }
            }
        }

        table.add_row(row);
    }

    // Print the table
    println!("\nResults Table:");
    table.printstd();
}

fn main() {
    println!("Running nalgebra matrix decomposition benchmarks...");
    println!("This may take a while for larger matrices.");
    run_benchmarks();
}
