use nalgebra::DMatrix;
use prettytable::{cell, format, row, Table};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::f64;
use std::time::{Duration, Instant};

// This benchmark compares different matrix decomposition algorithms in nalgebra.
// For square matrices, the benchmark uses symmetric matrices for SPD-requiring algorithms,
// and random general matrices for others.
// For overconstrained matrices (rectangular), LU, FullPivLU, and Cholesky compute A'A first,
// as denoted by the * symbol in the output.
// Timings are in milliseconds (or nanoseconds for 8-column rectangular), and factors are relative
// to the Cholesky decomposition. Accuracy of reconstruction is also reported.

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
    let a = DMatrix::from_fn(size, size, |_, _| dist.sample(&mut rng));
    let result = &a.transpose() * &a;
    result + DMatrix::identity(size, size) // Add identity to ensure positive definiteness
}

// Function to calculate the Frobenius norm of the difference between two matrices
fn matrix_error(original: &DMatrix<f64>, reconstructed: &DMatrix<f64>) -> f64 {
    if original.shape() != reconstructed.shape() {
        // This can happen if a decomposition for a rectangular matrix A (m,n)
        // reconstructs A'A (n,n) instead of A. In such cases, error calculation is against A'A.
        // If shapes mismatch unexpectedly otherwise, return infinity.
        if original.nrows() == original.ncols()
            && original.shape() == reconstructed.transpose().shape()
        {
            // Placeholder if we ever compare A with (U*S*V')'
        } else {
            // To prevent panic if shapes are fundamentally incompatible for a direct comparison
            // e.g. trying to compare A with reconstructed A'A if 'original' was A.
            // The calling bench_* function should ensure 'original' matches what was reconstructed.
            return f64::INFINITY;
        }
    }
    let diff = original - reconstructed;
    let squared_sum: f64 = diff.iter().map(|&x| x * x).sum();
    squared_sum.sqrt()
}

// Benchmark functions for each decomposition algorithm
// Each function returns both the duration and the reconstruction error

fn bench_cholesky(matrix: &DMatrix<f64>) -> (Duration, f64) {
    let start = Instant::now();
    if matrix.nrows() != matrix.ncols() {
        let a_transpose_a = matrix.transpose() * matrix; // matrix consumed
        let chol_result = a_transpose_a.clone().cholesky();
        let duration = start.elapsed();
        let error = if let Some(l_factor) = chol_result {
            let reconstructed = l_factor.l() * l_factor.l().transpose();
            matrix_error(&a_transpose_a, &reconstructed)
        } else {
            f64::INFINITY
        };
        (duration, error)
    } else {
        let chol_result = matrix.clone().cholesky();
        let duration = start.elapsed();
        let error = if let Some(l_factor) = chol_result {
            let reconstructed = l_factor.l() * l_factor.l().transpose();
            matrix_error(matrix, &reconstructed)
        } else {
            f64::INFINITY
        };
        (duration, error)
    }
}

fn bench_qr(matrix: &DMatrix<f64>) -> (Duration, f64) {
    let start = Instant::now();
    let qr = matrix.clone().qr();
    let duration = start.elapsed();

    let q = qr.q();
    let r = qr.r();
    let reconstructed = q * r;
    let error = matrix_error(matrix, &reconstructed);
    (duration, error)
}

fn bench_lu(matrix: &DMatrix<f64>) -> (Duration, f64) {
    let start = Instant::now();
    if matrix.nrows() != matrix.ncols() {
        let a_transpose_a = matrix.transpose() * matrix; // matrix is consumed here
        let lu_result = a_transpose_a.clone().lu();
        let duration = start.elapsed();
        let (_, l, u) = lu_result.unpack();
        let reconstructed = l * u; // removed transpose() call on p
        let error = matrix_error(&a_transpose_a, &reconstructed);
        (duration, error)
    } else {
        let lu_result = matrix.clone().lu();
        let duration = start.elapsed();
        let (_, l, u) = lu_result.unpack();
        let reconstructed = l * u; // removed transpose() call on p
        let error = matrix_error(matrix, &reconstructed);
        (duration, error)
    }
}

fn bench_full_piv_lu(matrix: &DMatrix<f64>) -> (Duration, f64) {
    let start = Instant::now();
    if matrix.nrows() != matrix.ncols() {
        let a_transpose_a = matrix.transpose() * matrix; // matrix is consumed
        let lu_result = a_transpose_a.clone().full_piv_lu();
        let duration = start.elapsed();
        let (_, l, u, _) = lu_result.unpack();
        let reconstructed = l * u;
        let error = matrix_error(&a_transpose_a, &reconstructed);
        (duration, error)
    } else {
        let lu_result = matrix.clone().full_piv_lu();
        let duration = start.elapsed();
        let (_, l, u, _) = lu_result.unpack();
        let reconstructed = l * u;
        let error = matrix_error(matrix, &reconstructed);
        (duration, error)
    }
}

fn bench_hessenberg(matrix: &DMatrix<f64>) -> (Duration, f64) {
    let start = Instant::now();
    let hess = matrix.clone().hessenberg();
    let duration = start.elapsed();
    let q = hess.q();
    let h = hess.h();
    let reconstructed = &q * h * q.transpose();
    let error = matrix_error(matrix, &reconstructed);
    (duration, error)
}

fn bench_schur(matrix: &DMatrix<f64>) -> (Duration, f64) {
    let start = Instant::now();
    let schur_decomp = matrix.clone().schur();
    let duration = start.elapsed();
    // let reconstructed = schur_decomp.recompose();
    let (q, t) = schur_decomp.unpack();
    let reconstructed = q.clone() * t.clone() * q.clone().transpose();
    let error = matrix_error(matrix, &reconstructed);
    (duration, error)
}

fn bench_hermitian_eigen(matrix: &DMatrix<f64>) -> (Duration, f64) {
    let start = Instant::now();
    let eigen = matrix.clone().symmetric_eigen();
    let duration = start.elapsed();

    // let q = eigen.eigenvectors;
    // let d_values = eigen.eigenvalues;
    // let d_matrix = DMatrix::from_diagonal(&d_values);
    // let reconstructed = &q * d_matrix * q.transpose();
    let reconstructed = eigen.recompose();
    let error = matrix_error(matrix, &reconstructed);
    (duration, error)
}

fn bench_svd(matrix: &DMatrix<f64>) -> (Duration, f64) {
    let start = Instant::now();
    let svd_decomp = matrix.clone().svd(true, true);
    let duration = start.elapsed();

    let error = if let (Some(u), Some(v_t)) = (svd_decomp.u, svd_decomp.v_t) {
        // Ensure s is conformable for multiplication: (nrows, ncols)
        // For A (m,n), U is (m,k), S is (k,k) diagonal, V^T is (k,n) where k = min(m,n)
        // We need S as a (k,k) diagonal matrix.
        let s_diag = DMatrix::from_diagonal(&svd_decomp.singular_values);

        // U is (m,k), s_diag is (k,k), v_t is (k,n)
        // reconstructed = U * S_diag * V_t
        let reconstructed = u * s_diag * v_t;
        matrix_error(matrix, &reconstructed)
    } else {
        f64::INFINITY
    };
    (duration, error)
}

// Benchmark with multiple iterations for more stable results
fn bench_with_iterations<F>(
    matrix: &DMatrix<f64>,
    bench_fn: F,
    iterations: usize,
) -> (Duration, f64)
where
    F: Fn(&DMatrix<f64>) -> (Duration, f64),
{
    let mut min_duration = Duration::new(u64::MAX, 1_000_000_000 - 1); // Initialize to max duration
    let mut error_at_min_duration = f64::INFINITY;

    for _ in 0..iterations {
        let (duration, error) = bench_fn(matrix);
        if duration < min_duration {
            min_duration = duration;
            error_at_min_duration = error;
        }
    }
    (min_duration, error_at_min_duration)
}

fn run_benchmarks() {
    const SMALL_MATRIX_ITERATIONS: usize = 10;
    const MEDIUM_MATRIX_ITERATIONS: usize = 5;

    // Create two tables - one for timing and one for accuracy
    let mut time_table = Table::new();
    let mut accuracy_table = Table::new();
    time_table.set_format(*format::consts::FORMAT_BOX_CHARS);
    accuracy_table.set_format(*format::consts::FORMAT_BOX_CHARS);

    // Create headers for both tables
    let mut time_header = row!["solver/size"];
    let mut accuracy_header = row!["solver/size"];
    
    for size in SQUARE_SIZES.iter() {
        let header_text = format!("{0}x{0}", size);
        time_header.add_cell(cell!(header_text.clone()));
        accuracy_header.add_cell(cell!(header_text));
    }
    
    for col in RECT_COLS.iter() {
        let header_text = format!("{0}x{1}", RECT_ROWS, col);
        time_header.add_cell(cell!(header_text.clone()));
        accuracy_header.add_cell(cell!(header_text));
    }
    
    time_table.add_row(time_header);
    accuracy_table.add_row(accuracy_header);

    let benchmarks: [(&str, fn(&DMatrix<f64>) -> (Duration, f64), bool); 8] = [
        ("Cholesky", bench_cholesky, true),
        ("LU", bench_lu, false),
        ("FullPivLU", bench_full_piv_lu, false),
        ("QR", bench_qr, false),
        ("Hessenberg", bench_hessenberg, false), // For square matrices
        ("Schur", bench_schur, false),           // For square matrices
        ("HermitianEigen", bench_hermitian_eigen, true), // Needs SPD matrix
        ("SVD", bench_svd, false),
    ];

    let mut reference_times = Vec::new();
    let mut time_rows = Vec::new();
    let mut accuracy_rows = Vec::new();

    // Initialize rows for both tables
    for (name, _, _) in benchmarks.iter() {
        time_rows.push(row![name]);
        accuracy_rows.push(row![name]);
    }

    for (idx, (name, bench_fn, needs_spd)) in benchmarks.iter().enumerate() {
        println!("Benchmarking {} algorithm...", name);

        // Square matrices
        for (i, &size) in SQUARE_SIZES.iter().enumerate() {
            println!("  - Testing {}x{} matrix", size, size);

            let (time_ms, error) = if *needs_spd {
                let matrix = random_spd_matrix(size);
                let iterations = if size <= 100 {
                    SMALL_MATRIX_ITERATIONS
                } else if size <= 1000 {
                    MEDIUM_MATRIX_ITERATIONS
                } else {
                    1
                };
                let (duration, err_val) = bench_with_iterations(&matrix, *bench_fn, iterations);
                (duration.as_secs_f64() * 1000.0, err_val)
            } else {
                let matrix = random_square_matrix(size);
                let iterations = if size <= 100 {
                    SMALL_MATRIX_ITERATIONS
                } else if size <= 1000 {
                    MEDIUM_MATRIX_ITERATIONS
                } else {
                    1
                };
                let (duration, err_val) = bench_with_iterations(&matrix, *bench_fn, iterations);
                (duration.as_secs_f64() * 1000.0, err_val)
            };

            // Add to console output (keep the same format as before)
            if idx == 0 {
                reference_times.push(time_ms);
                println!("    - Time: {:.4} ms - Accuracy: {:.2e}", time_ms, error);
            } else {
                let ratio = if reference_times[i].is_nan() {
                    f64::NAN
                } else {
                    time_ms / reference_times[i]
                };
                if reference_times[i].is_nan() || time_ms.is_nan() {
                    println!(
                        "    - Time: {:.4} ms (xNaN) - Accuracy: {:.2e}",
                        time_ms, error
                    );
                } else {
                    println!(
                        "    - Time: {:.4} ms (x{:.2}) - Accuracy: {:.2e}",
                        time_ms, ratio, error
                    );
                }
            }

            // Add to time table
            if idx == 0 {
                time_rows[idx].add_cell(cell!(format!("{:.4}", time_ms)));
            } else {
                let ratio = if reference_times[i].is_nan() {
                    f64::NAN
                } else {
                    time_ms / reference_times[i]
                };
                if reference_times[i].is_nan() || time_ms.is_nan() {
                    time_rows[idx].add_cell(cell!(format!("{:.4} (xNaN)", time_ms)));
                } else {
                    time_rows[idx].add_cell(cell!(format!("{:.4} (x{:.2})", time_ms, ratio)));
                }
            }

            // Add to accuracy table (no ratios)
            accuracy_rows[idx].add_cell(cell!(format!("{:.2e}", error)));
        }

        // Rectangular matrices
        for (j, &cols) in RECT_COLS.iter().enumerate() {
            let current_rect_ref_idx = SQUARE_SIZES.len() + j;
            println!("  - Testing {}x{} matrix", RECT_ROWS, cols);

            if (*name == "Hessenberg" || *name == "Schur" || *name == "HermitianEigen")
                && RECT_ROWS != cols
            {
                println!("    - Skipping ({} requires square matrix)", name);
                time_rows[idx].add_cell(cell!("-"));
                accuracy_rows[idx].add_cell(cell!("-"));
                if idx == 0 {
                    reference_times.push(f64::NAN);
                }
                continue;
            }

            let (time_ms, error) = if *needs_spd {
                // Only HermitianEigen uses this path for rect if RECT_ROWS == cols
                // e.g. if we had a 10000x10000 case for SPD
                let matrix = random_spd_matrix(cols);
                let iterations = if cols <= 100 {
                    SMALL_MATRIX_ITERATIONS
                } else if cols <= 1000 {
                    MEDIUM_MATRIX_ITERATIONS
                } else {
                    1
                };
                let (duration, err_val) = bench_with_iterations(&matrix, *bench_fn, iterations);
                (duration.as_secs_f64() * 1000.0, err_val)

            } else {
                // Cholesky, LU, FullPivLU, QR, SVD can run on rectangular
                // Cholesky, LU, FullPivLU will internally handle A'A for rectangular
                let matrix = random_rect_matrix(RECT_ROWS, cols);
                let iterations = if cols <= 100 {
                    SMALL_MATRIX_ITERATIONS
                } else if cols <= 1000 {
                    MEDIUM_MATRIX_ITERATIONS
                } else {
                    1
                };
                let (duration, err_val) = bench_with_iterations(&matrix, *bench_fn, iterations);
                (duration.as_secs_f64() * 1000.0, err_val)
            };

            let asterisk = if *name == "Cholesky" || *name == "LU" || *name == "FullPivLU" {
                " *"
            } else {
                ""
            };

            // Add to console output (keep the same format as before)
            if idx == 0 {
                reference_times.push(time_ms);
                println!(
                    "    - Time: {:.3} ms{} - Accuracy: {:.2e}",
                    time_ms, asterisk, error
                );
            } else {
                let ratio = if reference_times[current_rect_ref_idx].is_nan() {
                    f64::NAN
                } else {
                    time_ms / reference_times[current_rect_ref_idx]
                };
                let ratio_str = if ratio.is_nan() {
                    "xNaN".to_string()
                } else {
                    format!("x{:.2}", ratio)
                };
                println!(
                    "    - Time: {:.3} ms ({}){} - Accuracy: {:.2e}",
                    time_ms, ratio_str, asterisk, error
                );
            }

            // Add to time table
            if idx == 0 {
                time_rows[idx].add_cell(cell!(format!("{:.3}{}", time_ms, asterisk)));
            } else {
                let ratio = if reference_times[current_rect_ref_idx].is_nan() {
                    f64::NAN
                } else {
                    time_ms / reference_times[current_rect_ref_idx]
                };
                let ratio_str = if ratio.is_nan() {
                    "xNaN".to_string()
                } else {
                    format!("x{:.2}", ratio)
                };
                time_rows[idx].add_cell(cell!(format!("{:.3} ({}){}", time_ms, ratio_str, asterisk)));
            }

            // Add to accuracy table (no ratios)
            accuracy_rows[idx].add_cell(cell!(format!("{:.2e}{}", error, asterisk)));
        }
    }

    // Add all rows to tables
    for i in 0..benchmarks.len() {
        time_table.add_row(time_rows[i].clone());
        accuracy_table.add_row(accuracy_rows[i].clone());
    }

    // Print both tables
    println!("\nTiming Results Table:");
    time_table.printstd();
    
    println!("\nAccuracy Results Table:");
    accuracy_table.printstd();
}

fn main() {
    println!("Running nalgebra matrix decomposition benchmarks...");
    println!("This may take a while for larger matrices.");
    println!("Reference algorithm for ratios is Cholesky.");
    run_benchmarks();
}
