#!/bin/bash

# Run nalgebra matrix decomposition benchmarks

echo "Building release version..."
cargo build --release

echo "Starting benchmarks (this may take a while)..."
time ./target/release/nalgebra-benchmark | tee benchmark_results.txt

echo "Benchmark completed! Results saved to benchmark_results.txt" 