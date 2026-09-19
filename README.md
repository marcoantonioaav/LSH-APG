# LSB-Tree C++ Library (`liblsbtree`)

A high-performance C++ library implementing the **Locality-Sensitive B-Tree (LSB-Tree)** algorithm for Approximate Nearest Neighbor (ANN) search. 

This repository is refactored from the original LSH-APG codebase to serve as a standalone, modular library for integration into ANN benchmarking suites (e.g., evaluating seed-selection strategies for graph-based ANN algorithms).

---

## Features

- **Standalone C++ Library**: Clean, modular API (`lsb::LSBTree`) decoupling index construction and query execution from graph dependencies.
- **Metric Support**:
  - **Euclidean Distance ($L_2$)**
  - **Cosine Distance ($1 - \cos(\theta)$)** via unit vector normalization.
- **Distance Evaluation Tracking**: Tracks exact distance computation counts per query (`lsb::QueryStats::dist_cmps`), essential for benchmark cost accounting.
- **Structured $k$-NN Results**: Returns approximate nearest neighbor point IDs along with their computed distances (`lsb::Neighbor`).
- **SIMD Accelerated**: Leverages AVX2/SSE vector primitives for fast distance evaluations and bitwise compiler intrinsics for Longest Common Prefix (LLCP) evaluation.
- **Binary Library Output**: Builds both static (`liblsbtree.a`) and shared (`liblsbtree.so`) libraries.

---

## Directory Structure

```text
.
├── cppCode/LSH-APG/
│   ├── include/lsb/
│   │   ├── LSBTree.h       # Main LSBTree library class
│   │   ├── Metric.h        # Distance metric evaluators (Euclidean, Cosine) & tracking
│   │   └── Types.h         # Common data structures (Neighbor, QueryStats, Parameters)
│   ├── src/
│   │   ├── LSBTree.cpp     # Index construction, Z-order coding & LLCP search
│   │   ├── Metric.cpp      # Metric evaluation & vector normalization
│   │   └── fastL2_ip.h     # SIMD distance kernels
│   ├── examples/
│   │   └── benchmark_driver.cpp # Test driver demonstrating library usage
│   └── Makefile            # Library and example build configuration
├── context/                # Original LSB-Tree and LSH-APG papers
└── README.md
```

---

## Compilation

### Prerequisites
- C++17 compatible compiler (`g++` or `clang++`).
- POSIX-compliant environment (Linux).

### Building the Library & Demo
Navigate to `cppCode/LSH-APG` and run `make`:

```bash
cd cppCode/LSH-APG
make clean && make
```

This compiles:
1. `liblsbtree.a` (Static Library)
2. `liblsbtree.so` (Shared Library)
3. `benchmark_driver` (Benchmark / Demo Executable)

### Running the Benchmark Test Driver

```bash
cd cppCode/LSH-APG
LD_LIBRARY_PATH=. ./benchmark_driver
```

---

## Usage & C++ API Example

Below is a complete example demonstrating how to instantiate, build, and query an LSB-Tree index using the library:

```cpp
#include "lsb/LSBTree.h"
#include <iostream>
#include <vector>

int main() {
    // 1. Configure LSB-Tree parameters
    lsb::Parameters params;
    params.L = 10;   // Number of trees / hash tables
    params.K = 4;    // Number of hash functions per tree
    params.W = 1.0f;  // LSH window width

    uint32_t N = 1000;    // Number of data points
    uint32_t dim = 128;   // Vector dimensionality
    uint32_t k = 5;       // k-nearest neighbors

    // Flattened dataset array [N * dim]
    std::vector<float> dataset(N * dim);
    // Fill dataset...

    // 2. Instantiate and build the LSB-Tree index (Euclidean or Cosine)
    lsb::LSBTree index(params, lsb::MetricType::EUCLIDEAN);
    index.fit(dataset.data(), N, dim);

    // 3. Prepare query vector
    std::vector<float> query_point(dim);
    // Fill query_point...

    // 4. Execute k-NN query with stats tracking
    lsb::QueryStats stats;
    std::vector<lsb::Neighbor> results = index.query(query_point.data(), k, &stats);

    // 5. Inspect results
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "Rank " << i + 1 << ": Point ID = " << results[i].id 
                  << ", Distance = " << results[i].distance << "\n";
    }
    std::cout << "Distance Computations: " << stats.dist_cmps << "\n";
    std::cout << "Query Execution Time: " << stats.query_time_ms << " ms\n";

    return 0;
}
```

---

## Index Persistence (Save / Load)

```cpp
// Save index to binary file
index.save("index.bin");

// Load index from binary file
lsb::LSBTree loaded_index;
loaded_index.load("index.bin");
```

---

## References

1. **LSB-Tree**: Yufei Tao, Ke Yi, Cheng Sheng, Panos Kalnis. *"Quality and Efficiency in High Dimensional Nearest Neighbor Search"*. SIGMOD 2009.
2. **LSH-APG**: Xi Zhao, Yao Tian, Kai Huang, Bolong Zheng, Xiaofang Zhou. *"Towards Efficient Index Construction and Approximate Nearest Neighbor Search in High-Dimensional Spaces"*. PVLDB 2023.
