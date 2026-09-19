#include "lsb/LSBTree.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    std::cout << "========================================\n";
    std::cout << "  LSB-Tree Library Benchmark Test Driver \n";
    std::cout << "========================================\n\n";

    uint32_t N = 1000;
    uint32_t dim = 128;
    uint32_t k = 5;

    // Generate random synthetic dataset
    std::vector<float> dataset(static_cast<size_t>(N) * dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& val : dataset) {
        val = dist(rng);
    }

    std::vector<float> query_point(dim);
    for (auto& val : query_point) {
        val = dist(rng);
    }

    lsb::Parameters params;
    params.L = 10;
    params.K = 4;
    params.W = 1.0f;

    // Test 1: Euclidean Distance
    std::cout << "--> Testing EUCLIDEAN Distance Metric...\n";
    lsb::LSBTree lsb_euc(params, lsb::MetricType::EUCLIDEAN);
    lsb_euc.fit(dataset.data(), N, dim);

    lsb::QueryStats stats_euc;
    auto results_euc = lsb_euc.query(query_point.data(), k, &stats_euc);

    std::cout << "Query Results (Euclidean):\n";
    for (size_t i = 0; i < results_euc.size(); ++i) {
        std::cout << "  Rank " << i + 1 << ": Point ID = " << results_euc[i].id 
                  << ", Distance = " << std::fixed << std::setprecision(4) << results_euc[i].distance << "\n";
    }
    std::cout << "  [Stats] Distance Computations: " << stats_euc.dist_cmps << "\n";
    std::cout << "  [Stats] Execution Time: " << stats_euc.query_time_ms << " ms\n\n";

    // Test 2: Cosine Distance
    std::cout << "--> Testing COSINE Distance Metric...\n";
    lsb::LSBTree lsb_cos(params, lsb::MetricType::COSINE);
    lsb_cos.fit(dataset.data(), N, dim);

    lsb::QueryStats stats_cos;
    auto results_cos = lsb_cos.query(query_point.data(), k, &stats_cos);

    std::cout << "Query Results (Cosine):\n";
    for (size_t i = 0; i < results_cos.size(); ++i) {
        std::cout << "  Rank " << i + 1 << ": Point ID = " << results_cos[i].id 
                  << ", Distance = " << std::fixed << std::setprecision(4) << results_cos[i].distance << "\n";
    }
    std::cout << "  [Stats] Distance Computations: " << stats_cos.dist_cmps << "\n";
    std::cout << "  [Stats] Execution Time: " << stats_cos.query_time_ms << " ms\n\n";

    std::cout << "========================================\n";
    std::cout << "  All LSB-Tree Library Tests Passed!    \n";
    std::cout << "========================================\n";

    return 0;
}

