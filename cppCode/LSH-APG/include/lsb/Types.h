#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace lsb {

enum class MetricType {
    EUCLIDEAN = 0,
    COSINE = 1
};

struct Neighbor {
    uint32_t id;
    float distance;

    bool operator<(const Neighbor& rhs) const noexcept {
        return distance < rhs.distance;
    }

    bool operator>(const Neighbor& rhs) const noexcept {
        return distance > rhs.distance;
    }

    bool operator==(const Neighbor& rhs) const noexcept {
        return id == rhs.id;
    }
};

struct QueryStats {
    size_t dist_cmps = 0;      // Number of distance computations performed
    double query_time_ms = 0.0; // Total query execution time in milliseconds
};

struct Parameters {
    uint32_t N = 0;          // Number of data points
    uint32_t dim = 0;        // Dimension of data points
    uint32_t L = 10;         // Number of LSH hash tables / trees
    uint32_t K = 10;         // Number of hash functions per table
    float W = 1.0f;          // LSH window size / bin width
};

} // namespace lsb

