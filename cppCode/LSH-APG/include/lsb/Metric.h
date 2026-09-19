#pragma once
#include "Types.h"
#include <cmath>
#include <vector>

namespace lsb {

class Metric {
public:
    // Calculates distance between v1 and v2 based on metric type.
    // If cmps_counter is provided, increments distance evaluation count by 1.
    static float compute_distance(const float* v1, const float* v2, uint32_t dim, MetricType metric, size_t* cmps_counter = nullptr);

    // Calculates inner product (dot product) of v1 and v2
    static float inner_product(const float* v1, const float* v2, uint32_t dim);

    // Calculates squared L2 distance between v1 and v2
    static float l2_sqr(const float* v1, const float* v2, uint32_t dim);

    // Calculates vector norm (magnitude)
    static float vector_norm(const float* v, uint32_t dim);

    // Normalizes vector src into dst (dst = src / ||src||)
    static void normalize(float* dst, const float* src, uint32_t dim);
};

} // namespace lsb
