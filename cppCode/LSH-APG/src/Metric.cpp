#include "lsb/Metric.h"
#include "fastL2_ip.h"
#include <cmath>

namespace lsb {

float Metric::l2_sqr(const float* v1, const float* v2, uint32_t dim) {
    return calL2Sqr_fast(const_cast<float*>(v1), const_cast<float*>(v2), static_cast<int>(dim));
}

float Metric::inner_product(const float* v1, const float* v2, uint32_t dim) {
    return calIp_fast(const_cast<float*>(v1), const_cast<float*>(v2), static_cast<int>(dim));
}

float Metric::vector_norm(const float* v, uint32_t dim) {
    float ip = inner_product(v, v, dim);
    return std::sqrt(std::max(0.0f, ip));
}

void Metric::normalize(float* dst, const float* src, uint32_t dim) {
    float norm = vector_norm(src, dim);
    if (norm > 1e-12f) {
        float inv_norm = 1.0f / norm;
        for (uint32_t i = 0; i < dim; ++i) {
            dst[i] = src[i] * inv_norm;
        }
    } else {
        for (uint32_t i = 0; i < dim; ++i) {
            dst[i] = 0.0f;
        }
    }
}

float Metric::compute_distance(const float* v1, const float* v2, uint32_t dim, MetricType metric, size_t* cmps_counter) {
    if (cmps_counter) {
        (*cmps_counter)++;
    }

    if (metric == MetricType::COSINE) {
        float ip = inner_product(v1, v2, dim);
        float norm1 = vector_norm(v1, dim);
        float norm2 = vector_norm(v2, dim);
        if (norm1 < 1e-12f || norm2 < 1e-12f) {
            return 1.0f; // Maximum cosine distance
        }
        float cos_sim = ip / (norm1 * norm2);
        // Clamp cosine similarity to [-1, 1]
        cos_sim = std::max(-1.0f, std::min(1.0f, cos_sim));
        return 1.0f - cos_sim;
    } else {
        // EUCLIDEAN
        float sqr_dist = l2_sqr(v1, v2, dim);
        return std::sqrt(std::max(0.0f, sqr_dist));
    }
}

} // namespace lsb
