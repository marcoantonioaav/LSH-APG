#pragma once
#include "Types.h"
#include "Metric.h"
#include <vector>
#include <map>
#include <string>
#include <cstdint>
#include <memory>

namespace lsb {

using zint = uint64_t;

struct PosInfo {
    uint32_t table_id = 0;
    int llcp = -1;

    bool operator<(const PosInfo& rhs) const noexcept {
        return llcp < rhs.llcp;
    }
};

class LSBTree {
private:
    Parameters params_;
    MetricType metric_;
    bool is_built_ = false;

    // Projected space variables
    uint32_t S_ = 0; // S = L * K
    int u_ = 0;      // Number of bits per hash value in Z-code

    // Hash random parameters
    std::vector<std::vector<float>> rnd_as_; // [S][dim]
    std::vector<float> rnd_bs_;              // [S]
    std::vector<float> hash_mins_;           // [S]
    std::vector<float> hash_maxs_;           // [S]

    // Dataset reference or copy
    std::vector<float> dataset_storage_;     // Flattened dataset [N * dim] if stored
    const float* dataset_ptr_ = nullptr;     // Pointer to data points

    // Index structure: L hash tables mapping Z-order key to Point ID
    std::vector<std::multimap<zint, uint32_t>> hash_tables_;

private:
    void generate_hash_params();
    void compute_hash_values(std::vector<float>& hashval_flat);
    void normalize_hash_values(std::vector<float>& hashval_flat);
    zint compute_z_code(const float* hashval_chunk);
    int compute_llcp(zint k1, zint k2) const;

public:
    LSBTree() = default;
    LSBTree(const Parameters& params, MetricType metric = MetricType::EUCLIDEAN);
    ~LSBTree() = default;

    // Fits the index using the given dataset pointer [N x dim]
    void fit(const float* dataset, uint32_t N, uint32_t dim, bool is_normalized = false);

    // Queries the k nearest neighbors for query_point
    std::vector<Neighbor> query(const float* query_point, uint32_t k, QueryStats* stats = nullptr) const;

    // Saves the index to a binary file
    bool save(const std::string& filepath) const;

    // Loads the index from a binary file
    bool load(const std::string& filepath);

    const Parameters& get_params() const { return params_; }
    MetricType get_metric() const { return metric_; }
    bool is_built() const { return is_built_; }
};

} // namespace lsb
