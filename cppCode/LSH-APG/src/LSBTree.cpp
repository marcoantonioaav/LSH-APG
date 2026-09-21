#include "lsb/LSBTree.h"
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <queue>
#include <chrono>
#include <cfloat>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace lsb {

LSBTree::LSBTree(const Parameters& params, MetricType metric)
    : params_(params), metric_(metric), is_built_(false) {
    S_ = params_.L * params_.K;
}

void LSBTree::generate_hash_params() {
    rnd_as_.assign(S_, std::vector<float>(params_.dim));
    rnd_bs_.resize(S_);

    std::mt19937 rng(0); // Deterministic seed for reproducible index generation
    std::uniform_real_distribution<float> ur(0.0f, params_.W);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (uint32_t j = 0; j < S_; ++j) {
        for (uint32_t i = 0; i < params_.dim; ++i) {
            rnd_as_[j][i] = nd(rng);
        }
        rnd_bs_[j] = ur(rng);
    }
}

void LSBTree::compute_hash_values(std::vector<float>& hashval_flat) {
    hashval_flat.resize(static_cast<size_t>(params_.N) * S_);
    hash_mins_.assign(S_, FLT_MAX);
    hash_maxs_.assign(S_, -FLT_MAX);

    for (uint32_t j = 0; j < params_.N; ++j) {
        const float* point = dataset_ptr_ + j * params_.dim;
        for (uint32_t i = 0; i < S_; ++i) {
            float ip = Metric::inner_product(point, rnd_as_[i].data(), params_.dim);
            float h_val = (ip + rnd_bs_[i]) / params_.W;
            hashval_flat[static_cast<size_t>(j) * S_ + i] = h_val;

            if (h_val < hash_mins_[i]) hash_mins_[i] = h_val;
            if (h_val > hash_maxs_[i]) hash_maxs_[i] = h_val;
        }
    }
}

void LSBTree::normalize_hash_values(std::vector<float>& hashval_flat) {
    int hMax = 1 << (64 / params_.K);
    float rMax = -1.0f;
    std::vector<float> ranges(S_);

    for (uint32_t i = 0; i < S_; ++i) {
        float b = std::floor(hash_mins_[i]);
        ranges[i] = hash_maxs_[i] - b;
        if (rMax < ranges[i]) rMax = ranges[i];
    }

    float oldW = params_.W;
    float r = 1.0f;

    if (rMax > static_cast<float>(hMax)) {
        --hMax;
        r = rMax / static_cast<float>(hMax);
        params_.W *= r;
        rMax = static_cast<float>(hMax);
    }

    for (uint32_t i = 0; i < S_; ++i) {
        float b = -std::floor(hash_mins_[i]);
        hash_maxs_[i] = (hash_maxs_[i] + b) / r;
        hash_mins_[i] = (hash_mins_[i] + b) / r;
        rnd_bs_[i] += b * oldW;

        for (uint32_t j = 0; j < params_.N; ++j) {
            size_t idx = static_cast<size_t>(j) * S_ + i;
            hashval_flat[idx] = (hashval_flat[idx] + b) / r;
        }
    }

    u_ = static_cast<int>(std::floor(std::log2(std::max(1.0f, rMax)))) + 1;
}

zint LSBTree::compute_z_code(const float* hashval_chunk) {
    zint res = 0;
    for (int i = u_ - 1; i >= 0; i--) {
        int mask = 1 << i;
        for (uint32_t j = 0; j < params_.K; j++) {
            res <<= 1;
            if (static_cast<int>(std::floor(hashval_chunk[j])) & mask) {
                ++res;
            }
        }
    }
    return res;
}

int LSBTree::compute_llcp(zint k1, zint k2) const {
    if (k1 == k2) return 64;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(k1 ^ k2);
#elif defined(_MSC_VER)
    return static_cast<int>(_lzcnt_u64(k1 ^ k2));
#else
    zint diff = k1 ^ k2;
    int count = 0;
    while ((diff & (1ULL << 63)) == 0) {
        count++;
        diff <<= 1;
    }
    return count;
#endif
}

void LSBTree::fit(const float* dataset, uint32_t N, uint32_t dim, bool is_normalized) {
    params_.N = N;
    params_.dim = dim;
    S_ = params_.L * params_.K;

    if (metric_ == MetricType::COSINE && !is_normalized) {
        dataset_storage_.resize(static_cast<size_t>(N) * dim);
        for (uint32_t i = 0; i < N; ++i) {
            Metric::normalize(dataset_storage_.data() + i * dim, dataset + i * dim, dim);
        }
        dataset_ptr_ = dataset_storage_.data();
    } else {
        dataset_ptr_ = dataset;
    }

    generate_hash_params();

    std::vector<float> hashval_flat;
    compute_hash_values(hashval_flat);
    normalize_hash_values(hashval_flat);

    hash_tables_.clear();
    hash_tables_.resize(params_.L);

    for (uint32_t j = 0; j < params_.L; ++j) {
        for (uint32_t i = 0; i < params_.N; ++i) {
            zint key = compute_z_code(hashval_flat.data() + static_cast<size_t>(i) * S_ + j * params_.K);
            hash_tables_[j].insert({ key, i });
        }
    }

    is_built_ = true;
}

std::vector<Neighbor> LSBTree::query(const float* query_point, uint32_t k, QueryStats* stats, uint32_t max_candidates) const {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (stats) {
        stats->dist_cmps = 0;
        stats->query_time_ms = 0.0;
    }

    if (!is_built_ || params_.N == 0) {
        return {};
    }

    // Process query point (normalize if COSINE)
    std::vector<float> norm_q_storage;
    const float* q_ptr = query_point;
    if (metric_ == MetricType::COSINE) {
        norm_q_storage.resize(params_.dim);
        Metric::normalize(norm_q_storage.data(), query_point, params_.dim);
        q_ptr = norm_q_storage.data();
    }

    // Compute hash values for query point
    std::vector<float> q_hashval(S_);
    for (uint32_t i = 0; i < S_; ++i) {
        float ip = Metric::inner_product(q_ptr, rnd_as_[i].data(), params_.dim);
        q_hashval[i] = (ip + rnd_bs_[i]) / params_.W;
    }

    std::vector<bool> visited(params_.N, false);
    std::vector<Neighbor> candidates;

    uint32_t UB = (max_candidates > 0) ? std::min(params_.N, max_candidates)
                                       : std::min(params_.N, static_cast<uint32_t>(params_.N / 10 + k));
    int step = 100;

    std::vector<std::multimap<zint, uint32_t>::const_iterator> lpos(params_.L), rpos(params_.L), qpos(params_.L);
    std::priority_queue<PosInfo> lEntries, rEntries;

    for (uint32_t j = 0; j < params_.L; ++j) {
        zint key = const_cast<LSBTree*>(this)->compute_z_code(q_hashval.data() + j * params_.K);
        qpos[j] = hash_tables_[j].lower_bound(key);

        if (qpos[j] != hash_tables_[j].begin()) {
            lpos[j] = qpos[j];
            --lpos[j];
            lEntries.push({ j, compute_llcp(lpos[j]->first, qpos[j] != hash_tables_[j].end() ? qpos[j]->first : key) });
        }

        rpos[j] = qpos[j];
        if (rpos[j] != hash_tables_[j].end()) {
            rEntries.push({ j, compute_llcp(rpos[j]->first, qpos[j]->first) });
        }
    }

    size_t dist_count = 0;

    while (!(lEntries.empty() && rEntries.empty())) {
        PosInfo t;
        bool move_left = true;

        if (lEntries.empty()) move_left = false;
        else if (rEntries.empty()) move_left = true;
        else if (rEntries.top().llcp > lEntries.top().llcp) move_left = false;

        if (move_left) {
            t = lEntries.top();
            lEntries.pop();
            for (int i = 0; i < step; ++i) {
                uint32_t point_id = lpos[t.table_id]->second;
                if (!visited[point_id]) {
                    visited[point_id] = true;
                    float dist = Metric::compute_distance(q_ptr, dataset_ptr_ + point_id * params_.dim, params_.dim, metric_, &dist_count);
                    candidates.push_back({ point_id, dist });
                }

                if (lpos[t.table_id] != hash_tables_[t.table_id].begin()) {
                    --lpos[t.table_id];
                } else {
                    break;
                }
            }

            if (lpos[t.table_id] != hash_tables_[t.table_id].begin()) {
                t.llcp = compute_llcp(lpos[t.table_id]->first, qpos[t.table_id] != hash_tables_[t.table_id].end() ? qpos[t.table_id]->first : 0);
                lEntries.push(t);
            }
        } else {
            t = rEntries.top();
            rEntries.pop();
            for (int i = 0; i < step; ++i) {
                uint32_t point_id = rpos[t.table_id]->second;
                if (!visited[point_id]) {
                    visited[point_id] = true;
                    float dist = Metric::compute_distance(q_ptr, dataset_ptr_ + point_id * params_.dim, params_.dim, metric_, &dist_count);
                    candidates.push_back({ point_id, dist });
                }

                if (++rpos[t.table_id] == hash_tables_[t.table_id].end()) {
                    break;
                }
            }

            if (rpos[t.table_id] != hash_tables_[t.table_id].end()) {
                t.llcp = compute_llcp(rpos[t.table_id]->first, qpos[t.table_id] != hash_tables_[t.table_id].end() ? qpos[t.table_id]->first : 0);
                rEntries.push(t);
            }
        }

        if (candidates.size() >= UB) break;
    }

    std::sort(candidates.begin(), candidates.end());

    std::vector<Neighbor> results;
    if (candidates.size() <= k) {
        results = candidates;
    } else {
        results.assign(candidates.begin(), candidates.begin() + k);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->dist_cmps = dist_count;
        stats->query_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

    return results;
}

bool LSBTree::save(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.good()) return false;

    out.write(reinterpret_cast<const char*>(&params_.N), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&params_.dim), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&params_.L), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&params_.K), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&params_.W), sizeof(float));
    out.write(reinterpret_cast<const char*>(&u_), sizeof(int));

    int metric_val = static_cast<int>(metric_);
    out.write(reinterpret_cast<const char*>(&metric_val), sizeof(int));

    for (uint32_t i = 0; i < S_; ++i) {
        out.write(reinterpret_cast<const char*>(rnd_as_[i].data()), sizeof(float) * params_.dim);
    }
    out.write(reinterpret_cast<const char*>(rnd_bs_.data()), sizeof(float) * S_);

    for (uint32_t i = 0; i < params_.L; ++i) {
        size_t size = hash_tables_[i].size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        for (const auto& kv : hash_tables_[i]) {
            out.write(reinterpret_cast<const char*>(&kv.first), sizeof(zint));
            out.write(reinterpret_cast<const char*>(&kv.second), sizeof(uint32_t));
        }
    }
    return true;
}

bool LSBTree::load(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.good()) return false;

    in.read(reinterpret_cast<char*>(&params_.N), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&params_.dim), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&params_.L), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&params_.K), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&params_.W), sizeof(float));
    in.read(reinterpret_cast<char*>(&u_), sizeof(int));

    int metric_val;
    in.read(reinterpret_cast<char*>(&metric_val), sizeof(int));
    metric_ = static_cast<MetricType>(metric_val);

    S_ = params_.L * params_.K;

    rnd_as_.assign(S_, std::vector<float>(params_.dim));
    rnd_bs_.resize(S_);

    for (uint32_t i = 0; i < S_; ++i) {
        in.read(reinterpret_cast<char*>(rnd_as_[i].data()), sizeof(float) * params_.dim);
    }
    in.read(reinterpret_cast<char*>(rnd_bs_.data()), sizeof(float) * S_);

    hash_tables_.clear();
    hash_tables_.resize(params_.L);

    for (uint32_t i = 0; i < params_.L; ++i) {
        size_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size_t));
        for (size_t j = 0; j < size; ++j) {
            zint key;
            uint32_t point_id;
            in.read(reinterpret_cast<char*>(&key), sizeof(zint));
            in.read(reinterpret_cast<char*>(&point_id), sizeof(uint32_t));
            hash_tables_[i].insert({ key, point_id });
        }
    }

    is_built_ = true;
    return true;
}

} // namespace lsb
