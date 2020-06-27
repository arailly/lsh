//
// Created by Yusuke Arai on 2019/12/02.
//

#ifndef LSH_LSH_HPP
#define LSH_LSH_HPP

#include <vector>
#include <string>
#include <functional>
#include <map>
#include <random>
#include <chrono>
#include <numeric>
#include <arailib.hpp>

using namespace std;
using namespace arailib;

namespace lsh {
    struct VectorHash {
        size_t operator () (const vector<int>& key) const {
            string str;
            for (const auto e : key) str += to_string(e) + ",";
            return hash<string>()(str);
        }
    };

    using HashFunc = function<int(const Data<>&)>;
    using HashFamilyFunc = function<vector<int>(const Data<>&)>;
    using HashTable = unordered_map<vector<int>, vector<int>, VectorHash>;

    struct SearchResult {
        time_t time = 0;
        time_t lsh_time = 0;
        time_t graph_time = 0;
        vector<int> result;
        unsigned long n_bucket_content = 0;
        unsigned long n_node_access = 0;
        unsigned long n_distinct_node_access = 0;
    };

    struct SearchResults {
        vector<SearchResult> results;
        void push_back(const SearchResult& result) { results.push_back(result); }

        void save(const string& log_path, const string& result_path, int k) {
            ofstream log_ofs(log_path);
            string line = "time,n_bucket_content";
            log_ofs << line << endl;

            ofstream result_ofs(result_path);
            line = "query_id,data_id";
            result_ofs << line << endl;

            int query_id = 0;
            for (const auto& result : results) {
                line = to_string(result.time) + "," +
                       to_string(result.n_bucket_content);
                log_ofs << line << endl;

                for (const auto& data_id : result.result) {
                    line = to_string(query_id) + "," + to_string(data_id);
                    result_ofs << line << endl;
                }

                const auto n_miss = k - result.result.size();
                for (int i = 0; i < n_miss; i++) {
                    line = to_string(query_id) + "," + to_string(-1);
                    result_ofs << line << endl;
                }

                query_id++;
            }
        }
    };

    struct LSHIndex {
        const int m, L;
        int dim;
        const DistanceFunction<> distance_function;
        const string distance_type;
        const double r;
        Dataset<> dataset;
        vector<HashFamilyFunc> G;
        vector<HashTable> hash_tables;
        mt19937 engine;

        LSHIndex(int n_hash_func_, double r, int L,
                 string distance = "euclidean", unsigned random_state = 42) :
                m(n_hash_func_), r(r), L(L),
                distance_type(distance), distance_function(select_distance(distance)),
                hash_tables(vector<unordered_map<vector<int>, vector<int>, VectorHash>>(L)),
                engine(random_state) {}

        HashFunc create_hash_func() {
            cauchy_distribution<double> cauchy_dist(0, 1);
            normal_distribution<double> norm_dist(0, 1);
            uniform_real_distribution<double> unif_dist(0, r);

            const auto a = [&]() {
                vector<double> random_vector;
                for (int j = 0; j < dim; j++) {
                    if (distance_type == "manhattan")
                        random_vector.push_back(cauchy_dist(engine));
                    else random_vector.push_back(norm_dist(engine));
                }
                return random_vector;
            }();

            const auto b = unif_dist(engine);

            if (distance_type == "angular") {
                return [=](const Data<>& p) {
                    const auto normalized = normalize(p);
                    const auto ip = inner_product(normalized.begin(), normalized.end(), a.begin(), 0.0);
                    return static_cast<int>((ip + b) / (r * 1.0));
                };
            } else {
                return [=](const Data<>& p) {
                    const auto ip = inner_product(p.begin(), p.end(), a.begin(), 0.0);
                    return static_cast<int>((ip + b) / (r * 1.0));
                };
            }
        }

        HashFamilyFunc create_hash_family() {
            vector<HashFunc> hash_funcs;
            for (int i = 0; i < m; i++) {
                const auto h = create_hash_func();
                hash_funcs.push_back(h);
            }

            return [=](const Data<>& p) {
                vector<int> hash_vector;
                for (const auto& h : hash_funcs) hash_vector.push_back(h(p));
                return hash_vector;
            };
        }

        Data<> normalize(const Data<>& data) const {
            auto normalized = vector<double>(data.size(), 0);
            const auto origin = Data<>(data.id, vector<double>(data.size(), 0));
            const double norm = euclidean_distance(data, origin);
            for (int i = 0; i < data.size(); i++) {
                normalized[i] = data[i] / norm;
            }
            return Data<>(data.id, normalized);
        }

        void insert(const Data<>& data) {
#pragma omp parallel for
            for (int i = 0; i < L; i++) {
                const auto key = G[i](data);
                auto& hash_table = hash_tables[i];
                auto& val = hash_table[key];
                val.emplace_back(data.id);
            }
        }

        void build(const Dataset<>& in_dataset) {
            // set hash function
            dataset = in_dataset;
            dim = dataset[0].size();
            for (int i = 0; i < L; i++) G.push_back(create_hash_family());

            // insert dataset into hash table
            for (auto& data : dataset) insert(data);
        }

        void build(const string& data_path, int n) {
            // insert dataset into hash table
            auto in_dataset = load_data(data_path, n);
            build(in_dataset);
        }

        auto get_bucket_contents(const Data<>& query, int limit = -1) {
            vector<int> result;
            bool is_enough = false;

            for (int i = 0; i < L; i++) {
                HashTable& hash_table = hash_tables[i];
                const auto key = G[i](query);
                for (const auto& data_id : hash_table[key]) {
                    result.emplace_back(data_id);
                    if (limit != -1 && result.size() >= limit) {
                        is_enough = true;
                        break;
                    }
                }
                if (is_enough) break;
            }
            return result;
        }

        auto range_search(const Data<>& query, double range) {
            const auto start = get_now();
            auto result = SearchResult();

            unordered_map<size_t, bool> checked;
            const auto bucket_contents = get_bucket_contents(query);
            result.n_bucket_content += bucket_contents.size();

            for (const auto& data_id : bucket_contents) {
                if (checked[data_id]) continue;
                checked[data_id] = true;
                const auto& data = dataset[data_id];
                if (distance_function(query, data) < range)
                    result.result.emplace_back(data_id);
            }

            const auto end = get_now();
            result.time = get_duration(start, end);
            return result;
        }

        auto knn_search(const Data<>& query, int k) {
            const auto start = get_now();
            auto result = SearchResult();

            unordered_map<size_t, bool> checked;
            multimap<double, int> result_map;

            const auto bucket_contents = get_bucket_contents(query);
            result.n_bucket_content = bucket_contents.size();

            for (const auto& data_id : bucket_contents) {
                if (checked[data_id]) continue;
                checked[data_id] = true;

                const auto& data = dataset[data_id];
                const auto dist = distance_function(query, data);
                result_map.emplace(dist, data_id);

                if (result_map.size() > k) result_map.erase(--result_map.cend());
            }

            for (const auto& pair : result_map) result.result.emplace_back(pair.second);

            const auto end = get_now();
            result.time = get_duration(start, end);
            return result;
        }
    };
}

#endif //LSH_LSH_HPP
