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

    struct SearchResult {
        Series series;
        time_t time = 0;
        size_t n_bucket_content = 0;
    };

    typedef function<int(const Point&)> HashFunc;
    typedef function<vector<int>(const Point&)> HashFamilyFunc;

    struct LSHIndex {
        const int k, d, L;
        const float r;
        vector<HashFamilyFunc> G;
        vector<unordered_multimap<vector<int>, Point, VectorHash>> hash_tables;
        mt19937 engine;

        LSHIndex(int k, float r, int d, int L, unsigned random_state = 42) :
            k(k), r(r), d(d), L(L),
            hash_tables(vector<unordered_multimap<vector<int>, Point, VectorHash>>(L)),
            engine(mt19937(random_state)) {}

        HashFunc create_hash_func() {
            normal_distribution<float> norm_dist(0.0, 1.0);
            uniform_real_distribution<float> unif_dist(0, r);

            const auto a = [&]() {
                vector<float> random_vector;
                for (int j = 0; j < d; j++) random_vector.push_back(norm_dist(engine));
                return random_vector;
            }();

            const auto b = unif_dist(engine);

            return [=](const Point& p) {
                const auto ip = inner_product(p.begin(), p.end(), a.begin(), 0);
                return static_cast<int>((ip + b) / (r * 1.0));
            };
        }

        HashFamilyFunc create_hash_family() {
            vector<HashFunc> hash_funcs;
            for (int i = 0; i < k; i++) {
                const auto h = create_hash_func();
                hash_funcs.push_back(h);
            }

            return [=](const Point& p) {
                vector<int> hash_vector;
                for (const auto& h : hash_funcs) hash_vector.push_back(h(p));
                return hash_vector;
            };
        }

        void build(Series& series) {
            // set hash function
            for (int i = 0; i < L; i++) G.push_back(create_hash_family());

            // insert series into hash table
#pragma omp parallel for
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < series.size(); j++) {
                    auto point = series[j];
                    const auto hash_value = G[i](point);
                    hash_tables[i].emplace(hash_value, move(point));
                }
            }
        }

        void build(const string& data_path, int n) {
            // insert series into hash table
            auto series = load_data(data_path, n);
            build(series);
        }

        Series find(const vector<int> key,
                    const unordered_multimap<vector<int>, Point, VectorHash>&
                    hash_table) const {
            Series result;
            auto bucket_itr = hash_table.equal_range(key);
            for (auto itr = bucket_itr.first; itr != bucket_itr.second; itr++) {
                result.push_back(itr->second);
            }
            return result;
        }

        SearchResult search(const Point& query, float range) const {
            const auto start = get_now();
            auto result = SearchResult();

            unordered_map<size_t, bool> checked;
            for (int i = 0; i < L; i++) {
                const auto key = G[i](query);
                const auto bucket_contents = find(key, hash_tables[i]);
                result.n_bucket_content += bucket_contents.size();

                for (const auto point : bucket_contents) {
                    if (checked[point.id]) continue;
                    checked[point.id] = true;
                    if (euclidean_distance(query, point) < range)
                        result.series.push_back(point);
                }
            }

            const auto end = get_now();
            result.time = get_duration(start, end);
            return result;
        }
    };
}

#endif //LSH_LSH_HPP
