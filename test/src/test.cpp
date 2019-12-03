#include <gtest/gtest.h>
#include <random>
#include <arailib.hpp>
#include <lsh.hpp>

using namespace std;
using namespace arailib;
using namespace lsh;

TEST(lsh, create_hash_func) {
    const int k = 2, L = 1, d = 2, r = 1;
    auto index = LSHIndex(k, r, d);
    auto hash_func = index.create_hash_family();
    const Point p(0, {1, 2});
    const auto hash_vector = hash_func(p);
    ASSERT_EQ(hash_vector.size(), k);
}

TEST(lsh, find) {
    const int k = 2, d = 128, r = 1, n = 1;
    const string data_path = "/Users/yusuke-arai/workspace/dataset/sift/sift_base/";
    const auto series = load_data(data_path, n);
    auto series_for_index = series;

    auto index = LSHIndex(k, r, d);
    index.build(series_for_index);

    for (int i = 0; i < 10; i++) {
        bool include_self = false;
        const auto find_result = index.find(series[i]);
        for (const auto e : find_result) {
            if (e.id == i) include_self = true;
        }
        ASSERT_TRUE(include_self);
    }
}

TEST(lsh, search) {
    const int k = 2, d = 128, r = 250, n = 3;
    float range = 400;
    const string data_path = "/Users/yusuke-arai/workspace/dataset/sift/sift_base/";
    const auto series = load_data(data_path, n);
    auto series_for_index = series;

    auto index = LSHIndex(k, r, d);
    index.build(series_for_index);

    for (const auto i : vector<int>{12, 21, 22, 23, 30, 40}) {
        const auto result = index.search(series[i], range);
        ASSERT_GT(result.size(), 0);
    }
}

TEST(s, s) {
    mt19937 engine(42);
    normal_distribution<float> norm_dist(0.0, 1.0);
    auto a = norm_dist(engine);
    a = 1;
}