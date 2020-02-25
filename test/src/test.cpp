#include <gtest/gtest.h>
#include <random>
#include <arailib.hpp>
#include <lsh.hpp>

using namespace std;
using namespace arailib;
using namespace lsh;

TEST(lsh, create_hash_func) {
    const int k = 2, L = 1, d = 2, r = 1;
    auto index = LSHIndex(k, r, d, L);
    auto hash_func = index.create_hash_family();
    const Point p(0, {1, 2});
    const auto hash_vector = hash_func(p);
    ASSERT_EQ(hash_vector.size(), k);
}

TEST(lsh, find) {
    const int k = 2, d = 128, r = 1, n = 1, L = 1;
    const string data_path = "/Users/yusuke-arai/workspace/dataset/sift/sift_base/";
    const auto series = load_data(data_path, n);
    auto series_for_index = series;

    auto index = LSHIndex(k, r, d, L);
    index.build(series_for_index);

    for (int i = 0; i < 10; i++) {
        bool include_self = false;
        const auto find_result = index.find(index.G[0](series[i]), index.hash_tables[0]);
        for (const auto e : find_result) {
            if (e.id == i) include_self = true;
        }
        ASSERT_TRUE(include_self);
    }
}

TEST(lsh, search) {
    const int k = 2, d = 128, r = 250, n = 3, L = 3;
    float range = 350;
    const string data_path = "/Users/yusuke-arai/workspace/dataset/sift/sift_base/";
    const auto series = load_data(data_path, n);
    auto series_for_index = series;

    auto index = LSHIndex(k, r, d, L);
    index.build(series_for_index);

    for (const auto i : vector<int>{12, 21, 22, 23, 30, 40}) {
        const auto result = index.search(series[i], range);
        ASSERT_GT(result.series.size(), 0);
    }
}

TEST(lsh, euclidean) {
    const int k = 3, d = 2, r = 3, L = 4;
    const auto series = [&]() {
        auto series_ = Series();
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                const auto id = (size_t)(10 * i + j);
                const float x = i, y = j;
                const auto point = Point(id, {x, y});
                series_.push_back(point);
            }
        }
        return series_;
    }();
    auto series_for_index = series;

    auto index = LSHIndex(k, r, d, L);
    index.build(series_for_index);

    const auto query = Point(999, {4.5, 4.5});

    const auto result = index.search(query, 1.5);
    ASSERT_EQ(result.series.size(), 4);
}

TEST(lsh, manhattan) {
    const int k = 3, d = 2, r = 5, L = 10;
    const auto series = [&]() {
        auto series_ = Series();
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                const auto id = (size_t)(10 * i + j);
                const float x = i, y = j;
                const auto point = Point(id, {x, y});
                series_.push_back(point);
            }
        }
        return series_;
    }();
    auto series_for_index = series;

    auto index = LSHIndex(k, r, d, L, "manhattan");
    index.build(series_for_index);

    const auto query = Point(999, {4.5, 4.5});

    const auto result = index.search(query, 1.1);
    ASSERT_EQ(result.series.size(), 4);
}

TEST(lsh, angular) {
    const int k = 3, d = 2, r = 5, L = 20;
    const auto series = [&]() {
        auto series_ = Series();
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                const auto id = (size_t)(10 * i + j);
                const float x = i, y = j;
                const auto point = Point(id, {x, y});
                series_.push_back(point);
            }
        }
        return series_;
    }();
    auto series_for_index = series;

    auto index = LSHIndex(k, r, d, L, "angular");
    index.build(series_for_index);

    const auto query = Point(999, {4.5, 4.5});
    const auto result = index.search(query, 0.0001);
    ASSERT_EQ(result.series.size(), 6);
}