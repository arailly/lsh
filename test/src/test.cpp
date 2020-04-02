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

TEST(lsh, search) {
    const int k = 2, d = 128, r = 250, n = 3, L = 3;
    float range = 350;
    const string data_path = "/home/arai/workspace/dataset/sift/sift_base/";
    const auto series = load_data(data_path, n);
    auto series_for_index = series;

    auto index = LSHIndex(k, r, d, L);
    index.build(series_for_index);

    for (const auto i : vector<int>{12, 21, 22, 23, 30, 40}) {
        const auto result = index.range_search(series[i], range);
        ASSERT_GT(result.result.size(), 0);
    }
}

TEST(lsh, euclidean) {
    const int k = 4, d = 2, r = 3, L = 8;
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

    const auto result = index.range_search(query, 1.5);
    ASSERT_EQ(result.result.size(), 4);
}

TEST(lsh, manhattan) {
    const int k = 3, d = 2, r = 4, L = 10;
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

    const auto result = index.range_search(query, 1.1);
    ASSERT_EQ(result.result.size(), 4);
}

TEST(lsh, angular) {
    const int k = 4, d = 2, L = 1;
    const float r = 0.00001;
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
    const auto result = index.range_search(query, 0.0001);
    ASSERT_EQ(result.result.size(), 7);
}

TEST(lsh, get_bucket_contents) {
    const int k = 4, d = 2, r = 3, L = 8;
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

    const auto limit = 10;
    const auto bucket_contents_1 = index.get_bucket_contents(query);
    ASSERT_GT(bucket_contents_1.size(), limit);

    const auto bucket_contents_2 = index.get_bucket_contents(query, limit);
    ASSERT_EQ(bucket_contents_2.size(), limit);
}

TEST(lsh, knn_search) {
    const int n_hash_func = 4, d = 2, r = 3, L = 8;
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

    auto index = LSHIndex(n_hash_func, r, d, L);
    index.build(series_for_index);

    const auto query = Point(999, {4.5, 4.5});
    const auto k = 4;

    const auto result = index.knn_search(query, k);
    ASSERT_EQ(result.result.size(), k);
    ASSERT_EQ(result.result[0].get().id, 44);
    ASSERT_EQ(result.result[1].get().id, 54);
    ASSERT_EQ(result.result[2].get().id, 45);
    ASSERT_EQ(result.result[3].get().id, 55);
}

TEST(lsh, knn_search_sift) {
    const int n = 1000, n_query = 10000, k = 5;
    const string data_path = "/home/arai/workspace/dataset/sift/data1m/";
    const string query_path = "/home/arai/workspace/dataset/sift/sift_query.csv";

    const auto queries = load_data(query_path, n_query);

    const int n_hash_func = 4, r = 250, d = queries[0].size();

    auto index = LSHIndex(n_hash_func, r, d, 1);
    index.build(data_path, n);

    cout << "complete: build index" << endl;

    SearchResults results;
    for (const auto& query : queries) {
        const auto result = index.knn_search(query, k);
        results.push_back(result);
    }

    const string log_path = "/home/arai/workspace/result/knn-search/lsh/sift/data1m/k5/"
                            "log-k4r250.csv";
    const string result_path = "/home/arai/workspace/result/knn-search/lsh/sift/data1m/k5/"
                               "result-k4r250.csv";
    results.save(log_path, result_path, k);
}