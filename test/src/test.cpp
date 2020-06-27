#include <gtest/gtest.h>
#include <random>
#include <arailib.hpp>
#include <lsh.hpp>

using namespace std;
using namespace arailib;
using namespace lsh;

TEST(lsh, search) {
    const int k = 2, r = 250, n = 3, L = 3;
    double range = 350;
    const string data_path = "/home/arai/workspace/dataset/sift/sift_base/";
    const auto series = load_data(data_path, n);
    auto series_for_index = series;

    auto index = LSHIndex(k, r, L);
    index.build(series_for_index);

    for (const auto i : vector<int>{12, 21, 22, 23, 30, 40}) {
        const auto result = index.range_search(series[i], range);
        ASSERT_GT(result.result.size(), 0);
    }
}

TEST(lsh, euclidean) {
    const int k = 4, r = 3, L = 8;
    const auto series = [&]() {
        auto series_ = Series<>();
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                const auto id = (size_t)(10 * i + j);
                const double x = i, y = j;
                const auto point = Data<>(id, {x, y});
                series_.push_back(point);
            }
        }
        return series_;
    }();
    auto series_for_index = series;

    auto index = LSHIndex(k, r, L);
    index.build(series_for_index);

    const auto query = Data<>(999, {4.5, 4.5});

    const auto result = index.range_search(query, 1.5);
    ASSERT_EQ(result.result.size(), 4);
}

TEST(lsh, manhattan) {
    const int k = 3, r = 4, L = 10;
    const auto series = [&]() {
        auto series_ = Series<>();
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                const auto id = (size_t)(10 * i + j);
                const double x = i, y = j;
                const auto point = Data<>(id, {x, y});
                series_.push_back(point);
            }
        }
        return series_;
    }();
    auto series_for_index = series;

    auto index = LSHIndex(k, r, L, "manhattan");
    index.build(series_for_index);

    const auto query = Data<>(999, {4.5, 4.5});

    const auto result = index.range_search(query, 1.1);
    ASSERT_EQ(result.result.size(), 4);
}

TEST(lsh, angular) {
    const int k = 4, L = 1;
    const double r = 0.00001;
    const auto series = [&]() {
        auto series_ = Series<>();
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                const auto id = (size_t)(10 * i + j);
                const double x = i, y = j;
                const auto point = Data<>(id, {x, y});
                series_.push_back(point);
            }
        }
        return series_;
    }();
    auto series_for_index = series;

    auto index = LSHIndex(k, r, L, "angular");
    index.build(series_for_index);

    const auto query = Data<>(999, {4.5, 4.5});
    const auto result = index.range_search(query, 0.0001);
    ASSERT_EQ(result.result.size(), 7);
}

TEST(lsh, get_bucket_contents) {
    const int k = 4, r = 3, L = 8;
    const auto series = [&]() {
        auto series_ = Series<>();
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                const auto id = (size_t)(10 * i + j);
                const double x = i, y = j;
                const auto point = Data<>(id, {x, y});
                series_.push_back(point);
            }
        }
        return series_;
    }();
    auto series_for_index = series;

    auto index = LSHIndex(k, r, L);
    index.build(series_for_index);

    const auto query = Data<>(999, {4.5, 4.5});

    const auto limit = 10;
    const auto bucket_contents_1 = index.find(query);
    ASSERT_GT(bucket_contents_1.size(), limit);

    const auto bucket_contents_2 = index.find(query, limit);
    ASSERT_EQ(bucket_contents_2.size(), limit);
}

TEST(lsh, knn_search) {
    const int n_hash_func = 4, r = 3, L = 8;
    const auto series = [&]() {
        auto series_ = Series<>();
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                const auto id = (size_t)(10 * i + j);
                const double x = i, y = j;
                const auto point = Data<>(id, {x, y});
                series_.push_back(point);
            }
        }
        return series_;
    }();
    auto series_for_index = series;

    auto index = LSHIndex(n_hash_func, r, L);
    index.build(series_for_index);

    const auto query = Data<>(999, {4.5, 4.5});
    const auto k = 4;

    const auto result = index.knn_search(query, k);
    ASSERT_EQ(result.result.size(), k);
    ASSERT_EQ(result.result[0], 44);
    ASSERT_EQ(result.result[1], 45);
    ASSERT_EQ(result.result[2], 54);
    ASSERT_EQ(result.result[3], 55);
}

TEST(lsh, knn_search_sift) {
    const int n = 1000, n_query = 10000, k = 10;
    const string data_path = "/home/arai/workspace/dataset/sift/data1m/";
    const string query_path = "/home/arai/workspace/dataset/sift/sift_query.csv";

    const auto queries = load_data(query_path, n_query);

    const int m = 4, r = 200, L = 10;

    auto index = LSHIndex(m, r, L);
    index.build(data_path, n);

    cout << "complete: build index" << endl;

    SearchResults results;
    for (const auto& query : queries) {
        const auto result = index.knn_search(query, k);
        results.push_back(result);
    }

    const string log_path = "/home/arai/workspace/result/knn-search/lsh/sift/data1m/k10/"
                            "log-m4r200L10.csv";
    const string result_path = "/home/arai/workspace/result/knn-search/lsh/sift/data1m/k10/"
                               "result-m4r200L10.csv";
    results.save(log_path, result_path, k);
}