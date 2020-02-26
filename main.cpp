#include <iostream>
#include <arailib.hpp>
#include <lsh.hpp>

using namespace std;
using namespace arailib;
using namespace lsh;

void save_result(const string& save_path, const vector<SearchResult>& results) {
    ofstream ofs(save_path);
    string line = "time,n_result,n_bucket_content\n";
    ofs << line;

    for (const auto& result : results) {
        line = to_string(result.time) + "," +
               to_string(result.series.size()) + "," +
               to_string(result.n_bucket_content) + "\n";
        ofs << line;
    }
}

int main() {
    const auto config = read_config();
    int n = config["n"], n_query = config["n_query"];
    float range = config["range"];
    int r = config["r"];
    int k = config["k"];
    int L = config["L"];
    const string distance = config["distance"];
    const string data_path = config["data_path"];
    const string query_path = config["query_path"];
    const string save_path = config["save_path"];

    const auto queries = read_csv(query_path, n_query);

    size_t d = queries[0].size();

    auto index = LSHIndex(k, r, d, L, distance);
    index.build(data_path, n);

    cout << "complete: build index" << endl;

    vector<SearchResult> results;
    for (const auto& query : queries) {
        const auto result = index.search(query, range);
        results.push_back(result);
    }

    cout << "complete: search" << endl;

    save_result(save_path, results);
}