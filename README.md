# LSH
## Overview
It is implementation of similarity search on LSH: Locality Sensitive Hashing.

The main algorithm is written `include/lsh.hpp` (header only), so you can use it easily.

Reference: Locality-Sensitive Hashing Scheme Based on p-Stable Distributions (M. Datar et al., SoCG 2004)

## Example
```
#include <iostream>
#include <arailib.hpp>
#include <lsh.hpp>

using namespace std;
using namespace arailib;
using namespace lsh;

int main() {
    const string data_path = "path/to/data.csv";
    const string query_path = "path/to/query.csv";
    const string save_path = "path/to/result.csv";
    const unsigned n = 1000; // data size
    const unsigned n_query = 10; // query size
    const float range = 10; // search range (distance)
    const int k = 3; // number of hash function of each table
    const int L = 10; // number of hash tables
    const float r = 1; // bucket witdh
    const string& distance = "euclidean"; // distance name; we support "euclidean", "manhattan", and "angular".
    
    const auto queries = load_data(query_path, n_query); // read query file

    auto index = LSHIndex(k, r, L, distance);
    index.build(data_path, n); // build index
    
    vector<SearchResult> results(queries.size());
    for (const auto& query : queries) {
        results[query.id] = index.range_search(query, range);
    }
}
```

## Input File Format
If you want to create index with this three vectors, `(0, 1), (2, 4), (3, 3)`, you must describe data.csv like following format:
```
0,1
2,4
3,3
```

Query format is same as data format.
