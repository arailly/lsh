cmake_minimum_required(VERSION 3.14)

add_executable(test_main test.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

target_link_libraries(test_main gtest gtest_main)

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)