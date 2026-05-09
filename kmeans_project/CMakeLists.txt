cmake_minimum_required(VERSION 3.14)

project(kmeans_project VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Prefer -O3 for optimization in research builds. OpenMP will be enabled when found.
if(CMAKE_BUILD_TYPE MATCHES "Release")
    add_compile_options(-O3)
endif()

add_executable(kmeans
    src/main.cpp
    src/Point.cpp
    src/Cluster.cpp
    src/KMeans.cpp
    src/Timer.cpp
    src/Utils.cpp
    src/BenchmarkRunner.cpp
)

target_include_directories(kmeans PRIVATE include)

# Keep warning levels high for correctness while preserving portability.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kmeans PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Enable OpenMP if available and link the target. This adds -fopenmp where needed.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: enabling OpenMP for kmeans target")
    target_link_libraries(kmeans PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found; building without OpenMP")
endif()
