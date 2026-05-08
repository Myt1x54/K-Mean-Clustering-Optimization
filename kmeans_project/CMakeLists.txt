cmake_minimum_required(VERSION 3.14)

project(kmeans_project VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(kmeans
    src/main.cpp
    src/Point.cpp
    src/Cluster.cpp
    src/KMeans.cpp
    src/Timer.cpp
    src/Utils.cpp
)

target_include_directories(kmeans PRIVATE include)

# Keep warning levels high for correctness while preserving portability.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kmeans PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Future OpenMP integration can be enabled with:
# find_package(OpenMP REQUIRED)
# target_link_libraries(kmeans PRIVATE OpenMP::OpenMP_CXX)
