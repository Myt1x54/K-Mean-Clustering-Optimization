#ifndef UTILS_H
#define UTILS_H

#include <cstddef>
#include <string>

struct AppConfig {
    std::size_t numPoints = 100000;
    int numClusters = 20;
    int maxIterations = 100;
    double minCoordinate = 0.0;
    double maxCoordinate = 1000.0;
    double convergenceThreshold = 1e-4;
    unsigned int randomSeed = 42;
    // Execution mode: "sequential", "parallel", or "both" (run sequential then parallel)
    std::string mode = "sequential";
    // Number of OpenMP threads to use for parallel mode (ignored for sequential)
    int numThreads = 1;
    // OpenMP static schedule chunk size (default per paper)
    int ompChunkSize = 1000;
    // Debug logging flag
    bool debug = false;
};

AppConfig parseArguments(int argc, char* argv[]);
void printUsage(const char* programName);

#endif
