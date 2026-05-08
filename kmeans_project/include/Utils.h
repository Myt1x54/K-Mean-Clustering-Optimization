#ifndef UTILS_H
#define UTILS_H

#include <cstddef>

struct AppConfig {
    std::size_t numPoints = 100000;
    int numClusters = 20;
    int maxIterations = 100;
    double minCoordinate = 0.0;
    double maxCoordinate = 1000.0;
    double convergenceThreshold = 1e-4;
    unsigned int randomSeed = 42;
};

AppConfig parseArguments(int argc, char* argv[]);
void printUsage(const char* programName);

#endif
