#include "Utils.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool parsePositiveSizeT(const char* text, std::size_t& outValue) {
    try {
        const std::string value(text);
        const std::size_t parsed = std::stoull(value);
        if (parsed == 0) {
            return false;
        }
        outValue = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parsePositiveInt(const char* text, int& outValue) {
    try {
        const std::string value(text);
        const int parsed = std::stoi(value);
        if (parsed <= 0) {
            return false;
        }
        outValue = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseDouble(const char* text, double& outValue) {
    try {
        const std::string value(text);
        outValue = std::stod(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parseUnsignedInt(const char* text, unsigned int& outValue) {
    try {
        const std::string value(text);
        outValue = static_cast<unsigned int>(std::stoul(value));
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

AppConfig parseArguments(int argc, char* argv[]) {
    AppConfig config;

    // Flexible argument parsing to support an optional mode string first.
    // Usage patterns supported:
    // 1) ./kmeans sequential|parallel|both|optimized|all|benchmark [N] [K] [maxIters] [min] [max] [threshold] [seed] [threads]
    // 2) ./kmeans N K maxIters [min] [max] [threshold] [seed] [threads]

    int idx = 1;
    if (argc > 1) {
        const std::string first(argv[1]);
        if (first == "sequential" || first == "parallel" || first == "both" || first == "optimized" || first == "all" || first == "benchmark") {
            config.mode = first;
            idx = 2; // subsequent args start here
            // If mode indicates scheduling experiments (optimized/all), allow an optional scheduling token next.
            if (argc > 2 && first != "benchmark") {
                const std::string second(argv[2]);
                if (second == "static" || second == "dynamic" || second == "guided") {
                    config.schedulePolicy = second;
                    idx = 3; // scheduling token consumed
                }
            }
        }
    }

    if (argc > idx) {
        if (!parsePositiveSizeT(argv[idx], config.numPoints)) {
            throw std::invalid_argument("Invalid number of points. Must be a positive integer.");
        }
    }

    if (argc > idx + 1) {
        if (!parsePositiveInt(argv[idx + 1], config.numClusters)) {
            throw std::invalid_argument("Invalid number of clusters. Must be a positive integer.");
        }
    }

    if (argc > idx + 2) {
        if (!parsePositiveInt(argv[idx + 2], config.maxIterations)) {
            throw std::invalid_argument("Invalid max iterations. Must be a positive integer.");
        }
    }

    if (argc > idx + 3) {
        if (!parseDouble(argv[idx + 3], config.minCoordinate)) {
            throw std::invalid_argument("Invalid min coordinate. Must be numeric.");
        }
    }

    if (argc > idx + 4) {
        if (!parseDouble(argv[idx + 4], config.maxCoordinate)) {
            throw std::invalid_argument("Invalid max coordinate. Must be numeric.");
        }
    }

    if (argc > idx + 5) {
        if (!parseDouble(argv[idx + 5], config.convergenceThreshold) || config.convergenceThreshold <= 0.0) {
            throw std::invalid_argument("Invalid convergence threshold. Must be > 0.");
        }
    }

    if (argc > idx + 6) {
        if (!parseUnsignedInt(argv[idx + 6], config.randomSeed)) {
            throw std::invalid_argument("Invalid seed. Must be an unsigned integer.");
        }
    }

    if (argc > idx + 7) {
        int nt = 0;
        if (!parsePositiveInt(argv[idx + 7], nt)) {
            throw std::invalid_argument("Invalid thread count. Must be a positive integer.");
        }
        config.numThreads = nt;
    }

    // Optional schedule chunk: if present after threads, consume it
    if (argc > idx + 8) {
        int chunk = 0;
        if (!parsePositiveInt(argv[idx + 8], chunk)) {
            throw std::invalid_argument("Invalid schedule chunk. Must be a positive integer.");
        }
        config.scheduleChunk = chunk;
    }

    if (config.minCoordinate >= config.maxCoordinate) {
        throw std::invalid_argument("min coordinate must be smaller than max coordinate.");
    }

    return config;
}

void printUsage(const char* programName) {
    std::cout << "Usage:\n"
              << "  " << programName << " [mode] [num_points] [num_clusters] [max_iterations]"
              << " [min_coord] [max_coord] [threshold] [seed] [threads]\n\n"
              << "Modes:\n"
              << "  sequential      Run sequential K-Means\n"
              << "  parallel        Run naive parallel (critical sections)\n"
              << "  optimized       Run optimized parallel (thread-local accumulators)\n"
              << "  both            Run sequential then parallel (for comparison)\n"
              << "  all             Run all implementations with correctness checks\n"
              << "  benchmark       Run comprehensive benchmark suite\n\n"
              << "Examples:\n"
              << "  " << programName << "\n"
              << "  " << programName << " sequential 100000 20 100\n"
              << "  " << programName << " optimized dynamic 1000000 32 200 0 5000 1e-5 123 8 100\n"
              << "  " << programName << " benchmark\n";
}
