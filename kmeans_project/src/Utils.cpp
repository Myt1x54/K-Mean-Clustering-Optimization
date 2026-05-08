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

    if (argc > 1) {
        if (!parsePositiveSizeT(argv[1], config.numPoints)) {
            throw std::invalid_argument("Invalid number of points. Must be a positive integer.");
        }
    }

    if (argc > 2) {
        if (!parsePositiveInt(argv[2], config.numClusters)) {
            throw std::invalid_argument("Invalid number of clusters. Must be a positive integer.");
        }
    }

    if (argc > 3) {
        if (!parsePositiveInt(argv[3], config.maxIterations)) {
            throw std::invalid_argument("Invalid max iterations. Must be a positive integer.");
        }
    }

    if (argc > 4) {
        if (!parseDouble(argv[4], config.minCoordinate)) {
            throw std::invalid_argument("Invalid min coordinate. Must be numeric.");
        }
    }

    if (argc > 5) {
        if (!parseDouble(argv[5], config.maxCoordinate)) {
            throw std::invalid_argument("Invalid max coordinate. Must be numeric.");
        }
    }

    if (argc > 6) {
        if (!parseDouble(argv[6], config.convergenceThreshold) || config.convergenceThreshold <= 0.0) {
            throw std::invalid_argument("Invalid convergence threshold. Must be > 0.");
        }
    }

    if (argc > 7) {
        if (!parseUnsignedInt(argv[7], config.randomSeed)) {
            throw std::invalid_argument("Invalid seed. Must be an unsigned integer.");
        }
    }

    if (config.minCoordinate >= config.maxCoordinate) {
        throw std::invalid_argument("min coordinate must be smaller than max coordinate.");
    }

    return config;
}

void printUsage(const char* programName) {
    std::cout << "Usage:\n"
              << "  " << programName << " [num_points] [num_clusters] [max_iterations]"
              << " [min_coord] [max_coord] [threshold] [seed]\n\n"
              << "Examples:\n"
              << "  " << programName << "\n"
              << "  " << programName << " 100000 20 100\n"
              << "  " << programName << " 1000000 32 200 0 5000 1e-5 123\n";
}
