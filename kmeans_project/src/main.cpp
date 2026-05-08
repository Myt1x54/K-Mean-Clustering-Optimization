#include <exception>
#include <iostream>

#include "KMeans.h"
#include "Utils.h"

int main(int argc, char* argv[]) {
    try {
        const AppConfig config = parseArguments(argc, argv);

        KMeans kmeans(
            config.numClusters,
            config.maxIterations,
            config.convergenceThreshold,
            config.randomSeed
        );

        kmeans.generateRandomPoints(config.numPoints, config.minCoordinate, config.maxCoordinate);
        kmeans.run();

        const bool valid = kmeans.validateAssignments();
        if (!valid) {
            std::cerr << "Error: Invalid cluster assignments detected.\n";
            return 2;
        }

        kmeans.printStatistics();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n\n";
        printUsage(argv[0]);
        return 1;
    }
}
