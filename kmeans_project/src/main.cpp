#include <exception>
#include <iostream>

#include "KMeans.h"
#include "Utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char* argv[]) {
    try {
        const AppConfig config = parseArguments(argc, argv);

        // Determine mode: sequential, parallel, or both
        const std::string mode = config.mode;

        if (mode != "sequential" && mode != "parallel" && mode != "both") {
            throw std::invalid_argument("Mode must be 'sequential', 'parallel', or 'both'");
        }

        // Build a reference dataset once to ensure sequential and parallel runs operate on the same input.
        KMeans generator(
            config.numClusters,
            config.maxIterations,
            config.convergenceThreshold,
            config.randomSeed
        );

        generator.generateRandomPoints(config.numPoints, config.minCoordinate, config.maxCoordinate);
        const auto initialPoints = generator.getPoints();

        double sequentialTime = -1.0;
        double parallelTime = -1.0;

        if (mode == "sequential" || mode == "both") {
            KMeans seq(
                config.numClusters,
                config.maxIterations,
                config.convergenceThreshold,
                config.randomSeed
            );
            seq.setPoints(initialPoints);
            seq.run();

            if (!seq.validateAssignments()) {
                std::cerr << "Error: Invalid cluster assignments detected in sequential run.\n";
                return 2;
            }
            seq.printStatistics();
            sequentialTime = seq.getTotalRuntimeMs();
        }

        if (mode == "parallel" || mode == "both") {
#ifdef _OPENMP
            std::cout << "Setting OpenMP threads to: " << config.numThreads << "\n";
            omp_set_num_threads(config.numThreads);
#else
            (void)config;
#endif

            KMeans par(
                config.numClusters,
                config.maxIterations,
                config.convergenceThreshold,
                config.randomSeed
            );
            par.setPoints(initialPoints);
            par.runParallel(config.numThreads, config.ompChunkSize);

            if (!par.validateAssignments()) {
                std::cerr << "Error: Invalid cluster assignments detected in parallel run.\n";
                return 2;
            }
            par.printStatistics();
            parallelTime = par.getTotalRuntimeMs();
        }

        if (mode == "both") {
            if (sequentialTime > 0.0 && parallelTime > 0.0) {
                const double speedup = sequentialTime / parallelTime;
                const double efficiency = speedup / static_cast<double>(config.numThreads);
                std::cout << "\n===== Parallel Comparison =====\n";
                std::cout << "Sequential time (ms): " << sequentialTime << "\n";
                std::cout << "Parallel time   (ms): " << parallelTime << "\n";
                std::cout << "Speedup: " << speedup << "\n";
                std::cout << "Efficiency: " << efficiency << "\n";
                std::cout << "===============================\n";
            } else {
                std::cout << "Could not compute speedup (missing timing data).\n";
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n\n";
        printUsage(argv[0]);
        return 1;
    }
}
