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

        // Determine mode: sequential, parallel (naive), optimized, both, or all
        const std::string mode = config.mode;

        if (mode != "sequential" && mode != "parallel" && mode != "both" && mode != "optimized" && mode != "all") {
            throw std::invalid_argument("Mode must be 'sequential', 'parallel', 'optimized', 'both', or 'all'");
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

#ifndef _OPENMP
        // Provide a clear message when OpenMP is not available but user requested parallel modes
        if ((mode == "parallel" || mode == "optimized" || mode == "all" || mode == "both") && false) {
        }
#endif

        if (mode == "parallel" || mode == "both" || mode == "all" || mode == "optimized") {
#ifdef _OPENMP
            std::cout << "Setting OpenMP threads to: " << config.numThreads << "\n";
            omp_set_num_threads(config.numThreads);
#else
            (void)config;
#endif

            // Run naive parallel (critical section) if requested
            double naiveTime = -1.0;
            if (mode == "parallel" || mode == "both" || mode == "all") {
                std::cout << "=== PARALLEL (naive) RUN ===\n";
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
                naiveTime = par.getTotalRuntimeMs();
            }

            // Run optimized parallel if requested
            double optTime = -1.0;
            if (mode == "optimized" || mode == "all") {
                std::cout << "=== PARALLEL (optimized) RUN ===\n";
                KMeans opt(
                    config.numClusters,
                    config.maxIterations,
                    config.convergenceThreshold,
                    config.randomSeed
                );
                opt.setPoints(initialPoints);
                opt.runParallelOptimized(config.numThreads, config.ompChunkSize, config.schedulePolicy);

                if (!opt.validateAssignments()) {
                    std::cerr << "Error: Invalid cluster assignments detected in optimized run.\n";
                    return 2;
                }
                opt.printStatistics();
                optTime = opt.getTotalRuntimeMs();

                // For correctness, comparisons will be handled in the 'all' mode block below.
            }

            // record parallel naive time for external reporting
            parallelTime = naiveTime;
        }


        if (mode == "all") {
            // Run a sequential reference
            KMeans seq_ref(
                config.numClusters,
                config.maxIterations,
                config.convergenceThreshold,
                config.randomSeed
            );
            seq_ref.setPoints(initialPoints);
            seq_ref.run();
            const double seqTimeRef = seq_ref.getTotalRuntimeMs();

            // Re-run optimized to capture final state for comparison
            KMeans opt2(
                config.numClusters,
                config.maxIterations,
                config.convergenceThreshold,
                config.randomSeed
            );
            opt2.setPoints(initialPoints);
            opt2.runParallelOptimized(config.numThreads, config.ompChunkSize, config.schedulePolicy);

            // Compare results using accessors
            const bool centroidsEqual = KMeans::compareCentroids(seq_ref.getClusters(), opt2.getClusters(), 1e-6);
            const bool assignEqual = KMeans::compareAssignments(seq_ref.getPoints(), opt2.getPoints());

            std::cout << "\n===== Correctness Comparison (seq vs optimized) =====\n";
            std::cout << "Centroids match (tol=1e-6): " << (centroidsEqual ? "Yes" : "No") << "\n";
            std::cout << "Assignments match: " << (assignEqual ? "Yes" : "No") << "\n";
            std::cout << "===================================================\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n\n";
        printUsage(argv[0]);
        return 1;
    }
}
