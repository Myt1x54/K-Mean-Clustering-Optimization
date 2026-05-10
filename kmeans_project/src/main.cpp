#include <exception>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <vector>

#include "KMeans.h"
#include "KMeansSoA.h"
#include "Utils.h"
#include "BenchmarkRunner.h"
#include "ProfileRunner.h"
#include "ScalabilityRunner.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

std::vector<std::string> splitStrings(const std::string& value) {
    std::vector<std::string> out;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) out.push_back(token);
    }
    return out;
}

std::vector<int> splitInts(const std::string& value) {
    std::vector<int> out;
    for (const auto& token : splitStrings(value)) {
        out.push_back(std::stoi(token));
    }
    return out;
}

std::vector<std::size_t> splitSizes(const std::string& value) {
    std::vector<std::size_t> out;
    for (const auto& token : splitStrings(value)) {
        out.push_back(static_cast<std::size_t>(std::stoull(token)));
    }
    return out;
}

bool parseBool(const char* value, bool defaultValue) {
    if (value == nullptr) return defaultValue;
    const std::string lowered = value;
    return !(lowered == "0" || lowered == "false" || lowered == "FALSE");
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc >= 2 && std::string(argv[1]) == "scalability") {
            ScalabilityConfig config;

            if (argc >= 3) {
                config.implementations = splitStrings(argv[2]);
            }
            if (argc >= 4) {
                config.pointCounts = splitSizes(argv[3]);
            }
            if (argc >= 5) {
                config.clusterCounts = splitInts(argv[4]);
            }
            if (argc >= 6) {
                config.maxIterations = std::stoi(argv[5]);
            }
            if (argc >= 7) {
                config.repetitions = std::stoi(argv[6]);
            }

            if (const char* envThreads = std::getenv("SCAL_THREADS")) config.threadCounts = splitInts(envThreads);
            if (const char* envSchedules = std::getenv("SCAL_SCHEDULES")) config.schedulePolicies = splitStrings(envSchedules);
            if (const char* envChunks = std::getenv("SCAL_CHUNKS")) config.chunkSizes = splitInts(envChunks);
            if (const char* envOut = std::getenv("SCAL_OUTDIR")) config.outputDir = envOut;
            if (const char* envProfiling = std::getenv("SCAL_PROFILING_CSV")) config.profilingCsvPath = envProfiling;
            config.capThreadsToHardware = parseBool(std::getenv("SCAL_CAP_THREADS"), true);

            ScalabilityRunner runner;
            runner.execute(config);
            return 0;
        }

        // Quick pre-check: support a separate 'profile' command before normal argument parsing
        if (argc >= 2 && std::string(argv[1]) == "profile") {
            // Expected usage: ./kmeans profile <implementation> <threads> <points> <clusters> [schedule] [chunk]
            std::string impl = (argc >= 3) ? argv[2] : "optimized";
            int threads = (argc >= 4) ? std::stoi(argv[3]) : 1;
            std::size_t points = (argc >= 5) ? static_cast<std::size_t>(std::stoull(argv[4])) : 100000;
            int clusters = (argc >= 6) ? std::stoi(argv[5]) : 10;
            std::string schedule = (argc >= 7) ? argv[6] : "static";
            int chunk = (argc >= 8) ? std::stoi(argv[7]) : 1000;

            std::ostringstream target;
            if (impl == "optimized") {
                target << "./build/kmeans optimized " << schedule << " " << points << " " << clusters << " 100 0 1000 1e-4 42 " << threads << " " << chunk;
            } else if (impl == "soa") {
                target << "./build/kmeans soa " << schedule << " " << points << " " << clusters << " 100 0 1000 1e-4 42 " << threads << " " << chunk;
            } else {
                std::cerr << "Profile mode currently supports only 'optimized' and 'soa'.\n";
                return 2;
            }

            ProfileRunner pr("./build/kmeans");
            ProfileConfig cfg;
            cfg.implementation = impl;
            cfg.targetCommand = target.str();
            cfg.threads = threads;
            cfg.points = points;
            cfg.clusters = clusters;
            cfg.schedule = schedule;
            cfg.chunk_size = chunk;

            if (!pr.run(cfg)) {
                std::cerr << "Profiling failed." << std::endl;
                return 2;
            }
            return 0;
        }

        const AppConfig config = parseArguments(argc, argv);

        // Determine mode: sequential, parallel (naive), optimized, both, all, or benchmark
        const std::string mode = config.mode;

        if (mode != "sequential" && mode != "parallel" && mode != "both" && mode != "optimized" && mode != "soa" && mode != "all" && mode != "benchmark") {
            throw std::invalid_argument("Mode must be 'sequential', 'parallel', 'optimized', 'soa', 'both', 'all', or 'benchmark'");
        }

        // Handle benchmark mode separately
        if (mode == "benchmark") {
            BenchmarkRunner runner;
            runner.executeFullBenchmark();
            runner.exportToCSV("/home/myt1x/PDC_Project/kmeans_project/benchmark/results/benchmark_results.csv");
            return 0;
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

        if (mode == "soa") {
#ifdef _OPENMP
            std::cout << "Setting OpenMP threads to: " << config.numThreads << "\n";
            omp_set_num_threads(config.numThreads);
#endif
            std::cout << "=== SOA PARALLEL RUN ===\n";
            KMeansSoA soa(
                config.numClusters,
                config.maxIterations,
                config.convergenceThreshold,
                config.randomSeed
            );
            soa.generateRandomPoints(config.numPoints, config.minCoordinate, config.maxCoordinate);
            soa.runParallelMemoryOptimized(config.numThreads, config.scheduleChunk, config.schedulePolicy);
            if (!soa.validateAssignments()) {
                std::cerr << "Error: Invalid cluster assignments detected in SoA run.\n";
                return 2;
            }
            std::cout << "SoA runtime (ms): " << soa.getTotalRuntimeMs() << "\n";
            return 0;
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
