#include "BenchmarkRunner.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "KMeans.h"
#include "KMeansSoA.h"

BenchmarkRunner::BenchmarkRunner()
    : sequentialBaseline_(0.0), totalExperiments_(0), currentExperiment_(0) {}

void BenchmarkRunner::executeFullBenchmark() {
    std::cout << "\n========================================\n";
    std::cout << "K-MEANS BENCHMARK SUITE\n";
    std::cout << "========================================\n\n";

    // Define experiment grid
    std::vector<int> threadCounts = {1, 2, 4, 8, 16};
    std::vector<std::size_t> pointCounts = {100000, 500000, 1000000};
    std::vector<int> clusterCounts = {5, 10, 20, 50};
    std::vector<std::string> implementations = {"sequential", "naive", "optimized", "soa"};
    std::vector<std::string> schedules = {"static", "dynamic", "guided"};
    std::vector<int> chunkSizes = {100, 1000};
    int maxIterations = 100;
    double convergenceThreshold = 1e-4;
    int repetitions = 5;

    // Pre-calculate total experiments for progress tracking
    totalExperiments_ = 0;
    for (const auto& impl : implementations) {
        for (const auto& pts : pointCounts) {
            for (const auto& clusters : clusterCounts) {
                for (const auto& tc : threadCounts) {
                    if (impl == "sequential") {
                        totalExperiments_++;  // One run per config
                    } else {
                        // For parallel: static is typical, but dynamic/guided are tested
                        totalExperiments_ += schedules.size();
                    }
                }
            }
        }
    }

    currentExperiment_ = 0;

    std::cout << "Total experiments to run: " << totalExperiments_ << "\n";
    std::cout << "Repetitions per experiment: " << repetitions << "\n\n";

    // Generate baseline: sequential time on smallest dataset
    std::cout << "--- Computing Sequential Baseline ---\n";
    ExperimentConfig baselineConfig;
    baselineConfig.implementation = "sequential";
    baselineConfig.schedulePolicy = "static";
    baselineConfig.chunkSize = 1000;
    baselineConfig.threadCount = 1;
    baselineConfig.numPoints = 100000;
    baselineConfig.numClusters = 5;
    baselineConfig.maxIterations = maxIterations;
    baselineConfig.convergenceThreshold = convergenceThreshold;
    baselineConfig.randomSeed = 42;
    baselineConfig.repetitions = 2;

    BenchmarkResult baselineResult;
    runSingleExperiment(baselineConfig, baselineResult);
    computeStatistics(baselineResult);
    sequentialBaseline_ = baselineResult.mean_runtime_ms;
    std::cout << "Sequential baseline: " << std::fixed << std::setprecision(3)
              << sequentialBaseline_ << " ms\n\n";

    // Main experiment loop
    std::cout << "--- Running Main Benchmark Suite ---\n";

    for (const auto& impl : implementations) {
        for (const auto& pts : pointCounts) {
            for (const auto& clusters : clusterCounts) {
                for (const auto& tc : threadCounts) {
                    if (impl == "sequential") {
                        ExperimentConfig config;
                        config.implementation = impl;
                        config.schedulePolicy = "static";
                        config.chunkSize = 1000;
                        config.threadCount = 1;
                        config.numPoints = pts;
                        config.numClusters = clusters;
                        config.maxIterations = maxIterations;
                        config.convergenceThreshold = convergenceThreshold;
                        config.randomSeed = 42;
                        config.repetitions = repetitions;

                        BenchmarkResult result;
                        currentExperiment_++;
                        displayProgress(config, currentExperiment_);
                        runSingleExperiment(config, result);
                        computeStatistics(result);
                        result.speedup = sequentialBaseline_ / result.mean_runtime_ms;
                        result.efficiency = result.speedup / 1.0;
                        allResults_.push_back(result);
                    } else {
                        // Parallel implementations: test all schedules
                        for (const auto& sched : schedules) {
                            for (const auto& chunk : chunkSizes) {
                                ExperimentConfig config;
                                config.implementation = impl;
                                config.schedulePolicy = sched;
                                config.chunkSize = chunk;
                                config.threadCount = tc;
                                config.numPoints = pts;
                                config.numClusters = clusters;
                                config.maxIterations = maxIterations;
                                config.convergenceThreshold = convergenceThreshold;
                                config.randomSeed = 42;
                                config.repetitions = repetitions;

                                BenchmarkResult result;
                                currentExperiment_++;
                                displayProgress(config, currentExperiment_);
                                runSingleExperiment(config, result);
                                computeStatistics(result);
                                result.speedup = sequentialBaseline_ / result.mean_runtime_ms;
                                result.efficiency = result.speedup / static_cast<double>(tc);
                                allResults_.push_back(result);
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "Benchmark completed. " << allResults_.size() << " results collected.\n";
    std::cout << "========================================\n\n";
}

void BenchmarkRunner::runSingleExperiment(const ExperimentConfig& config, BenchmarkResult& result) {
    result.config = config;
    result.runtimes_ms.clear();
    result.runtimes_ms.reserve(config.repetitions);

    // Warm-up run (not counted in statistics)
    {
        KMeans warmup(config.numClusters, config.maxIterations, config.convergenceThreshold, config.randomSeed);
        warmup.generateRandomPoints(config.numPoints, 0.0, 1000.0);

        if (config.implementation == "sequential") {
            warmup.run();
        } else if (config.implementation == "naive") {
            warmup.runParallel(config.threadCount, config.chunkSize);
        } else if (config.implementation == "optimized") {
            warmup.runParallelOptimized(config.threadCount, config.chunkSize, config.schedulePolicy);
        } else if (config.implementation == "soa") {
            KMeansSoA warmup_soa(config.numClusters, config.maxIterations, config.convergenceThreshold, config.randomSeed);
            warmup_soa.generateRandomPoints(config.numPoints, 0.0, 1000.0);
            warmup_soa.runParallelMemoryOptimized(config.threadCount, config.chunkSize, config.schedulePolicy);
        }
    }

    // Actual benchmark runs
    for (int rep = 0; rep < config.repetitions; ++rep) {
        KMeans kmeans(config.numClusters, config.maxIterations, config.convergenceThreshold, config.randomSeed + rep);
        kmeans.generateRandomPoints(config.numPoints, 0.0, 1000.0);

        if (config.implementation == "sequential") {
            kmeans.run();
        } else if (config.implementation == "naive") {
            kmeans.runParallel(config.threadCount, config.chunkSize);
        } else if (config.implementation == "optimized") {
            kmeans.runParallelOptimized(config.threadCount, config.chunkSize, config.schedulePolicy);
        } else if (config.implementation == "soa") {
            KMeansSoA ksoa(config.numClusters, config.maxIterations, config.convergenceThreshold, config.randomSeed + rep);
            ksoa.generateRandomPoints(config.numPoints, 0.0, 1000.0);
            ksoa.runParallelMemoryOptimized(config.threadCount, config.chunkSize, config.schedulePolicy);

            // Record SoA results
            result.runtimes_ms.push_back(ksoa.getTotalRuntimeMs());
            result.iterationsExecuted = ksoa.getIterationsExecuted();
            result.converged = (result.iterationsExecuted < config.maxIterations);
            if (!ksoa.validateAssignments()) {
                std::cerr << "WARNING: Validation failed for soa\n";
                result.correctnessValidated = false;
            } else {
                result.correctnessValidated = true;
            }
            continue; // skip remaining KMeans-based recording
        }

        // Validate correctness
        if (!kmeans.validateAssignments()) {
            std::cerr << "WARNING: Validation failed for " << config.implementation << "\n";
            result.correctnessValidated = false;
        } else {
            result.correctnessValidated = true;
        }

        result.runtimes_ms.push_back(kmeans.getTotalRuntimeMs());
        result.iterationsExecuted = kmeans.getIterationsExecuted();
        result.converged = (result.iterationsExecuted < config.maxIterations);
    }
}

void BenchmarkRunner::computeStatistics(BenchmarkResult& result) {
    if (result.runtimes_ms.empty()) {
        result.mean_runtime_ms = 0.0;
        result.stddev_ms = 0.0;
        return;
    }

    // Compute mean
    double sum = std::accumulate(result.runtimes_ms.begin(), result.runtimes_ms.end(), 0.0);
    result.mean_runtime_ms = sum / result.runtimes_ms.size();

    // Compute standard deviation
    if (result.runtimes_ms.size() > 1) {
        double sum_sq_diff = 0.0;
        for (const auto& t : result.runtimes_ms) {
            const double diff = t - result.mean_runtime_ms;
            sum_sq_diff += diff * diff;
        }
        result.stddev_ms = std::sqrt(sum_sq_diff / (result.runtimes_ms.size() - 1));
    } else {
        result.stddev_ms = 0.0;
    }
}

void BenchmarkRunner::displayProgress(const ExperimentConfig& config, int expNumber) {
    std::cout << "[" << std::setw(4) << expNumber << "/" << totalExperiments_ << "] "
              << std::setw(11) << config.implementation
              << " | " << std::setw(7) << config.schedulePolicy
              << " | Threads=" << std::setw(2) << config.threadCount
              << " | Points=" << std::setw(7) << config.numPoints
              << " | Clusters=" << std::setw(2) << config.numClusters
              << " | Chunk=" << std::setw(4) << config.chunkSize << "\n";
}

void BenchmarkRunner::exportToCSV(const std::string& outputFilePath) {
    std::ofstream file(outputFilePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << outputFilePath << "\n";
        return;
    }

    // Write CSV header
    file << "implementation,schedule,chunk_size,threads,points,clusters,iterations,"
         << "runtime_ms,mean_runtime_ms,stddev_ms,speedup,efficiency,correct\n";

    // Write each result
    for (const auto& result : allResults_) {
        // For each individual run
        for (const auto& rt : result.runtimes_ms) {
            file << result.config.implementation << ","
                 << result.config.schedulePolicy << ","
                 << result.config.chunkSize << ","
                 << result.config.threadCount << ","
                 << result.config.numPoints << ","
                 << result.config.numClusters << ","
                 << result.iterationsExecuted << ","
                 << std::fixed << std::setprecision(6) << rt << ","
                 << result.mean_runtime_ms << ","
                 << result.stddev_ms << ","
                 << result.speedup << ","
                 << result.efficiency << ","
                 << (result.correctnessValidated ? "1" : "0") << "\n";
        }
    }

    file.close();
    std::cout << "CSV exported to: " << outputFilePath << "\n";
}
