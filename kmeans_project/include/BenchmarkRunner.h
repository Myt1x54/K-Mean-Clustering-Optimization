#ifndef BENCHMARK_RUNNER_H
#define BENCHMARK_RUNNER_H

#include <string>
#include <vector>
#include <cstddef>

// Experiment configuration for a single benchmark run
struct ExperimentConfig {
    std::string implementation;  // "sequential", "naive", "optimized"
    std::string schedulePolicy;  // "static", "dynamic", "guided" (ignored for sequential/naive)
    int chunkSize;
    int threadCount;
    std::size_t numPoints;
    int numClusters;
    int maxIterations;
    double convergenceThreshold;
    unsigned int randomSeed;
    int repetitions;  // Number of runs for statistics
};

// Result from a single benchmark experiment
struct BenchmarkResult {
    ExperimentConfig config;
    std::vector<double> runtimes_ms;  // Individual run times
    double mean_runtime_ms;
    double stddev_ms;
    double speedup;
    double efficiency;
    int iterationsExecuted;
    bool converged;
    bool correctnessValidated;

    BenchmarkResult() 
        : mean_runtime_ms(0.0), stddev_ms(0.0), speedup(1.0), efficiency(0.0),
          iterationsExecuted(0), converged(false), correctnessValidated(false) {}
};

// Main benchmark orchestration class
class BenchmarkRunner {
public:
    BenchmarkRunner();

    // Run all experiments and collect results
    void executeFullBenchmark();

    // Export results to CSV
    void exportToCSV(const std::string& outputFilePath);

    // Get all collected results
    const std::vector<BenchmarkResult>& getResults() const { return allResults_; }

    // Set reference sequential baseline for speedup/efficiency calculations
    void setSequentialBaseline(double baselineMs) { sequentialBaseline_ = baselineMs; }

private:
    std::vector<BenchmarkResult> allResults_;
    double sequentialBaseline_;
    int totalExperiments_;
    int currentExperiment_;

    // Helper methods
    void runSingleExperiment(const ExperimentConfig& config, BenchmarkResult& result);
    void computeStatistics(BenchmarkResult& result);
    void displayProgress(const ExperimentConfig& config, int expNumber);
    void validateCorrectness(const std::string& impl, bool& isValid);

    // CSV export helpers
    void writeCSVHeader(const std::string& filepath);
    void appendResultToCSV(const std::string& filepath, const BenchmarkResult& result);
};

#endif
