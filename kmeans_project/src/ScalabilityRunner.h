#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

struct ScalabilityConfig {
    std::vector<std::string> implementations = {"sequential", "naive", "optimized", "soa"};
    std::vector<int> threadCounts = {1, 2, 4, 8, 16};
    std::vector<std::size_t> pointCounts = {50000, 100000};
    std::vector<int> clusterCounts = {10};
    std::vector<std::string> schedulePolicies = {"static"};
    std::vector<int> chunkSizes = {1000};
    int maxIterations = 1000;
    double convergenceThreshold = 1e-4;
    int repetitions = 3;
    unsigned int randomSeed = 42;
    double minCoordinate = 0.0;
    double maxCoordinate = 1000.0;
    bool includeSequentialBaseline = true;
    bool capThreadsToHardware = true;
    std::string outputDir = "report";
    std::string profilingCsvPath = "profiling/profiling_results.csv";
};

struct ScalabilityResult {
    std::string implementation;
    std::string schedule;
    int chunkSize = 0;
    int threads = 1;
    std::size_t points = 0;
    int clusters = 0;
    int iterations = 0;
    double runtimeMs = 0.0;
    double meanRuntimeMs = 0.0;
    double stddevMs = 0.0;
    double speedup = 0.0;
    double efficiency = 0.0;
    double ipc = 0.0;
    double cacheMissRate = 0.0;
    double arithmeticIntensity = 0.0;
    double achievedGflops = 0.0;
    double estimatedBandwidthGBs = 0.0;
    bool correctnessValidated = false;
};

class ScalabilityRunner {
public:
    ScalabilityRunner();

    void execute(const ScalabilityConfig& config);
    const std::vector<ScalabilityResult>& results() const { return results_; }

private:
    std::vector<ScalabilityResult> results_;
    double sequentialBaselineMs_;
    int detectedMaxThreads_;

    struct ProfilingMetrics {
        double ipc = 0.0;
        double cacheMissRate = 0.0;
        double runtimeMs = 0.0;
        double instructions = 0.0;
        double cycles = 0.0;
        double cacheMisses = 0.0;
        double cacheReferences = 0.0;
    };

    struct ExperimentSummary {
        std::vector<double> runtimes;
        int iterations = 0;
        bool correctnessValidated = true;
    };

    void ensureOutputDirectories(const ScalabilityConfig& config) const;
    int hardwareThreadLimit() const;
    std::vector<int> capThreadList(const std::vector<int>& requested) const;
    ExperimentSummary runExperiment(const ScalabilityConfig& config, const std::string& implementation, int threads, std::size_t points, int clusters, const std::string& schedule, int chunkSize) const;
    double computeMean(const std::vector<double>& values) const;
    double computeStddev(const std::vector<double>& values, double mean) const;
    ProfilingMetrics loadProfilingMetrics(const ScalabilityConfig& config, const std::string& implementation, int threads, std::size_t points, int clusters, const std::string& schedule, int chunkSize) const;
    double estimateArithmeticIntensity(const std::string& implementation, std::size_t points, int clusters) const;
    double estimateAchievedGflops(std::size_t points, int clusters, double runtimeMs) const;
    double estimateBandwidthGBs(std::size_t points, int clusters, double runtimeMs, const std::string& implementation) const;
    void exportCsv(const ScalabilityConfig& config) const;
    void exportMarkdownSummary(const ScalabilityConfig& config) const;
    void exportRooflineMetrics(const ScalabilityConfig& config) const;
    void exportTables(const ScalabilityConfig& config) const;
};
