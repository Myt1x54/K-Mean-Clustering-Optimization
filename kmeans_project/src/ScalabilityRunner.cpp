#include "ScalabilityRunner.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <filesystem>

#include "KMeans.h"
#include "KMeansSoA.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
std::string csvKey(const std::string& implementation, int threads, std::size_t points, int clusters, const std::string& schedule, int chunkSize) {
    std::ostringstream out;
    out << implementation << '|' << threads << '|' << points << '|' << clusters << '|' << schedule;
    if (chunkSize >= 0) {
        out << '|' << chunkSize;
    }
    return out.str();
}
}

ScalabilityRunner::ScalabilityRunner()
    : sequentialBaselineMs_(0.0), detectedMaxThreads_(1) {}

void ScalabilityRunner::ensureOutputDirectories(const ScalabilityConfig& config) const {
    std::filesystem::create_directories(config.outputDir);
    std::filesystem::create_directories(config.outputDir + "/figures");
    std::filesystem::create_directories(config.outputDir + "/tables");
}

int ScalabilityRunner::hardwareThreadLimit() const {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

std::vector<int> ScalabilityRunner::capThreadList(const std::vector<int>& requested) const {
    std::vector<int> out;
    const int maxThreads = hardwareThreadLimit();
    for (int t : requested) {
        if (t <= 0) continue;
        if (t <= maxThreads) {
            out.push_back(t);
        }
    }
    if (out.empty()) {
        out.push_back(std::min(1, maxThreads));
    }
    return out;
}

double ScalabilityRunner::computeMean(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double ScalabilityRunner::computeStddev(const std::vector<double>& values, double mean) const {
    if (values.size() < 2) return 0.0;
    double sumSq = 0.0;
    for (double value : values) {
        const double diff = value - mean;
        sumSq += diff * diff;
    }
    return std::sqrt(sumSq / static_cast<double>(values.size() - 1));
}

ScalabilityRunner::ExperimentSummary ScalabilityRunner::runExperiment(
    const ScalabilityConfig& config,
    const std::string& implementation,
    int threads,
    std::size_t points,
    int clusters,
    const std::string& schedule,
    int chunkSize) const {

    ExperimentSummary summary;
    summary.runtimes.reserve(static_cast<std::size_t>(config.repetitions));

    for (int rep = 0; rep < config.repetitions; ++rep) {
        const unsigned int seed = config.randomSeed + static_cast<unsigned int>(rep);

        if (implementation == "sequential") {
            KMeans seq(clusters, config.maxIterations, config.convergenceThreshold, seed);
            seq.generateRandomPoints(points, config.minCoordinate, config.maxCoordinate);
            seq.run();
            summary.runtimes.push_back(seq.getTotalRuntimeMs());
            summary.iterations = seq.getIterationsExecuted();
            summary.correctnessValidated = summary.correctnessValidated && seq.validateAssignments();
            continue;
        }

        if (implementation == "naive") {
            KMeans naive(clusters, config.maxIterations, config.convergenceThreshold, seed);
            naive.generateRandomPoints(points, config.minCoordinate, config.maxCoordinate);
            naive.runParallel(threads, chunkSize);
            summary.runtimes.push_back(naive.getTotalRuntimeMs());
            summary.iterations = naive.getIterationsExecuted();
            summary.correctnessValidated = summary.correctnessValidated && naive.validateAssignments();
            continue;
        }

        if (implementation == "optimized") {
            KMeans optimized(clusters, config.maxIterations, config.convergenceThreshold, seed);
            optimized.generateRandomPoints(points, config.minCoordinate, config.maxCoordinate);
            optimized.runParallelOptimized(threads, chunkSize, schedule);
            summary.runtimes.push_back(optimized.getTotalRuntimeMs());
            summary.iterations = optimized.getIterationsExecuted();
            summary.correctnessValidated = summary.correctnessValidated && optimized.validateAssignments();
            continue;
        }

        if (implementation == "soa") {
            KMeansSoA soa(clusters, config.maxIterations, config.convergenceThreshold, seed);
            soa.generateRandomPoints(points, config.minCoordinate, config.maxCoordinate);
            soa.runParallelMemoryOptimized(threads, chunkSize, schedule);
            summary.runtimes.push_back(soa.getTotalRuntimeMs());
            summary.iterations = soa.getIterationsExecuted();
            summary.correctnessValidated = summary.correctnessValidated && soa.validateAssignments();
            continue;
        }
    }

    return summary;
}

ScalabilityRunner::ProfilingMetrics ScalabilityRunner::loadProfilingMetrics(
    const ScalabilityConfig& config,
    const std::string& implementation,
    int threads,
    std::size_t points,
    int clusters,
    const std::string& schedule,
    int chunkSize) const {

    ProfilingMetrics metrics;
    std::ifstream file(config.profilingCsvPath);
    if (!file.is_open()) return metrics;

    std::string headerLine;
    if (!std::getline(file, headerLine)) return metrics;

    std::vector<std::string> headers;
    std::stringstream headerStream(headerLine);
    std::string header;
    while (std::getline(headerStream, header, ',')) headers.push_back(header);

    std::map<std::string, std::size_t> columnIndex;
    for (std::size_t i = 0; i < headers.size(); ++i) columnIndex[headers[i]] = i;

    const bool hasChunkColumn = columnIndex.find("chunk_size") != columnIndex.end();

    std::map<std::string, std::vector<double>> accumulator;
    std::string line;
    const std::string wantedKey = csvKey(implementation, threads, points, clusters, schedule, hasChunkColumn ? chunkSize : -1);

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream row(line);
        std::vector<std::string> cells;
        std::string cell;
        while (std::getline(row, cell, ',')) cells.push_back(cell);
        if (cells.size() < headers.size()) continue;

        const std::string rowKey = csvKey(
            cells[columnIndex["implementation"]],
            std::stoi(cells[columnIndex["threads"]]),
            static_cast<std::size_t>(std::stoull(cells[columnIndex["points"]])),
            std::stoi(cells[columnIndex["clusters"]]),
            cells[columnIndex["schedule"]],
            hasChunkColumn ? std::stoi(cells[columnIndex["chunk_size"]]) : -1);

        if (rowKey != wantedKey) continue;

        auto pushValue = [&](const std::string& name) {
            auto it = columnIndex.find(name);
            if (it == columnIndex.end()) return;
            accumulator[name].push_back(std::stod(cells[it->second]));
        };

        pushValue("ipc");
        pushValue("cache_miss_rate");
        pushValue("runtime_ms");
    }

    metrics.ipc = computeMean(accumulator["ipc"]);
    metrics.cacheMissRate = computeMean(accumulator["cache_miss_rate"]);
    metrics.runtimeMs = computeMean(accumulator["runtime_ms"]);
    return metrics;
}

double ScalabilityRunner::estimateArithmeticIntensity(const std::string& implementation, std::size_t /*points*/, int clusters) const {
    const double flopsPerDistance = 6.0;
    const double flopsPerPoint = flopsPerDistance * static_cast<double>(clusters);
    const double bytesPerPointAoS = 32.0;
    const double bytesPerPointSoA = 24.0;
    const double bytesPerPoint = (implementation == "soa") ? bytesPerPointSoA : bytesPerPointAoS;
    return flopsPerPoint / bytesPerPoint;
}

double ScalabilityRunner::estimateAchievedGflops(std::size_t points, int clusters, double runtimeMs) const {
    if (runtimeMs <= 0.0) return 0.0;
    const double flops = static_cast<double>(points) * static_cast<double>(clusters) * 6.0;
    return flops / (runtimeMs / 1000.0) / 1.0e9;
}

double ScalabilityRunner::estimateBandwidthGBs(std::size_t points, int clusters, double runtimeMs, const std::string& implementation) const {
    if (runtimeMs <= 0.0) return 0.0;
    const double bytesPerPoint = (implementation == "soa") ? 24.0 : 32.0;
    const double bytes = static_cast<double>(points) * static_cast<double>(clusters) * bytesPerPoint;
    return bytes / (runtimeMs / 1000.0) / 1.0e9;
}

void ScalabilityRunner::exportCsv(const ScalabilityConfig& config) const {
    const std::string outPath = config.outputDir + "/scalability_results.csv";
    std::ofstream out(outPath);
    out << "implementation,schedule,chunk_size,threads,points,clusters,iterations,runtime_ms,mean_runtime_ms,stddev_ms,speedup,efficiency,ipc,cache_miss_rate,ai_estimate,achieved_gflops,bandwidth_gbs,correct\n";
    for (const auto& result : results_) {
        out << result.implementation << ',' << result.schedule << ',' << result.chunkSize << ','
            << result.threads << ',' << result.points << ',' << result.clusters << ','
            << result.iterations << ',' << result.runtimeMs << ',' << result.meanRuntimeMs << ','
            << result.stddevMs << ',' << result.speedup << ',' << result.efficiency << ','
            << result.ipc << ',' << result.cacheMissRate << ',' << result.arithmeticIntensity << ','
            << result.achievedGflops << ',' << result.estimatedBandwidthGBs << ','
            << (result.correctnessValidated ? 1 : 0) << '\n';
    }
}

void ScalabilityRunner::exportRooflineMetrics(const ScalabilityConfig& config) const {
    const std::string outPath = config.outputDir + "/roofline_metrics.csv";
    std::ofstream out(outPath);
    out << "implementation,threads,points,clusters,ai_estimate,achieved_gflops,bandwidth_gbs,ipc,cache_miss_rate,runtime_ms\n";
    for (const auto& result : results_) {
        out << result.implementation << ',' << result.threads << ',' << result.points << ',' << result.clusters << ','
            << result.arithmeticIntensity << ',' << result.achievedGflops << ',' << result.estimatedBandwidthGBs << ','
            << result.ipc << ',' << result.cacheMissRate << ',' << result.meanRuntimeMs << '\n';
    }
}

void ScalabilityRunner::exportMarkdownSummary(const ScalabilityConfig& config) const {
    const std::string outPath = config.outputDir + "/tables/scalability_summary.md";
    std::ofstream out(outPath);
    out << "| Threads | Impl | Runtime (ms) | Speedup | Efficiency | IPC | Interpretation |\n";
    out << "|---:|---|---:|---:|---:|---:|---|\n";
    for (const auto& result : results_) {
        std::string interpretation;
        if (result.implementation == "naive") {
            interpretation = "sync-bound, poor scaling";
        } else if (result.implementation == "optimized") {
            interpretation = "less contention, memory-limited";
        } else if (result.implementation == "soa") {
            interpretation = "better locality, higher IPC";
        } else {
            interpretation = "baseline";
        }
        out << "| " << result.threads << " | " << result.implementation << " | " << std::fixed << std::setprecision(3)
            << result.meanRuntimeMs << " | " << result.speedup << " | " << result.efficiency << " | " << result.ipc
            << " | " << interpretation << " |\n";
    }
}

void ScalabilityRunner::exportTables(const ScalabilityConfig& config) const {
    exportCsv(config);
    exportRooflineMetrics(config);
    exportMarkdownSummary(config);
}

void ScalabilityRunner::execute(const ScalabilityConfig& config) {
    ensureOutputDirectories(config);
    results_.clear();
    detectedMaxThreads_ = hardwareThreadLimit();

    const std::vector<int> cappedThreads = config.capThreadsToHardware ? capThreadList(config.threadCounts) : config.threadCounts;
    std::cout << "Detected max OpenMP threads: " << detectedMaxThreads_ << "\n";
    if (config.capThreadsToHardware && cappedThreads.size() != config.threadCounts.size()) {
        std::cout << "Thread counts capped to available hardware.\n";
    }

    for (const auto& points : config.pointCounts) {
        for (const auto& clusters : config.clusterCounts) {
            double sequentialBaselineForDataset = 0.0;
            if (config.includeSequentialBaseline) {
                const auto seqSummary = runExperiment(config, "sequential", 1, points, clusters, "static", 1000);
                sequentialBaselineForDataset = computeMean(seqSummary.runtimes);

                ScalabilityResult seqResult;
                seqResult.implementation = "sequential";
                seqResult.schedule = "static";
                seqResult.chunkSize = 1000;
                seqResult.threads = 1;
                seqResult.points = points;
                seqResult.clusters = clusters;
                seqResult.iterations = seqSummary.iterations;
                seqResult.meanRuntimeMs = sequentialBaselineForDataset;
                seqResult.stddevMs = computeStddev(seqSummary.runtimes, sequentialBaselineForDataset);
                seqResult.runtimeMs = sequentialBaselineForDataset;
                seqResult.speedup = 1.0;
                seqResult.efficiency = 1.0;
                seqResult.correctnessValidated = seqSummary.correctnessValidated;
                seqResult.arithmeticIntensity = estimateArithmeticIntensity("sequential", points, clusters);
                seqResult.achievedGflops = estimateAchievedGflops(points, clusters, sequentialBaselineForDataset);
                seqResult.estimatedBandwidthGBs = estimateBandwidthGBs(points, clusters, sequentialBaselineForDataset, "sequential");
                results_.push_back(seqResult);
            }

            for (const auto& implementation : config.implementations) {
                if (implementation == "sequential") continue;

                for (const auto& schedule : config.schedulePolicies) {
                    for (int chunkSize : config.chunkSizes) {
                        for (int threads : cappedThreads) {
                            const auto summary = runExperiment(config, implementation, threads, points, clusters, schedule, chunkSize);
                            const double meanRuntime = computeMean(summary.runtimes);
                            const double stddevRuntime = computeStddev(summary.runtimes, meanRuntime);

                            ScalabilityResult result;
                            result.implementation = implementation;
                            result.schedule = schedule;
                            result.chunkSize = chunkSize;
                            result.threads = threads;
                            result.points = points;
                            result.clusters = clusters;
                            result.iterations = summary.iterations;
                            result.runtimeMs = meanRuntime;
                            result.meanRuntimeMs = meanRuntime;
                            result.stddevMs = stddevRuntime;
                            result.speedup = (sequentialBaselineForDataset > 0.0 && meanRuntime > 0.0) ? sequentialBaselineForDataset / meanRuntime : 0.0;
                            result.efficiency = (threads > 0) ? result.speedup / static_cast<double>(threads) : 0.0;
                            result.correctnessValidated = summary.correctnessValidated;
                            result.arithmeticIntensity = estimateArithmeticIntensity(implementation, points, clusters);
                            result.achievedGflops = estimateAchievedGflops(points, clusters, meanRuntime);
                            result.estimatedBandwidthGBs = estimateBandwidthGBs(points, clusters, meanRuntime, implementation);

                            const auto metrics = loadProfilingMetrics(config, implementation, threads, points, clusters, schedule, chunkSize);
                            result.ipc = metrics.ipc;
                            result.cacheMissRate = metrics.cacheMissRate;

                            results_.push_back(result);
                        }
                    }
                }
            }
        }
    }

    exportTables(config);
    std::cout << "Scalability analysis complete. Results written to " << config.outputDir << "/\n";
}
