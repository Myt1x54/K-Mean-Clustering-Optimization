#ifndef KMEANS_SOA_H
#define KMEANS_SOA_H

#include <random>
#include <vector>
#include <string>

#include "Cluster.h"

class KMeansSoA {
public:
    KMeansSoA(int k, int maxIterations, double convergenceThreshold = 1e-4, unsigned int seed = 42);

    void generateRandomPoints(std::size_t numPoints, double minCoordinate, double maxCoordinate);

    // Memory-optimized SoA parallel run
    void runParallelMemoryOptimized(int numThreads, int chunkSize, const std::string& schedulePolicy = "static");

    bool validateAssignments() const;
    double getTotalRuntimeMs() const;
    int getIterationsExecuted() const;

private:
    int K_;
    int maxIterations_;
    double convergenceThreshold_;

    std::mt19937 rng_;

    std::vector<double> xs_;
    std::vector<double> ys_;
    std::vector<int> clusters_ids_;

    std::vector<Cluster> clusters_;

    int iterationsExecuted_;
    bool converged_;
    double totalRuntimeMs_;
    std::vector<double> iterationTimesMs_;

    double computeEuclideanDistance(double x1, double y1, double x2, double y2) const noexcept;
    void initializeCentroids();
    double updateCentroids(bool& convergedAll);
};

#endif
