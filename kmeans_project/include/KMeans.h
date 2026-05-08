#ifndef KMEANS_H
#define KMEANS_H

#include <random>
#include <vector>

#include "Cluster.h"
#include "Point.h"

class KMeans {
public:
    KMeans(int k, int maxIterations, double convergenceThreshold = 1e-4, unsigned int seed = 42);

    void generateRandomPoints(std::size_t numPoints, double minCoordinate, double maxCoordinate);
    void initializeCentroids();

    double computeEuclideanDistance(double x1, double y1, double x2, double y2) const noexcept;
    int assignPointsToNearestCluster();
    double updateCentroids(bool& convergedAll);

    void run();
    void runParallel(int numThreads, int chunkSize);
    void printStatistics() const;
    bool validateAssignments() const;

    // Accessors for external benchmarking and comparison
    const std::vector<Point>& getPoints() const;
    void setPoints(const std::vector<Point>& pts);
    double getTotalRuntimeMs() const;
    int getIterationsExecuted() const;
    const std::vector<double>& getIterationTimesMs() const;

private:
    std::vector<Point> points_;
    std::vector<Cluster> clusters_;

    int K_;
    int maxIterations_;
    double convergenceThreshold_;

    std::mt19937 rng_;

    int iterationsExecuted_;
    bool converged_;
    double totalRuntimeMs_;
    std::vector<double> iterationTimesMs_;
};

#endif
