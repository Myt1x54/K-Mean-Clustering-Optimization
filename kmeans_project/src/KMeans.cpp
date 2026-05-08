#include "KMeans.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "Timer.h"

KMeans::KMeans(int k, int maxIterations, double convergenceThreshold, unsigned int seed)
    : K_(k),
      maxIterations_(maxIterations),
      convergenceThreshold_(convergenceThreshold),
      rng_(seed),
      iterationsExecuted_(0),
      converged_(false),
      totalRuntimeMs_(0.0) {
    if (K_ <= 0) {
        throw std::invalid_argument("K must be positive.");
    }
    if (maxIterations_ <= 0) {
        throw std::invalid_argument("maxIterations must be positive.");
    }
    if (convergenceThreshold_ <= 0.0) {
        throw std::invalid_argument("convergenceThreshold must be > 0.");
    }
}

void KMeans::generateRandomPoints(std::size_t numPoints, double minCoordinate, double maxCoordinate) {
    if (numPoints == 0) {
        throw std::invalid_argument("Number of points must be greater than zero.");
    }

    points_.clear();
    points_.reserve(numPoints);

    std::uniform_real_distribution<double> distribution(minCoordinate, maxCoordinate);

    // O(N): contiguous push-back keeps storage cache-friendly for later assignment passes.
    for (std::size_t i = 0; i < numPoints; ++i) {
        points_.emplace_back(distribution(rng_), distribution(rng_), -1);
    }
}

void KMeans::initializeCentroids() {
    if (points_.empty()) {
        throw std::runtime_error("Cannot initialize centroids without points.");
    }

    clusters_.clear();
    clusters_.reserve(static_cast<std::size_t>(K_));

    // Initialize each centroid from a random existing point.
    // Sampling with replacement keeps behavior valid even when K > N.
    std::uniform_int_distribution<std::size_t> indexDistribution(0, points_.size() - 1);

    for (int clusterId = 0; clusterId < K_; ++clusterId) {
        const Point& seedPoint = points_[indexDistribution(rng_)];
        clusters_.emplace_back(clusterId, seedPoint.getX(), seedPoint.getY());
    }
}

double KMeans::computeEuclideanDistance(double x1, double y1, double x2, double y2) const noexcept {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

int KMeans::assignPointsToNearestCluster() {
    int pointsReassigned = 0;

    // O(N*K): dominant cost in sequential K-Means.
    // This loop shape is intentionally simple for future OpenMP parallelization.
    for (Point& point : points_) {
        int bestClusterId = 0;
        double bestDistance = computeEuclideanDistance(
            point.getX(), point.getY(), clusters_[0].getCentroidX(), clusters_[0].getCentroidY());

        for (int clusterId = 1; clusterId < K_; ++clusterId) {
            const double distance = computeEuclideanDistance(
                point.getX(), point.getY(), clusters_[clusterId].getCentroidX(), clusters_[clusterId].getCentroidY());

            if (distance < bestDistance) {
                bestDistance = distance;
                bestClusterId = clusterId;
            }
        }

        if (point.getClusterId() != bestClusterId) {
            ++pointsReassigned;
            point.setClusterId(bestClusterId);
        }

        clusters_[bestClusterId].addPoint(point.getX(), point.getY());
    }

    return pointsReassigned;
}

double KMeans::updateCentroids(bool& convergedAll) {
    convergedAll = true;
    double maxMovement = 0.0;

    // O(K): centroid updates are cheap relative to assignment.
    for (Cluster& cluster : clusters_) {
        const double movement = cluster.updateCentroid();
        maxMovement = std::max(maxMovement, movement);

        if (!cluster.movementBelowThreshold(convergenceThreshold_)) {
            convergedAll = false;
        }
    }

    return maxMovement;
}

void KMeans::run() {
    if (points_.empty()) {
        throw std::runtime_error("No points available. Call generateRandomPoints() first.");
    }

    initializeCentroids();

    iterationTimesMs_.clear();
    iterationTimesMs_.reserve(static_cast<std::size_t>(maxIterations_));

    iterationsExecuted_ = 0;
    converged_ = false;
    totalRuntimeMs_ = 0.0;

    Timer totalTimer;
    totalTimer.start();

    std::cout << "\nStarting K-Means (sequential baseline)\n";
    std::cout << "Points: " << points_.size() << ", Clusters: " << K_
              << ", Max Iterations: " << maxIterations_ << "\n\n";

    for (int iter = 1; iter <= maxIterations_; ++iter) {
        Timer iterationTimer;
        iterationTimer.start();

        // Reset per-cluster accumulators once per iteration.
        for (Cluster& cluster : clusters_) {
            cluster.resetAccumulators();
        }

        const int pointsReassigned = assignPointsToNearestCluster();

        bool convergedAll = false;
        const double maxMovement = updateCentroids(convergedAll);

        const double iterationMs = iterationTimer.elapsedMilliseconds();
        iterationTimesMs_.push_back(iterationMs);
        iterationsExecuted_ = iter;

        std::cout << "Iteration " << std::setw(4) << iter
                  << " | reassigned: " << std::setw(8) << pointsReassigned
                  << " | max movement: " << std::setw(12) << std::setprecision(6) << std::fixed << maxMovement
                  << " | time (ms): " << std::setw(10) << std::setprecision(3) << iterationMs << "\n";

        if (convergedAll) {
            converged_ = true;
            break;
        }
    }

    totalRuntimeMs_ = totalTimer.elapsedMilliseconds();
}

void KMeans::printStatistics() const {
    double averageIterationMs = 0.0;
    if (!iterationTimesMs_.empty()) {
        double sum = 0.0;
        for (const double t : iterationTimesMs_) {
            sum += t;
        }
        averageIterationMs = sum / static_cast<double>(iterationTimesMs_.size());
    }

    std::cout << "\n===== Final Statistics =====\n";
    std::cout << "Total points:         " << points_.size() << "\n";
    std::cout << "Total clusters:       " << K_ << "\n";
    std::cout << "Iterations executed:  " << iterationsExecuted_ << "\n";
    std::cout << "Total runtime (ms):   " << std::fixed << std::setprecision(3) << totalRuntimeMs_ << "\n";
    std::cout << "Avg iteration (ms):   " << std::fixed << std::setprecision(3) << averageIterationMs << "\n";
    std::cout << "Converged:            " << (converged_ ? "Yes" : "No (hit max iterations)") << "\n";

    std::cout << "\nCluster sizes:\n";
    for (const Cluster& cluster : clusters_) {
        std::cout << "  Cluster " << cluster.getId() << ": " << cluster.getSize() << " points\n";
    }

    std::cout << "============================\n";
}

bool KMeans::validateAssignments() const {
    // O(N): each point must have exactly one valid cluster assignment.
    for (const Point& point : points_) {
        const int clusterId = point.getClusterId();
        if (clusterId < 0 || clusterId >= K_) {
            return false;
        }
    }
    return true;
}
