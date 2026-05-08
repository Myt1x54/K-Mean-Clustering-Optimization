#include "KMeans.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "Timer.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstring>

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

void KMeans::runParallel(int numThreads, int chunkSize) {
#ifdef _OPENMP
    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }
#else
    (void)numThreads;
#endif

    if (points_.empty()) {
        throw std::runtime_error("No points available. Call generateRandomPoints() or setPoints() first.");
    }

    initializeCentroids();

    iterationTimesMs_.clear();
    iterationTimesMs_.reserve(static_cast<std::size_t>(maxIterations_));

    iterationsExecuted_ = 0;
    converged_ = false;

    Timer totalTimer;
    totalTimer.start();

    std::cout << "\nStarting K-Means (naive OpenMP baseline)\n";
    std::cout << "Points: " << points_.size() << ", Clusters: " << K_
              << ", Max Iterations: " << maxIterations_ << "\n";

#ifdef _OPENMP
    std::cout << "Configured threads: " << numThreads
              << " | OpenMP reports max threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP not available in this build; running sequentially.\n";
#endif

    for (int iter = 1; iter <= maxIterations_; ++iter) {
        Timer iterationTimer;
        iterationTimer.start();

        for (Cluster& cluster : clusters_) {
            cluster.resetAccumulators();
        }

        // Naive parallel assignment: each thread finds nearest cluster for its points
        // and updates the shared cluster accumulators inside a critical section.
        // This reproduces the synchronization bottleneck described in the paper.
        int pointsReassigned = 0;

#ifdef _OPENMP
        const int N = static_cast<int>(points_.size());
        #pragma omp parallel
        {
            int localReassigned = 0;
            #pragma omp for schedule(static, chunkSize)
            for (int i = 0; i < N; ++i) {
                double px = points_[static_cast<std::size_t>(i)].getX();
                double py = points_[static_cast<std::size_t>(i)].getY();

                int bestClusterId = 0;
                double bestDistance = computeEuclideanDistance(
                    px, py, clusters_[0].getCentroidX(), clusters_[0].getCentroidY());

                for (int cid = 1; cid < K_; ++cid) {
                    const double d = computeEuclideanDistance(px, py, clusters_[cid].getCentroidX(), clusters_[cid].getCentroidY());
                    if (d < bestDistance) {
                        bestDistance = d;
                        bestClusterId = cid;
                    }
                }

                if (points_[static_cast<std::size_t>(i)].getClusterId() != bestClusterId) {
                    localReassigned++;
                    points_[static_cast<std::size_t>(i)].setClusterId(bestClusterId);
                }

                // Critical section to protect shared cluster accumulators per paper.
                #pragma omp critical
                {
                    clusters_[bestClusterId].addPoint(px, py);
                }
            }
            #pragma omp atomic
            pointsReassigned += localReassigned;
        }
#else
        // If OpenMP not available, fall back to sequential assignment.
        pointsReassigned = assignPointsToNearestCluster();
#endif

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

void KMeans::runParallelOptimized(int numThreads, int chunkSize, const std::string& schedulePolicy) {
#ifdef _OPENMP
    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }
#else
    (void)numThreads;
#endif

    if (points_.empty()) {
        throw std::runtime_error("No points available. Call generateRandomPoints() or setPoints() first.");
    }

    initializeCentroids();

    iterationTimesMs_.clear();
    iterationTimesMs_.reserve(static_cast<std::size_t>(maxIterations_));

    iterationsExecuted_ = 0;
    converged_ = false;

    Timer totalTimer;
    totalTimer.start();

    std::cout << "\nStarting K-Means (optimized OpenMP baseline)\n";
    std::cout << "Points: " << points_.size() << ", Clusters: " << K_
              << ", Max Iterations: " << maxIterations_ << "\n";

#ifdef _OPENMP
    const int usedThreads = (numThreads > 0) ? numThreads : omp_get_max_threads();
    std::cout << "Configured threads: " << usedThreads
              << " | OpenMP reports max threads: " << omp_get_max_threads() << "\n";
#else
    const int usedThreads = 1;
    std::cout << "OpenMP not available; running single-threaded optimized version.\n";
#endif

    // Preallocate thread-local accumulators as flat arrays: [thread * K_ + cluster]
    const int T = usedThreads;
    const std::size_t Ksz = static_cast<std::size_t>(K_);

    std::vector<double> local_sum_x(static_cast<std::size_t>(T) * Ksz);
    std::vector<double> local_sum_y(static_cast<std::size_t>(T) * Ksz);
    std::vector<int> local_count(static_cast<std::size_t>(T) * Ksz);

    for (int iter = 1; iter <= maxIterations_; ++iter) {
        Timer iterationTimer;
        iterationTimer.start();

        // Reset global cluster accumulators
        for (Cluster& cluster : clusters_) {
            cluster.resetAccumulators();
        }

        // Reset local accumulators
        std::fill(local_sum_x.begin(), local_sum_x.end(), 0.0);
        std::fill(local_sum_y.begin(), local_sum_y.end(), 0.0);
        std::fill(local_count.begin(), local_count.end(), 0);

        int pointsReassigned = 0;

        // Measure assignment & reduction phases separately for workload analysis.
        Timer assignTimer;
        assignTimer.start();

        // Parallel assignment: update only thread-local accumulators
        // Use runtime scheduling so we can switch between static/dynamic/guided at runtime.
        // Configure scheduling policy for OpenMP runtime.
    #ifdef _OPENMP
        omp_sched_t sched_kind = omp_sched_static;
        if (schedulePolicy == "dynamic") sched_kind = omp_sched_dynamic;
        else if (schedulePolicy == "guided") sched_kind = omp_sched_guided;
        omp_set_schedule(sched_kind, chunkSize);

        const int N = static_cast<int>(points_.size());
        #pragma omp parallel for schedule(runtime) reduction(+:pointsReassigned)
        for (int i = 0; i < N; ++i) {
            const double px = points_[static_cast<std::size_t>(i)].getX();
            const double py = points_[static_cast<std::size_t>(i)].getY();

            int bestClusterId = 0;
            double bestDistance = computeEuclideanDistance(
                px, py, clusters_[0].getCentroidX(), clusters_[0].getCentroidY());

            for (int cid = 1; cid < K_; ++cid) {
                const double d = computeEuclideanDistance(px, py, clusters_[cid].getCentroidX(), clusters_[cid].getCentroidY());
                if (d < bestDistance) {
                    bestDistance = d;
                    bestClusterId = cid;
                }
            }

            if (points_[static_cast<std::size_t>(i)].getClusterId() != bestClusterId) {
                pointsReassigned++;
                points_[static_cast<std::size_t>(i)].setClusterId(bestClusterId);
            }

            const int tid = omp_get_thread_num();
            const std::size_t idx = static_cast<std::size_t>(tid) * Ksz + static_cast<std::size_t>(bestClusterId);
            local_sum_x[idx] += px;
            local_sum_y[idx] += py;
            local_count[idx] += 1;
        }
    #else
        // Fallback to sequential path
        pointsReassigned = assignPointsToNearestCluster();
    #endif

        const double assignMs = assignTimer.elapsedMilliseconds();

        // Reduction: merge thread-local accumulators into global cluster accumulators
        Timer reductionTimer;
        reductionTimer.start();
        for (int t = 0; t < T; ++t) {
            const std::size_t base = static_cast<std::size_t>(t) * Ksz;
            for (std::size_t cid = 0; cid < Ksz; ++cid) {
                const double sx = local_sum_x[base + cid];
                const double sy = local_sum_y[base + cid];
                const int cnt = local_count[base + cid];
                if (cnt != 0) {
                    clusters_[static_cast<int>(cid)].addAccumulated(sx, sy, cnt);
                }
            }
        }

        const double reductionMs = reductionTimer.elapsedMilliseconds();

        bool convergedAll = false;
        const double maxMovement = updateCentroids(convergedAll);

        const double iterationMs = iterationTimer.elapsedMilliseconds();
        iterationTimesMs_.push_back(iterationMs);
        iterationsExecuted_ = iter;

        std::cout << "Iteration " << std::setw(4) << iter
                  << " | reassigned: " << std::setw(8) << pointsReassigned
              << " | max movement: " << std::setw(12) << std::setprecision(6) << std::fixed << maxMovement
              << " | assign (ms): " << std::setw(8) << std::setprecision(3) << assignMs
              << " | reduce (ms): " << std::setw(8) << std::setprecision(3) << reductionMs
              << " | iter (ms): " << std::setw(8) << std::setprecision(3) << iterationMs << "\n";

        if (convergedAll) {
            converged_ = true;
            break;
        }
    }

    totalRuntimeMs_ = totalTimer.elapsedMilliseconds();
}

bool KMeans::compareCentroids(const std::vector<Cluster>& a, const std::vector<Cluster>& b, double tol) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double dx = a[i].getCentroidX() - b[i].getCentroidX();
        const double dy = a[i].getCentroidY() - b[i].getCentroidY();
        if (std::sqrt(dx*dx + dy*dy) > tol) return false;
    }
    return true;
}

bool KMeans::compareAssignments(const std::vector<Point>& a, const std::vector<Point>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].getClusterId() != b[i].getClusterId()) return false;
    }
    return true;
}

const std::vector<Point>& KMeans::getPoints() const { return points_; }

void KMeans::setPoints(const std::vector<Point>& pts) { points_ = pts; }

double KMeans::getTotalRuntimeMs() const { return totalRuntimeMs_; }

int KMeans::getIterationsExecuted() const { return iterationsExecuted_; }

const std::vector<double>& KMeans::getIterationTimesMs() const { return iterationTimesMs_; }

const std::vector<Cluster>& KMeans::getClusters() const { return clusters_; }

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
