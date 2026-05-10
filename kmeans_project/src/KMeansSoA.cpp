#include "KMeansSoA.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Timer.h"

KMeansSoA::KMeansSoA(int k, int maxIterations, double convergenceThreshold, unsigned int seed)
    : K_(k), maxIterations_(maxIterations), convergenceThreshold_(convergenceThreshold), rng_(seed),
      iterationsExecuted_(0), converged_(false), totalRuntimeMs_(0.0) {
    if (K_ <= 0) throw std::invalid_argument("K must be positive.");
}

void KMeansSoA::generateRandomPoints(std::size_t numPoints, double minCoordinate, double maxCoordinate) {
    if (numPoints == 0) throw std::invalid_argument("Number of points must be > 0");

    xs_.clear(); ys_.clear(); clusters_ids_.clear();
    xs_.reserve(numPoints); ys_.reserve(numPoints); clusters_ids_.reserve(numPoints);

    std::uniform_real_distribution<double> distribution(minCoordinate, maxCoordinate);
    for (std::size_t i = 0; i < numPoints; ++i) {
        xs_.push_back(distribution(rng_));
        ys_.push_back(distribution(rng_));
        clusters_ids_.push_back(-1);
    }
}

double KMeansSoA::computeEuclideanDistance(double x1, double y1, double x2, double y2) const noexcept {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return std::sqrt(dx*dx + dy*dy);
}

void KMeansSoA::initializeCentroids() {
    if (xs_.empty()) throw std::runtime_error("Cannot initialize centroids without points.");

    clusters_.clear();
    clusters_.reserve(static_cast<std::size_t>(K_));

    // Choose first K distinct points as centroids (deterministic-ish)
    std::size_t n = xs_.size();
    for (int i = 0; i < K_; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i) % n;
        clusters_.emplace_back(i, xs_[idx], ys_[idx]);
    }
}

double KMeansSoA::updateCentroids(bool& convergedAll) {
    double maxMovement = 0.0;
    convergedAll = true;

    for (Cluster& c : clusters_) {
        const double oldx = c.getCentroidX();
        const double oldy = c.getCentroidY();
        if (c.getSize() > 0) {
            c.updateCentroid();
        }
        const double dx = c.getCentroidX() - oldx;
        const double dy = c.getCentroidY() - oldy;
        const double move = std::sqrt(dx*dx + dy*dy);
        if (move > maxMovement) maxMovement = move;
        if (move > convergenceThreshold_) convergedAll = false;
    }
    return maxMovement;
}

void KMeansSoA::runParallelMemoryOptimized(int numThreads, int chunkSize, const std::string& schedulePolicy) {
    if (xs_.empty()) throw std::runtime_error("No points: call generateRandomPoints first.");

    initializeCentroids();

    iterationTimesMs_.clear();
    iterationTimesMs_.reserve(static_cast<std::size_t>(maxIterations_));

    iterationsExecuted_ = 0;
    converged_ = false;

    Timer totalTimer; totalTimer.start();

#ifdef _OPENMP
    const int usedThreads = (numThreads > 0) ? numThreads : omp_get_max_threads();
    omp_set_num_threads(usedThreads);
#else
    const int usedThreads = 1;
#endif

    const int T = usedThreads;
    const std::size_t Ksz = static_cast<std::size_t>(K_);
    const std::size_t N = xs_.size();

    // Padding to reduce false sharing: stride = K + pad
    const int pad = 8; // conservative padding (8 doubles)
    const std::size_t stride = static_cast<std::size_t>(K_) + static_cast<std::size_t>(pad);
    const std::size_t totalSize = static_cast<std::size_t>(T) * stride;

    // Aligned allocations for thread-local accumulators
    std::vector<double> local_sum_x(totalSize);
    std::vector<double> local_sum_y(totalSize);
    std::vector<int> local_count(totalSize);

    for (int iter = 1; iter <= maxIterations_; ++iter) {
        Timer iterationTimer; iterationTimer.start();

        // Reset cluster accumulators
        for (Cluster& c : clusters_) c.resetAccumulators();

        std::fill(local_sum_x.begin(), local_sum_x.end(), 0.0);
        std::fill(local_sum_y.begin(), local_sum_y.end(), 0.0);
        std::fill(local_count.begin(), local_count.end(), 0);

        int pointsReassigned = 0;

        // Assignment phase (thread-local updates into padded arrays)
        Timer assignTimer; assignTimer.start();

#ifdef _OPENMP
        omp_sched_t sched_kind = omp_sched_static;
        if (schedulePolicy == "dynamic") sched_kind = omp_sched_dynamic;
        else if (schedulePolicy == "guided") sched_kind = omp_sched_guided;
        omp_set_schedule(sched_kind, chunkSize);

        #pragma omp parallel for schedule(runtime) reduction(+:pointsReassigned)
        for (std::size_t i = 0; i < N; ++i) {
            const double px = xs_[i];
            const double py = ys_[i];

            int bestClusterId = 0;
            double bestDistance = computeEuclideanDistance(px, py, clusters_[0].getCentroidX(), clusters_[0].getCentroidY());
            for (int cid = 1; cid < K_; ++cid) {
                const double d = computeEuclideanDistance(px, py, clusters_[cid].getCentroidX(), clusters_[cid].getCentroidY());
                if (d < bestDistance) { bestDistance = d; bestClusterId = cid; }
            }

            if (clusters_ids_[i] != bestClusterId) {
                pointsReassigned++;
                clusters_ids_[i] = bestClusterId;
            }

            const int tid = omp_get_thread_num();
            const std::size_t base = static_cast<std::size_t>(tid) * stride;
            const std::size_t idx = base + static_cast<std::size_t>(bestClusterId);
            local_sum_x[idx] += px;
            local_sum_y[idx] += py;
            local_count[idx] += 1;
        }
#else
        // Sequential fallback
        for (std::size_t i = 0; i < N; ++i) {
            const double px = xs_[i];
            const double py = ys_[i];

            int bestClusterId = 0;
            double bestDistance = computeEuclideanDistance(px, py, clusters_[0].getCentroidX(), clusters_[0].getCentroidY());
            for (int cid = 1; cid < K_; ++cid) {
                const double d = computeEuclideanDistance(px, py, clusters_[cid].getCentroidX(), clusters_[cid].getCentroidY());
                if (d < bestDistance) { bestDistance = d; bestClusterId = cid; }
            }
            if (clusters_ids_[i] != bestClusterId) {
                pointsReassigned++;
                clusters_ids_[i] = bestClusterId;
            }
            const std::size_t base = 0;
            const std::size_t idx = base + static_cast<std::size_t>(bestClusterId);
            local_sum_x[idx] += px;
            local_sum_y[idx] += py;
            local_count[idx] += 1;
        }
#endif

        const double assignMs = assignTimer.elapsedMilliseconds();

        // Reduction phase: merge thread-local accumulators into global clusters
        Timer reductionTimer; reductionTimer.start();
        for (int t = 0; t < T; ++t) {
            const std::size_t base = static_cast<std::size_t>(t) * stride;
            for (std::size_t cid = 0; cid < Ksz; ++cid) {
                const double sx = local_sum_x[base + cid];
                const double sy = local_sum_y[base + cid];
                const int cnt = local_count[base + cid];
                if (cnt != 0) clusters_[static_cast<int>(cid)].addAccumulated(sx, sy, cnt);
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

        if (convergedAll) { converged_ = true; break; }
    }

    totalRuntimeMs_ = totalTimer.elapsedMilliseconds();
}

bool KMeansSoA::validateAssignments() const {
    for (std::size_t i = 0; i < clusters_ids_.size(); ++i) {
        const int cid = clusters_ids_[i];
        if (cid < 0 || cid >= K_) return false;
    }
    return true;
}

double KMeansSoA::getTotalRuntimeMs() const { return totalRuntimeMs_; }
int KMeansSoA::getIterationsExecuted() const { return iterationsExecuted_; }

