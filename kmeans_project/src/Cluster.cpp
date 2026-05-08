#include "Cluster.h"

#include <cmath>

Cluster::Cluster()
    : id_(-1),
      centroidX_(0.0),
      centroidY_(0.0),
      sumX_(0.0),
      sumY_(0.0),
      size_(0),
      lastMovement_(0.0) {}

Cluster::Cluster(int id, double centroidX, double centroidY)
    : id_(id),
      centroidX_(centroidX),
      centroidY_(centroidY),
      sumX_(0.0),
      sumY_(0.0),
      size_(0),
      lastMovement_(0.0) {}

int Cluster::getId() const noexcept {
    return id_;
}

double Cluster::getCentroidX() const noexcept {
    return centroidX_;
}

double Cluster::getCentroidY() const noexcept {
    return centroidY_;
}

double Cluster::getLastMovement() const noexcept {
    return lastMovement_;
}

int Cluster::getSize() const noexcept {
    return size_;
}

void Cluster::setCentroid(double x, double y) noexcept {
    centroidX_ = x;
    centroidY_ = y;
}

void Cluster::resetAccumulators() noexcept {
    sumX_ = 0.0;
    sumY_ = 0.0;
    size_ = 0;
}

void Cluster::addPoint(double x, double y) noexcept {
    sumX_ += x;
    sumY_ += y;
    ++size_;
}

double Cluster::updateCentroid() {
    // Empty clusters keep their previous centroid to avoid invalid division.
    if (size_ == 0) {
        lastMovement_ = 0.0;
        return lastMovement_;
    }

    const double newX = sumX_ / static_cast<double>(size_);
    const double newY = sumY_ / static_cast<double>(size_);

    const double dx = newX - centroidX_;
    const double dy = newY - centroidY_;
    lastMovement_ = std::sqrt(dx * dx + dy * dy);

    centroidX_ = newX;
    centroidY_ = newY;

    return lastMovement_;
}

bool Cluster::movementBelowThreshold(double threshold) const noexcept {
    return lastMovement_ < threshold;
}
