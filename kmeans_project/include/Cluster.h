#ifndef CLUSTER_H
#define CLUSTER_H

class Cluster {
public:
    Cluster();
    Cluster(int id, double centroidX, double centroidY);

    int getId() const noexcept;
    double getCentroidX() const noexcept;
    double getCentroidY() const noexcept;
    double getLastMovement() const noexcept;
    int getSize() const noexcept;

    void setCentroid(double x, double y) noexcept;
    void resetAccumulators() noexcept;
    void addPoint(double x, double y) noexcept;

    // Updates the centroid using current accumulators and returns movement magnitude.
    double updateCentroid();

    bool movementBelowThreshold(double threshold) const noexcept;

private:
    int id_;
    double centroidX_;
    double centroidY_;

    // Per-iteration accumulators used for computing the new centroid.
    double sumX_;
    double sumY_;
    int size_;

    double lastMovement_;
};

#endif
