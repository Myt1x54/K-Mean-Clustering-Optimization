#include "Point.h"

Point::Point() : x_(0.0), y_(0.0), clusterId_(-1) {}

Point::Point(double x, double y, int clusterId) : x_(x), y_(y), clusterId_(clusterId) {}

double Point::getX() const noexcept {
    return x_;
}

double Point::getY() const noexcept {
    return y_;
}

int Point::getClusterId() const noexcept {
    return clusterId_;
}

void Point::setX(double x) noexcept {
    x_ = x;
}

void Point::setY(double y) noexcept {
    y_ = y;
}

void Point::setClusterId(int clusterId) noexcept {
    clusterId_ = clusterId;
}
