#ifndef POINT_H
#define POINT_H

class Point {
public:
    Point();
    Point(double x, double y, int clusterId = -1);

    double getX() const noexcept;
    double getY() const noexcept;
    int getClusterId() const noexcept;

    void setX(double x) noexcept;
    void setY(double y) noexcept;
    void setClusterId(int clusterId) noexcept;

private:
    double x_;
    double y_;
    int clusterId_;
};

#endif
