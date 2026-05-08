#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    Timer();

    void start() noexcept;
    double elapsedMilliseconds() const noexcept;

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point startTime_;
};

#endif
