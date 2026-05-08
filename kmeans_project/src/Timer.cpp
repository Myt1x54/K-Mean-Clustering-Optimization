#include "Timer.h"

Timer::Timer() : startTime_(Clock::now()) {}

void Timer::start() noexcept {
    startTime_ = Clock::now();
}

double Timer::elapsedMilliseconds() const noexcept {
    return std::chrono::duration<double, std::milli>(Clock::now() - startTime_).count();
}
