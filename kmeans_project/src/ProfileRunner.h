#pragma once

#include <string>
#include <vector>
#include <map>

struct ProfileConfig {
    std::string implementation;
    std::string targetCommand;
    int threads = 1;
    std::string schedule = "static";
    int chunk_size = 1000;
    std::size_t points = 100000;
    int clusters = 10;
    int repetitions = 3;
    std::string output_dir = "profiling";
};

class ProfileRunner {
public:
    explicit ProfileRunner(const std::string& exePath);
    // run profiling for the single config, writes raw perf and CSV summary
    bool run(const ProfileConfig& cfg);

private:
    std::string exePath_;
    std::string perfBin_;
    std::vector<std::string> perfEvents() const;
    bool ensureDir(const std::string& path) const;
    bool resolvePerfBinary();
    bool isExecutable(const std::string& path) const;
    bool parsePerfValues(const std::string& perfFile, std::map<std::string,double>& valuesOut) const;
    bool parsePerfOutput(const std::string& perfFile, const ProfileConfig& cfg, std::string& csvLine) const;
    bool isPerfAvailable() const;
};
