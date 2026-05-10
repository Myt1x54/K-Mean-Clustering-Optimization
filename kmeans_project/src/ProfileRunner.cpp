#include "ProfileRunner.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <filesystem>

ProfileRunner::ProfileRunner(const std::string& exePath)
    : exePath_(exePath) {
    resolvePerfBinary();
}

namespace {
std::string trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\n\r");
    if (begin == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t\n\r");
    return value.substr(begin, end - begin + 1);
}
}

std::vector<std::string> ProfileRunner::perfEvents() const {
    return {"cache-misses", "cache-references", "instructions", "cycles", "task-clock", "branches", "branch-misses", "context-switches", "page-faults"};
}

bool ProfileRunner::ensureDir(const std::string& path) const {
    struct stat st{};
    if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) return true;
    std::string cmd = "mkdir -p " + path;
    return system(cmd.c_str()) == 0;
}

bool ProfileRunner::isExecutable(const std::string& path) const {
    return !path.empty() && access(path.c_str(), X_OK) == 0;
}

bool ProfileRunner::resolvePerfBinary() {
    const char* envPerf = std::getenv("PERF_BIN");
    if (envPerf != nullptr) {
        const std::string envPath = trim(envPerf);
        if (isExecutable(envPath)) {
            perfBin_ = envPath;
            return true;
        }
    }

    std::error_code ec;
    const std::filesystem::path root{"/usr/lib/linux-tools"};
    if (std::filesystem::exists(root, ec)) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root, ec)) {
            if (ec) break;
            if (entry.is_regular_file(ec) && entry.path().filename() == "perf" && isExecutable(entry.path().string())) {
                perfBin_ = entry.path().string();
                return true;
            }
        }
    }

    const std::vector<std::string> fallbackCandidates = {
        "/usr/bin/perf",
        "/bin/perf"
    };

    for (const auto& candidate : fallbackCandidates) {
        if (isExecutable(candidate)) {
            perfBin_ = candidate;
            return true;
        }
    }

    perfBin_.clear();
    return false;
}

bool ProfileRunner::parsePerfValues(const std::string& perfFile, std::map<std::string, double>& valuesOut) const {
    std::ifstream in(perfFile);
    if (!in) return false;

    std::map<std::string, double> values;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::string numStr;
        if (!(iss >> numStr)) continue;
        numStr.erase(std::remove(numStr.begin(), numStr.end(), ','), numStr.end());
        double val = 0.0;
        try { val = std::stod(numStr); } catch (...) { continue; }

        if (line.find("cache-misses") != std::string::npos) values["cache_misses"] = val;
        else if (line.find("cache-references") != std::string::npos) values["cache_references"] = val;
        else if (line.find("instructions") != std::string::npos) values["instructions"] = val;
        else if (line.find("cycles") != std::string::npos && line.find("task-clock") == std::string::npos) values["cycles"] = val;
        else if (line.find("task-clock") != std::string::npos) values["task_clock_ms"] = val; // in ms
        else if (line.find("seconds time elapsed") != std::string::npos) values["elapsed_seconds"] = val;
        else if (line.find("branches") != std::string::npos && line.find("branch-misses") == std::string::npos) values["branches"] = val;
        else if (line.find("branch-misses") != std::string::npos) values["branch_misses"] = val;
        else if (line.find("context-switches") != std::string::npos) values["context_switches"] = val;
        else if (line.find("page-faults") != std::string::npos) values["page_faults"] = val;
    }

    valuesOut = std::move(values);
    return true;
}

bool ProfileRunner::parsePerfOutput(const std::string& perfFile, const ProfileConfig& cfg, std::string& csvLine) const {
    std::map<std::string,double> values;
    if (!parsePerfValues(perfFile, values)) return false;

    double instructions = values.count("instructions") ? values.at("instructions") : 0.0;
    double cycles = values.count("cycles") ? values.at("cycles") : 0.0;
    double cache_misses = values.count("cache_misses") ? values.at("cache_misses") : 0.0;
    double cache_refs = values.count("cache_references") ? values.at("cache_references") : 0.0;
    double ipc = (cycles > 0.0) ? (instructions / cycles) : 0.0;
    double cache_miss_rate = (cache_refs > 0.0) ? (cache_misses / cache_refs) : 0.0;

    std::ostringstream out;
    out << cfg.implementation << "," << cfg.threads << "," << cfg.schedule << "," << cfg.points << "," << cfg.clusters << ",";
    double runtime_ms = values.count("task_clock_ms") ? values.at("task_clock_ms") : 0.0;
    out << runtime_ms << ",";
    out << static_cast<uint64_t>(cache_misses) << "," << static_cast<uint64_t>(cache_refs) << ",";
    out << cache_miss_rate << ",";
    out << static_cast<uint64_t>(instructions) << "," << static_cast<uint64_t>(cycles) << "," << ipc << ",";
    out << (values.count("cpu_util") ? values.at("cpu_util") : 0.0);

    csvLine = out.str();
    return true;
}

bool ProfileRunner::run(const ProfileConfig& cfg) {
    if (!resolvePerfBinary() || perfBin_.empty()) {
        std::cerr << "Error: no working perf binary was found. Profiling requires a kernel-matched perf tool.\n";
        struct utsname uts {};
        const bool isWsl = (uname(&uts) == 0) && (std::string(uts.release).find("microsoft") != std::string::npos || std::string(uts.release).find("WSL") != std::string::npos);
        if (isWsl) {
            std::cerr << "This is a WSL kernel (" << uts.release << "), so install the matching WSL tools package first.\n";
            std::cerr << "Try: sudo apt update && sudo apt install linux-tools-" << uts.release << " linux-cloud-tools-" << uts.release << "\n";
            std::cerr << "Or install the meta packages: sudo apt install linux-tools-standard-WSL2 linux-cloud-tools-standard-WSL2\n";
        } else {
            std::cerr << "Install on Debian/Ubuntu: sudo apt update && sudo apt install linux-tools-common linux-tools-$(uname -r)\n";
            std::cerr << "Or: sudo apt install linux-perf (if available)\n";
            std::cerr << "On Fedora/CentOS: sudo dnf install perf\n";
        }
        std::cerr << "You can also override the binary explicitly with PERF_BIN=/path/to/perf.\n";
        std::cerr << "Or use your distribution's package manager to install a kernel-matched 'perf' tool.\n";
        std::cerr << "As a fallback, use the script 'scripts/profile_runner.sh' after installing perf, or collect platform counters with other tools.\n";
        return false;
    }
    if (!ensureDir(cfg.output_dir)) return false;
    std::string implDir = cfg.output_dir + "/" + cfg.implementation;
    if (!ensureDir(implDir)) return false;

    // Run multiple repetitions and aggregate
    std::vector<std::map<std::string,double>> runs;
    auto events = perfEvents();
    for (int r = 0; r < cfg.repetitions; ++r) {
        std::ostringstream perfFile;
        perfFile << implDir << "/perf_threads" << cfg.threads << "_pts" << cfg.points << "_cls" << cfg.clusters << "_run" << r << ".txt";

        const std::string targetCommand = cfg.targetCommand.empty() ? exePath_ : cfg.targetCommand;

        std::ostringstream cmd;
        cmd << perfBin_ << " stat -e ";
        for (size_t i = 0; i < events.size(); ++i) {
            if (i) cmd << ",";
            cmd << events[i];
        }
        cmd << " -- " << targetCommand;
        cmd << " 2> " << perfFile.str();

        std::cerr << "Running: " << cmd.str() << std::endl;
        int rc = system(cmd.str().c_str());
        if (rc != 0) {
            std::cerr << "perf run failed with rc=" << rc << std::endl;
            return false;
        }

        std::map<std::string,double> vals;
        if (!parsePerfValues(perfFile.str(), vals)) {
            std::cerr << "Failed to parse perf output: " << perfFile.str() << std::endl;
            return false;
        }
        runs.push_back(std::move(vals));
    }

    // Aggregate metrics across runs
    auto mean_or_zero = [&](const std::string& key)->double{
        double sum = 0.0; int cnt = 0;
        for (auto &m : runs) {
            if (m.count(key)) { sum += m.at(key); ++cnt; }
        }
        return cnt ? (sum / cnt) : 0.0;
    };

    // runtime mean and stddev (task_clock_ms)
    std::vector<double> runtimes;
    for (auto &m : runs) if (m.count("task_clock_ms")) runtimes.push_back(m.at("task_clock_ms"));
    double runtime_mean = 0.0;
    double runtime_std = 0.0;
    if (!runtimes.empty()) {
        runtime_mean = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
        double acc = 0.0;
        for (double v : runtimes) acc += (v - runtime_mean) * (v - runtime_mean);
        runtime_std = (runtimes.size() > 1) ? std::sqrt(acc / (runtimes.size() - 1)) : 0.0;
    }

    double instructions_mean = mean_or_zero("instructions");
    double cycles_mean = mean_or_zero("cycles");
    double cache_misses_mean = mean_or_zero("cache_misses");
    double cache_refs_mean = mean_or_zero("cache_references");
    std::vector<double> cpuUtils;
    for (const auto& m : runs) {
        const auto taskClockIt = m.find("task_clock_ms");
        const auto elapsedIt = m.find("elapsed_seconds");
        if (taskClockIt != m.end() && elapsedIt != m.end() && elapsedIt->second > 0.0) {
            cpuUtils.push_back((taskClockIt->second / (elapsedIt->second * 1000.0)) * 100.0);
        }
    }
    double cpu_util_mean = 0.0;
    if (!cpuUtils.empty()) {
        cpu_util_mean = std::accumulate(cpuUtils.begin(), cpuUtils.end(), 0.0) / cpuUtils.size();
    }

    double ipc = (cycles_mean > 0.0) ? (instructions_mean / cycles_mean) : 0.0;
    double cache_miss_rate = (cache_refs_mean > 0.0) ? (cache_misses_mean / cache_refs_mean) : 0.0;

    std::ostringstream out;
    out << cfg.implementation << "," << cfg.threads << "," << cfg.schedule << "," << cfg.points << "," << cfg.clusters << ",";
    out << runtime_mean << ",";
    out << static_cast<uint64_t>(std::llround(cache_misses_mean)) << "," << static_cast<uint64_t>(std::llround(cache_refs_mean)) << ",";
    out << cache_miss_rate << ",";
    out << static_cast<uint64_t>(std::llround(instructions_mean)) << "," << static_cast<uint64_t>(std::llround(cycles_mean)) << "," << ipc << ",";
    out << cpu_util_mean << "," << runtime_std;

    std::string csvLine = out.str();

    // write or append summary
    std::string summaryFile = cfg.output_dir + "/profiling_results.csv";
    bool exists = (access(summaryFile.c_str(), F_OK) == 0);
    std::ofstream fout(summaryFile, std::ios::app);
    if (!exists) {
        fout << "implementation,threads,schedule,points,clusters,runtime_ms,cache_misses,cache_references,cache_miss_rate,instructions,cycles,ipc,cpu_utilization,runtime_stddev_ms\n";
    }
    fout << csvLine << "\n";
    fout.close();

    std::cerr << "Perf results saved under: " << cfg.output_dir << " (raw + summary)" << std::endl;
    return true;
}

bool ProfileRunner::isPerfAvailable() const {
    return !perfBin_.empty();
}
