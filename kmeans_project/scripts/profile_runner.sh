#!/usr/bin/env bash
set -euo pipefail

# Simple helper script to run profiling for an implementation.
# Usage: ./profile_runner.sh IMPLEMENTATION THREADS POINTS CLUSTERS [SCHEDULE] [CHUNK]

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="${HERE}/.."
BUILD="${ROOT}/build"
EXE="${BUILD}/kmeans"

IMPLEMENTATION=${1:-optimized}
THREADS=${2:-8}
POINTS=${3:-100000}
CLUSTERS=${4:-10}
SCHEDULE=${5:-static}
CHUNK=${6:-1000}

OUTDIR="${ROOT}/profiling/${IMPLEMENTATION}"
mkdir -p "${OUTDIR}"

PERF_EVENTS="cache-misses,cache-references,instructions,cycles,task-clock,branches,branch-misses,context-switches,page-faults"
OUTFILE="${OUTDIR}/perf_t${THREADS}_p${POINTS}_c${CLUSTERS}.txt"

resolve_perf_bin() {
	if [[ -n "${PERF_BIN:-}" && -x "${PERF_BIN}" ]]; then
		printf '%s' "${PERF_BIN}"
		return 0
	fi

	local candidates=(
		"/usr/lib/linux-tools/5.15.0-177-generic/perf"
		"/usr/lib/linux-tools/5.15.0-177/perf"
	)
	local candidate
	for candidate in "${candidates[@]}"; do
		if [[ -x "${candidate}" ]]; then
			printf '%s' "${candidate}"
			return 0
		fi
	done

	if command -v perf >/dev/null 2>&1; then
		command -v perf
		return 0
	fi

	return 1
}

PERF_BIN_RESOLVED=$(resolve_perf_bin || true)
if [[ -z "${PERF_BIN_RESOLVED}" ]]; then
	echo "Error: no working perf binary found. Set PERF_BIN=/path/to/perf or install a kernel-matched perf package." >&2
	echo "Example: PERF_BIN=/usr/lib/linux-tools/5.15.0-177-generic/perf ./scripts/profile_runner.sh optimized 8 50000 10" >&2
	exit 2
fi

echo "Profiling ${IMPLEMENTATION} threads=${THREADS} points=${POINTS} clusters=${CLUSTERS} schedule=${SCHEDULE} chunk=${CHUNK}"
echo "Command: ${PERF_BIN_RESOLVED} stat -e ${PERF_EVENTS} -- ${EXE} ${IMPLEMENTATION} --threads ${THREADS} --points ${POINTS} --clusters ${CLUSTERS} --schedule ${SCHEDULE} --chunk ${CHUNK}"

"${PERF_BIN_RESOLVED}" stat -e ${PERF_EVENTS} -- ${EXE} ${IMPLEMENTATION} --threads ${THREADS} --points ${POINTS} --clusters ${CLUSTERS} --schedule ${SCHEDULE} --chunk ${CHUNK} 2> "${OUTFILE}"

echo "Saved raw perf output to ${OUTFILE}"

echo "To summarize multiple runs, run the C++ ProfileRunner or aggregate the files under profiling/."
