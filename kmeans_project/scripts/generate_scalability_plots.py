#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def group_mean(rows, key_fields, value_fields):
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        for field in value_fields:
            buckets[key][field].append(to_float(row.get(field)))
    summary = []
    for key, metrics in buckets.items():
        entry = dict(zip(key_fields, key))
        for field in value_fields:
            values = metrics[field]
            entry[field] = sum(values) / len(values) if values else 0.0
        summary.append(entry)
    return summary


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report and presentation assets from scalability CSV data.")
    parser.add_argument("--input-csv", default="report/scalability_results.csv", help="Input scalability CSV path")
    parser.add_argument("--outdir", default="report", help="Output base directory for figures/tables/csv summaries")
    parser.add_argument("--figures-subdir", default="figures", help="Figures subdirectory under outdir")
    parser.add_argument("--tables-subdir", default="tables", help="Tables subdirectory under outdir")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated export formats, e.g. png,pdf",
    )
    return parser.parse_args()


def normalize_rows(rows):
    normalized = []
    for row in rows:
        item = dict(row)
        if to_float(item.get("mean_runtime_ms"), 0.0) <= 0.0:
            item["mean_runtime_ms"] = item.get("runtime_ms", "0")
        normalized.append(item)
    return normalized


def save_figure(fig, figures_dir: Path, base_name: str, formats, dpi):
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(figures_dir / f"{base_name}.{fmt}", dpi=dpi)


def safe_sort_int(value):
    return to_int(value, default=0)


def plot_lines(ax, grouped_rows, x_field, y_field, label_field, title, ylabel, xlabel):
    label_values = sorted({row[label_field] for row in grouped_rows})
    for label in label_values:
        series = [row for row in grouped_rows if row[label_field] == label]
        series.sort(key=lambda row: safe_sort_int(row[x_field]))
        xs = [safe_sort_int(row[x_field]) for row in series]
        ys = [to_float(row[y_field]) for row in series]
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()


def plot_with_ideal_speedup(ax, grouped_rows, thread_field, y_field, label_field, title):
    label_values = sorted({row[label_field] for row in grouped_rows})
    max_thread = 1
    for label in label_values:
        series = [row for row in grouped_rows if row[label_field] == label]
        series.sort(key=lambda row: safe_sort_int(row[thread_field]))
        xs = [safe_sort_int(row[thread_field]) for row in series]
        ys = [to_float(row[y_field]) for row in series]
        if xs:
            max_thread = max(max_thread, max(xs))
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)

    ideal_x = list(range(1, max_thread + 1))
    ideal_y = ideal_x
    ax.plot(ideal_x, ideal_y, linestyle="--", color="black", linewidth=1.5, label="Ideal linear")
    ax.set_title(title)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.grid(True, alpha=0.25)
    ax.legend()


def write_markdown_table(path: Path, rows, columns, title):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("|" + "---|" * len(columns) + "\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n")


def write_latex_table(path: Path, rows, columns, caption, label):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[ht]\n")
        handle.write("\\centering\n")
        handle.write("\\small\n")
        handle.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        handle.write("\\hline\n")
        handle.write(" & ".join(columns) + " \\\\ \n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(" & ".join(str(row.get(column, "")) for column in columns) + " \\\\ \n")
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")
        handle.write(f"\\caption{{{caption}}}\n")
        handle.write(f"\\label{{{label}}}\n")
        handle.write("\\end{table}\n")


def write_csv_table(path: Path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def extract_metric_rows(rows, columns):
    extracted = []
    for row in rows:
        extracted.append({column: row.get(column, "") for column in columns})
    return extracted


def filter_rows(rows, required_fields):
    filtered = []
    for row in rows:
        if all(str(row.get(field, "")).strip() != "" for field in required_fields):
            filtered.append(row)
    return filtered


def scheduling_aggregate(rows):
    data = [row for row in rows if row.get("implementation") in {"naive", "optimized", "soa"}]
    return group_mean(
        data,
        ["implementation", "schedule", "threads", "chunk_size"],
        ["mean_runtime_ms", "speedup", "efficiency"],
    )


def runtime_dataset_aggregate(rows):
    return group_mean(
        rows,
        ["implementation", "points"],
        ["mean_runtime_ms"],
    )


def runtime_cluster_aggregate(rows):
    return group_mean(
        rows,
        ["implementation", "clusters"],
        ["mean_runtime_ms"],
    )


def write_analysis_summary(path: Path, runtime_table, hardware_rows, roofline_rows):
    by_impl_thread = defaultdict(list)
    for row in runtime_table:
        by_impl_thread[row["implementation"]].append(row)

    def best_speedup(impl):
        entries = by_impl_thread.get(impl, [])
        if not entries:
            return 0.0
        return max(to_float(e.get("speedup")) for e in entries)

    naive_best = best_speedup("naive")
    opt_best = best_speedup("optimized")
    soa_best = best_speedup("soa")

    ipc_by_impl = defaultdict(list)
    miss_by_impl = defaultdict(list)
    for row in hardware_rows:
        impl = row.get("implementation", "")
        ipc_by_impl[impl].append(to_float(row.get("ipc")))
        miss_by_impl[impl].append(to_float(row.get("cache_miss_rate")))

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    opt_ipc = mean(ipc_by_impl.get("optimized", []))
    soa_ipc = mean(ipc_by_impl.get("soa", []))
    opt_miss = mean(miss_by_impl.get("optimized", []))
    soa_miss = mean(miss_by_impl.get("soa", []))

    ai_by_impl = defaultdict(list)
    for row in roofline_rows:
        ai_by_impl[row.get("implementation", "")].append(to_float(row.get("ai_estimate")))
    opt_ai = mean(ai_by_impl.get("optimized", []))
    soa_ai = mean(ai_by_impl.get("soa", []))

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Final Performance Analysis Summary\n\n")
        handle.write("## Naive OpenMP\n")
        handle.write("- Naive runs show weak scaling due to synchronization overhead from shared updates and lock contention.\n")
        handle.write(f"- Best observed naive speedup is {naive_best:.3f}, indicating limited parallel efficiency at higher thread counts.\n\n")

        handle.write("## Optimized OpenMP (Thread-Local Reduction)\n")
        handle.write("- Thread-local accumulation reduces synchronization pressure and improves scalability relative to naive OpenMP.\n")
        handle.write(f"- Best observed optimized speedup is {opt_best:.3f}, reflecting better parallel utilization.\n\n")

        handle.write("## SoA Implementation\n")
        handle.write("- SoA layout improves memory locality by keeping hot coordinate streams contiguous.\n")
        handle.write(f"- Best observed SoA speedup is {soa_best:.3f}.\n")
        handle.write(f"- Mean IPC: optimized={opt_ipc:.3f}, soa={soa_ipc:.3f}. Mean cache miss rate: optimized={opt_miss:.4f}, soa={soa_miss:.4f}.\n")
        handle.write(f"- Arithmetic intensity trend: optimized={opt_ai:.4f}, soa={soa_ai:.4f}.\n\n")

        handle.write("## Hardware Bottlenecks\n")
        handle.write("- Scaling plateaus are consistent with memory bandwidth pressure and shared-cache contention at higher thread counts.\n")
        handle.write("- Efficiency drops at larger thread counts indicate parallel overhead and memory subsystem saturation.\n")
        handle.write("- Roofline positioning can be used to explain whether each implementation is closer to memory-bound or compute-bound limits.\n")


def roofline_plot(rows, figures_dir: Path, formats, dpi):
    if not rows:
        return
    selected = [row for row in rows if row.get("implementation") in {"optimized", "soa"}]
    if not selected:
        selected = rows

    ai_values = [to_float(row.get("ai_estimate")) for row in selected if to_float(row.get("ai_estimate")) > 0.0]
    perf_values = [to_float(row.get("achieved_gflops")) for row in selected if to_float(row.get("achieved_gflops")) > 0.0]
    bw_values = [to_float(row.get("bandwidth_gbs")) for row in selected if to_float(row.get("bandwidth_gbs")) > 0.0]
    if not ai_values or not perf_values:
        return

    compute_ceiling = max(perf_values)
    memory_ceiling = max(bw_values) if bw_values else max(perf_values)
    x_max = max(ai_values) * 1.4
    x_min = max(min(ai_values) / 4.0, 0.01)
    xs = [x_min * (x_max / x_min) ** (i / 200.0) for i in range(201)]
    roof = [min(compute_ceiling, memory_ceiling * x) for x in xs]

    fig, ax = plt.subplots(figsize=(8, 6))
    for impl in sorted({row["implementation"] for row in selected}):
        impl_rows = [row for row in selected if row["implementation"] == impl]
        ax.scatter(
            [to_float(row.get("ai_estimate")) for row in impl_rows],
            [to_float(row.get("achieved_gflops")) for row in impl_rows],
            label=impl,
            s=60,
        )
    ax.plot(xs, roof, linestyle="--", color="black", linewidth=2, label="Empirical roofline")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("Achieved performance (GFLOP/s)")
    ax.set_title("Empirical Roofline")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir, "roofline_plot", formats, dpi)
    save_figure(fig, figures_dir, "roofline_analysis", formats, dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{row.get('implementation', '')}-{row.get('threads', '')}" for row in selected]
    ax.bar(labels, [to_float(row.get("ai_estimate")) for row in selected])
    ax.set_title("Arithmetic Intensity by Run")
    ax.set_ylabel("FLOPs/byte")
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, figures_dir, "arithmetic_intensity", formats, dpi)
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = root / input_csv

    report_dir = Path(args.outdir)
    if not report_dir.is_absolute():
        report_dir = root / report_dir

    figures_dir = report_dir / args.figures_subdir
    tables_dir = report_dir / args.tables_subdir
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    if not formats:
        formats = ["png"]

    if not input_csv.exists():
        raise SystemExit(f"Missing scalability CSV: {input_csv}")

    rows = normalize_rows(read_csv_rows(input_csv))
    if not rows:
        raise SystemExit("Scalability CSV is empty.")

    grouped = group_mean(
        rows,
        ["implementation", "threads"],
        ["mean_runtime_ms", "speedup", "efficiency", "ipc", "cache_miss_rate", "achieved_gflops"],
    )

    runtime_table = sorted(grouped, key=lambda row: (row["implementation"], safe_sort_int(row["threads"])))
    write_markdown_table(
        tables_dir / "scalability_summary.md",
        runtime_table,
        ["implementation", "threads", "mean_runtime_ms", "speedup", "efficiency", "ipc", "cache_miss_rate"],
        "Scalability Summary",
    )

    write_latex_table(
        tables_dir / "scalability_summary.tex",
        runtime_table,
        ["implementation", "threads", "mean_runtime_ms", "speedup", "efficiency", "ipc", "cache_miss_rate"],
        "Scalability summary across implementations and thread counts",
        "tab:scalability_summary",
    )

    write_csv_table(
        report_dir / "speedup_results.csv",
        runtime_table,
        ["implementation", "threads", "mean_runtime_ms", "speedup"],
    )
    write_csv_table(
        report_dir / "efficiency_results.csv",
        runtime_table,
        ["implementation", "threads", "mean_runtime_ms", "efficiency"],
    )
    write_csv_table(
        report_dir / "ipc_results.csv",
        runtime_table,
        ["implementation", "threads", "ipc", "cache_miss_rate"],
    )

    runtime_summary_rows = extract_metric_rows(
        sorted(
            rows,
            key=lambda row: (
                row.get("implementation", ""),
                safe_sort_int(row.get("points", 0)),
                safe_sort_int(row.get("clusters", 0)),
                safe_sort_int(row.get("threads", 0)),
                row.get("schedule", ""),
                safe_sort_int(row.get("chunk_size", 0)),
            ),
        ),
        ["implementation", "schedule", "chunk_size", "threads", "points", "clusters", "mean_runtime_ms"],
    )
    write_csv_table(
        report_dir / "final_runtime_summary.csv",
        runtime_summary_rows,
        ["implementation", "schedule", "chunk_size", "threads", "points", "clusters", "mean_runtime_ms"],
    )
    write_csv_table(
        report_dir / "final_speedup_summary.csv",
        extract_metric_rows(
            rows,
            ["implementation", "schedule", "chunk_size", "threads", "points", "clusters", "speedup"],
        ),
        ["implementation", "schedule", "chunk_size", "threads", "points", "clusters", "speedup"],
    )
    write_csv_table(
        report_dir / "final_efficiency_summary.csv",
        extract_metric_rows(
            rows,
            ["implementation", "schedule", "chunk_size", "threads", "points", "clusters", "efficiency"],
        ),
        ["implementation", "schedule", "chunk_size", "threads", "points", "clusters", "efficiency"],
    )
    write_csv_table(
        report_dir / "final_hardware_summary.csv",
        extract_metric_rows(
            rows,
            [
                "implementation",
                "schedule",
                "chunk_size",
                "threads",
                "points",
                "clusters",
                "ipc",
                "cache_miss_rate",
                "cycles",
                "instructions",
                "cache_references",
                "cache_misses",
                "branch_misses",
                "cpu_utilization",
            ],
        ),
        [
            "implementation",
            "schedule",
            "chunk_size",
            "threads",
            "points",
            "clusters",
            "ipc",
            "cache_miss_rate",
            "cycles",
            "instructions",
            "cache_references",
            "cache_misses",
            "branch_misses",
            "cpu_utilization",
        ],
    )
    write_csv_table(
        report_dir / "final_roofline_summary.csv",
        extract_metric_rows(
            rows,
            ["implementation", "schedule", "chunk_size", "threads", "points", "clusters", "ai_estimate", "achieved_gflops", "bandwidth_gbs"],
        ),
        ["implementation", "schedule", "chunk_size", "threads", "points", "clusters", "ai_estimate", "achieved_gflops", "bandwidth_gbs"],
    )

    roofline_table = sorted(
        (
            {
                "implementation": row["implementation"],
                "threads": row["threads"],
                "points": row.get("points", ""),
                "clusters": row.get("clusters", ""),
                "ai_estimate": row["ai_estimate"],
                "achieved_gflops": row["achieved_gflops"],
                "bandwidth_gbs": row["bandwidth_gbs"],
            }
            for row in rows
        ),
        key=lambda row: (
            row["implementation"],
            safe_sort_int(row["points"]),
            safe_sort_int(row["clusters"]),
            safe_sort_int(row["threads"]),
        ),
    )
    write_markdown_table(
        tables_dir / "roofline_summary.md",
        roofline_table,
        ["implementation", "threads", "points", "clusters", "ai_estimate", "achieved_gflops", "bandwidth_gbs"],
        "Roofline Summary",
    )
    write_latex_table(
        tables_dir / "roofline_summary.tex",
        roofline_table,
        ["implementation", "threads", "points", "clusters", "ai_estimate", "achieved_gflops", "bandwidth_gbs"],
        "Roofline-related metrics summary",
        "tab:roofline_summary",
    )

    hardware_rows = sorted(
        group_mean(
            rows,
            ["implementation", "threads", "points", "clusters"],
            ["ipc", "cache_miss_rate", "cycles", "instructions", "cache_misses", "cache_references", "branch_misses", "cpu_utilization"],
        ),
        key=lambda row: (
            row.get("implementation", ""),
            safe_sort_int(row.get("points", 0)),
            safe_sort_int(row.get("clusters", 0)),
            safe_sort_int(row.get("threads", 0)),
        ),
    )
    write_csv_table(
        report_dir / "cache_analysis.csv",
        hardware_rows,
        ["implementation", "threads", "points", "clusters", "cache_misses", "cache_references", "cache_miss_rate", "branch_misses"],
    )
    write_csv_table(
        report_dir / "ipc_analysis.csv",
        hardware_rows,
        ["implementation", "threads", "points", "clusters", "ipc", "cycles", "instructions", "cpu_utilization"],
    )
    write_markdown_table(
        tables_dir / "hardware_summary.md",
        hardware_rows,
        ["implementation", "threads", "points", "clusters", "ipc", "cache_miss_rate", "cycles", "instructions", "cpu_utilization"],
        "Hardware Counter Summary",
    )
    write_latex_table(
        tables_dir / "hardware_summary.tex",
        hardware_rows,
        ["implementation", "threads", "points", "clusters", "ipc", "cache_miss_rate", "cycles", "instructions", "cpu_utilization"],
        "Hardware counter summary",
        "tab:hardware_summary",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lines(ax, runtime_table, "threads", "mean_runtime_ms", "implementation", "Runtime vs Threads", "Runtime (ms)", "Threads")
    save_figure(fig, figures_dir, "runtime_scaling", formats, args.dpi)
    save_figure(fig, figures_dir, "runtime_threads", formats, args.dpi)
    plt.close(fig)

    dataset_runtime = sorted(
        runtime_dataset_aggregate(rows),
        key=lambda row: (row["implementation"], safe_sort_int(row["points"])),
    )
    if dataset_runtime:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_lines(
            ax,
            dataset_runtime,
            "points",
            "mean_runtime_ms",
            "implementation",
            "Runtime vs Dataset Size",
            "Runtime (ms)",
            "Points",
        )
        save_figure(fig, figures_dir, "runtime_dataset_size", formats, args.dpi)
        plt.close(fig)

    cluster_runtime = sorted(
        runtime_cluster_aggregate(rows),
        key=lambda row: (row["implementation"], safe_sort_int(row["clusters"])),
    )
    if cluster_runtime:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_lines(
            ax,
            cluster_runtime,
            "clusters",
            "mean_runtime_ms",
            "implementation",
            "Runtime vs Cluster Count",
            "Runtime (ms)",
            "Clusters",
        )
        save_figure(fig, figures_dir, "runtime_clusters", formats, args.dpi)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_with_ideal_speedup(ax, runtime_table, "threads", "speedup", "implementation", "Speedup vs Threads")
    save_figure(fig, figures_dir, "speedup_scaling", formats, args.dpi)
    plt.close(fig)

    parallel_only = [row for row in runtime_table if row.get("implementation") in {"naive", "optimized", "soa"}]
    if parallel_only:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_with_ideal_speedup(ax, parallel_only, "threads", "speedup", "implementation", "Speedup Comparison")
        save_figure(fig, figures_dir, "speedup_comparison", formats, args.dpi)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lines(ax, runtime_table, "threads", "efficiency", "implementation", "Efficiency vs Threads", "Efficiency", "Threads")
    save_figure(fig, figures_dir, "efficiency_scaling", formats, args.dpi)
    plt.close(fig)

    if parallel_only:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_lines(ax, parallel_only, "threads", "efficiency", "implementation", "Efficiency Comparison", "Efficiency", "Threads")
        save_figure(fig, figures_dir, "efficiency_comparison", formats, args.dpi)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lines(ax, runtime_table, "threads", "ipc", "implementation", "IPC vs Threads", "IPC", "Threads")
    save_figure(fig, figures_dir, "ipc_scaling", formats, args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lines(
        ax,
        runtime_table,
        "threads",
        "cache_miss_rate",
        "implementation",
        "Cache Miss Rate vs Threads",
        "Cache miss rate",
        "Threads",
    )
    save_figure(fig, figures_dir, "cache_behavior", formats, args.dpi)
    plt.close(fig)

    sched_rows = sorted(
        scheduling_aggregate(rows),
        key=lambda row: (
            row.get("implementation", ""),
            row.get("schedule", ""),
            safe_sort_int(row.get("chunk_size", 0)),
            safe_sort_int(row.get("threads", 0)),
        ),
    )
    if sched_rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        schedule_runtime_view = [
            {
                "label": f"{row['implementation']}:{row['schedule']}:c{row['chunk_size']}",
                "threads": row["threads"],
                "mean_runtime_ms": row["mean_runtime_ms"],
            }
            for row in sched_rows
        ]
        plot_lines(
            ax,
            schedule_runtime_view,
            "threads",
            "mean_runtime_ms",
            "label",
            "Scheduling Runtime Comparison",
            "Runtime (ms)",
            "Threads",
        )
        save_figure(fig, figures_dir, "scheduling_runtime", formats, args.dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        schedule_speedup_view = [
            {
                "label": f"{row['implementation']}:{row['schedule']}:c{row['chunk_size']}",
                "threads": row["threads"],
                "speedup": row["speedup"],
            }
            for row in sched_rows
        ]
        plot_with_ideal_speedup(ax, schedule_speedup_view, "threads", "speedup", "label", "Scheduling Speedup Comparison")
        save_figure(fig, figures_dir, "scheduling_speedup", formats, args.dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        schedule_eff_view = [
            {
                "label": f"{row['implementation']}:{row['schedule']}:c{row['chunk_size']}",
                "threads": row["threads"],
                "efficiency": row["efficiency"],
            }
            for row in sched_rows
        ]
        plot_lines(
            ax,
            schedule_eff_view,
            "threads",
            "efficiency",
            "label",
            "Scheduling Comparison (Efficiency)",
            "Efficiency",
            "Threads",
        )
        save_figure(fig, figures_dir, "scheduling_comparison", formats, args.dpi)
        plt.close(fig)

    roofline_plot(rows, figures_dir, formats, args.dpi)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("mean_runtime_ms", "Runtime (ms)", axes[0, 0], "Runtime vs Threads"),
        ("speedup", "Speedup", axes[0, 1], "Speedup vs Threads"),
        ("efficiency", "Efficiency", axes[0, 2], "Efficiency vs Threads"),
        ("ipc", "IPC", axes[1, 0], "IPC vs Threads"),
        ("cache_miss_rate", "Cache Miss Rate", axes[1, 1], "Cache Miss Rate vs Threads"),
    ]
    for field, ylabel, ax, title in panels:
        if field == "speedup":
            plot_with_ideal_speedup(ax, runtime_table, "threads", field, "implementation", title)
        else:
            plot_lines(ax, runtime_table, "threads", field, "implementation", title, ylabel, "Threads")
    axes[1, 2].axis("off")
    save_figure(fig, figures_dir, "scalability_summary", formats, args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_with_ideal_speedup(ax, parallel_only if parallel_only else runtime_table, "threads", "speedup", "implementation", "Scalability Plot")
    save_figure(fig, figures_dir, "scalability_plot", formats, args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if parallel_only:
        plot_lines(axes[0], parallel_only, "threads", "mean_runtime_ms", "implementation", "Runtime (Parallel)", "Runtime (ms)", "Threads")
        plot_lines(axes[1], parallel_only, "threads", "efficiency", "implementation", "Efficiency (Parallel)", "Efficiency", "Threads")
    else:
        plot_lines(axes[0], runtime_table, "threads", "mean_runtime_ms", "implementation", "Runtime", "Runtime (ms)", "Threads")
        plot_lines(axes[1], runtime_table, "threads", "efficiency", "implementation", "Efficiency", "Efficiency", "Threads")
    save_figure(fig, figures_dir, "presentation_scalability_overview", formats, args.dpi)
    plt.close(fig)

    final_table_rows = sorted(
        group_mean(
            rows,
            ["implementation", "threads"],
            ["mean_runtime_ms", "speedup", "efficiency", "ipc", "cache_miss_rate", "ai_estimate", "achieved_gflops", "bandwidth_gbs"],
        ),
        key=lambda row: (row["implementation"], safe_sort_int(row["threads"])),
    )
    final_columns = [
        "implementation",
        "threads",
        "mean_runtime_ms",
        "speedup",
        "efficiency",
        "ipc",
        "cache_miss_rate",
        "ai_estimate",
        "achieved_gflops",
        "bandwidth_gbs",
    ]
    write_markdown_table(
        tables_dir / "final_comparison_summary.md",
        final_table_rows,
        final_columns,
        "Final Runtime/Speedup/Efficiency/Hardware Comparison",
    )
    write_latex_table(
        tables_dir / "final_comparison_summary.tex",
        final_table_rows,
        final_columns,
        "Final runtime, scaling, and hardware comparison",
        "tab:final_comparison",
    )
    write_csv_table(
        report_dir / "comparison_table_presentation.csv",
        final_table_rows,
        final_columns,
    )

    write_analysis_summary(tables_dir / "analysis_summary.md", runtime_table, hardware_rows, roofline_table)
    print(f"Generated report assets under: {report_dir}")
    print(f"Figures: {figures_dir}")
    print(f"Tables: {tables_dir}")


if __name__ == "__main__":
    main()
