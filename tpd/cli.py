"""
CLI tools for TPD.
Provides commands for comparing results and querying stack traces.
"""

import argparse
import json
import os
import sys
from typing import Any

from .stack import StackManager
from .storage import load_config_from_dir, load_records_from_dir


def compare_results(dir1: str, dir2: str, output_file: str, rank: int = 0, tolerance: float = 1e-6):
    """
    Compare two result directories and output differences.
    Matches records by (step, hook_type, module_name, tensor_path).
    """
    print(f"[TPD] Comparing results...", file=sys.stderr)
    print(f"  Dir1: {dir1}", file=sys.stderr)
    print(f"  Dir2: {dir2}", file=sys.stderr)
    print(f"  Rank: {rank}", file=sys.stderr)
    print(f"  Tolerance: {tolerance}", file=sys.stderr)

    # Load records
    records1 = load_records_from_dir(dir1, rank)
    records2 = load_records_from_dir(dir2, rank)

    if not records1:
        print(f"[TPD] Warning: No records found in {dir1}", file=sys.stderr)
    if not records2:
        print(f"[TPD] Warning: No records found in {dir2}", file=sys.stderr)

    # Build index: key -> record
    def build_index(records):
        index = {}
        for rec in records:
            key = (rec["step"], rec["hook_type"], rec["module_name"], rec.get("tensor_path", ""))
            index[key] = rec
        return index

    index1 = build_index(records1)
    index2 = build_index(records2)

    keys1 = set(index1.keys())
    keys2 = set(index2.keys())

    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common_keys = keys1 & keys2

    # Compare common entries
    differences = []
    for key in sorted(common_keys):
        rec1 = index1[key]
        rec2 = index2[key]

        diffs = _compare_records(rec1, rec2, tolerance)
        if diffs:
            differences.append({
                "key": {
                    "step": key[0],
                    "hook_type": key[1],
                    "module_name": key[2],
                    "tensor_path": key[3],
                },
                "stack_id_1": rec1.get("stack_id"),
                "stack_id_2": rec2.get("stack_id"),
                "differences": diffs,
            })

    # Write output
    output = {
        "summary": {
            "dir1": dir1,
            "dir2": dir2,
            "rank": rank,
            "tolerance": tolerance,
            "total_entries_dir1": len(records1),
            "total_entries_dir2": len(records2),
            "only_in_dir1": len(only_in_1),
            "only_in_dir2": len(only_in_2),
            "common_entries": len(common_keys),
            "entries_with_differences": len(differences),
        },
        "only_in_dir1": [
            {"step": k[0], "hook_type": k[1], "module_name": k[2], "tensor_path": k[3]}
            for k in sorted(only_in_1)
        ],
        "only_in_dir2": [
            {"step": k[0], "hook_type": k[1], "module_name": k[2], "tensor_path": k[3]}
            for k in sorted(only_in_2)
        ],
        "differences": differences,
    }

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[TPD] Comparison complete!", file=sys.stderr)
    print(f"  Total entries in dir1: {len(records1)}", file=sys.stderr)
    print(f"  Total entries in dir2: {len(records2)}", file=sys.stderr)
    print(f"  Only in dir1: {len(only_in_1)}", file=sys.stderr)
    print(f"  Only in dir2: {len(only_in_2)}", file=sys.stderr)
    print(f"  Common entries: {len(common_keys)}", file=sys.stderr)
    print(f"  Entries with differences: {len(differences)}", file=sys.stderr)
    print(f"  Output written to: {output_file}", file=sys.stderr)

    return output


def _compare_records(rec1: dict, rec2: dict, tolerance: float) -> list[dict]:
    """Compare two records and return list of differences."""
    diffs = []

    # Compare summary fields
    summary1 = rec1.get("summary", {})
    summary2 = rec2.get("summary", {})

    # Basic info comparison
    for field in ["shape", "stride", "device", "dtype", "numel"]:
        val1 = summary1.get(field)
        val2 = summary2.get(field)
        if val1 != val2:
            diffs.append({
                "type": "basic_info",
                "field": field,
                "value_dir1": val1,
                "value_dir2": val2,
            })

    # Numerical stats comparison
    for field in ["max", "min", "mean", "var", "nan_count", "inf_count", "raw_max", "raw_min"]:
        val1 = summary1.get(field)
        val2 = summary2.get(field)
        if val1 is None and val2 is None:
            continue
        if val1 is None or val2 is None:
            diffs.append({
                "type": "numerical_stat",
                "field": field,
                "value_dir1": val1,
                "value_dir2": val2,
            })
            continue

        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if abs(val1 - val2) > tolerance:
                diffs.append({
                    "type": "numerical_stat",
                    "field": field,
                    "value_dir1": val1,
                    "value_dir2": val2,
                    "abs_diff": abs(val1 - val2),
                })
        elif val1 != val2:
            diffs.append({
                "type": "numerical_stat",
                "field": field,
                "value_dir1": val1,
                "value_dir2": val2,
            })

    # Compare sampled values
    samples1 = rec1.get("samples", [])
    samples2 = rec2.get("samples", [])

    if len(samples1) != len(samples2):
        diffs.append({
            "type": "sample_count",
            "count_dir1": len(samples1),
            "count_dir2": len(samples2),
        })
    else:
        sample_diffs = []
        for i, (s1, s2) in enumerate(zip(samples1, samples2)):
            if isinstance(s1, (int, float)) and isinstance(s2, (int, float)):
                if abs(s1 - s2) > tolerance:
                    sample_diffs.append({
                        "index": i,
                        "value_dir1": s1,
                        "value_dir2": s2,
                        "abs_diff": abs(s1 - s2),
                    })
        if sample_diffs:
            diffs.append({
                "type": "sample_values",
                "diff_count": len(sample_diffs),
                "total_samples": len(samples1),
                "sample_diffs": sample_diffs[:20],  # Limit output size
            })

    return diffs


def lookup_stack(result_dir: str, stack_id: str, rank: int = 0):
    """Look up a full stack trace by ID."""
    stacks_file = os.path.join(result_dir, f"stacks_rank{rank}.json")
    if not os.path.exists(stacks_file):
        print(f"[TPD] Error: Stacks file not found: {stacks_file}", file=sys.stderr)
        return None

    with open(stacks_file) as f:
        stacks = json.load(f)

    if stack_id not in stacks:
        print(f"[TPD] Error: Stack ID not found: {stack_id}", file=sys.stderr)
        print(f"[TPD] Available stack IDs: {list(stacks.keys())[:20]}...", file=sys.stderr)
        return None

    frames = stacks[stack_id]
    return frames


def cmd_compare(args):
    """CLI command: compare two result directories."""
    compare_results(
        dir1=args.dir1,
        dir2=args.dir2,
        output_file=args.output,
        rank=args.rank,
        tolerance=args.tolerance,
    )


def cmd_stack(args):
    """CLI command: look up stack trace by ID."""
    frames = lookup_stack(args.result_dir, args.stack_id, args.rank)
    if frames:
        print(f"\nStack trace for ID: {args.stack_id}")
        print("=" * 80)
        for frame in frames:
            print(frame.rstrip())
        print("=" * 80)


def cmd_cmp_summary(args):
    """CLI command: show comparison summary."""
    from .compare_view import load_comparison, print_summary
    data = load_comparison(args.comparison)
    print_summary(data)


def cmd_cmp_list(args):
    """CLI command: list diff entries."""
    from .compare_view import load_comparison, list_entries
    data = load_comparison(args.comparison)
    list_entries(data, args.issue_type, args.sort, args.threshold)


def cmd_cmp_show(args):
    """CLI command: show specific diff entry(ies)."""
    from .compare_view import load_comparison, show_entry, show_entry_range
    data = load_comparison(args.comparison)
    if "-" in args.index:
        parts = args.index.split("-")
        show_entry_range(data, int(parts[0]), int(parts[1]))
    else:
        show_entry(data, int(args.index), args.window)


def cmd_cmp_count(args):
    """CLI command: count diff entries."""
    from .compare_view import load_comparison, count_entries
    data = load_comparison(args.comparison)
    count_entries(data, args.issue_type, args.threshold)


def cmd_cmp_first(args):
    """CLI command: find first occurrence of an issue type."""
    from .compare_view import load_comparison, find_first
    data = load_comparison(args.comparison)
    find_first(data, args.issue_type, args.window, args.threshold)


def cmd_report(args):
    """CLI command: generate summary report."""
    from .report import generate_report
    output = generate_report(args.comparison, args.output, args.threshold)
    print(f"\n[TPD] Report saved to: {output}")
    print(f"[TPD] You can view it with: cat {output}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="tpd",
        description="Torch Precision Debugger - Precision issue locator for PyTorch training",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two result directories",
    )
    compare_parser.add_argument("dir1", help="First result directory")
    compare_parser.add_argument("dir2", help="Second result directory")
    compare_parser.add_argument(
        "-o", "--output",
        default="tpd_comparison.json",
        help="Output file for comparison results (default: tpd_comparison.json)",
    )
    compare_parser.add_argument(
        "-r", "--rank",
        type=int,
        default=0,
        help="Rank to compare (default: 0)",
    )
    compare_parser.add_argument(
        "-t", "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for numerical comparison (default: 1e-6)",
    )
    compare_parser.set_defaults(func=cmd_compare)

    # Stack lookup command
    stack_parser = subparsers.add_parser(
        "stack",
        help="Look up stack trace by ID",
    )
    stack_parser.add_argument("result_dir", help="Result directory")
    stack_parser.add_argument("stack_id", help="Stack trace ID")
    stack_parser.add_argument(
        "-r", "--rank",
        type=int,
        default=0,
        help="Rank (default: 0)",
    )
    stack_parser.set_defaults(func=cmd_stack)

    # ================================================================
    # cmp commands - query / explore comparison results
    # ================================================================
    cmp_parser = subparsers.add_parser(
        "cmp",
        help="Query and explore comparison results",
    )
    cmp_subparsers = cmp_parser.add_subparsers(dest="cmp_command", help="CMP sub-commands")

    # cmp summary
    p_summary = cmp_subparsers.add_parser("summary", help="Show comparison summary")
    p_summary.add_argument("comparison", help="Path to comparison JSON file")
    p_summary.set_defaults(func=cmd_cmp_summary)

    # cmp list
    p_list = cmp_subparsers.add_parser("list", help="List all diff entries")
    p_list.add_argument("comparison", help="Path to comparison JSON file")
    p_list.add_argument(
        "-t", "--type",
        dest="issue_type",
        default="all",
        choices=["all", "naninf", "large-diff", "shape"],
        help="Filter by issue type (default: all)",
    )
    p_list.add_argument(
        "--sort",
        default="step",
        choices=["step", "index"],
        help="Sort order (default: step)",
    )
    p_list.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Threshold for large-diff detection (default: 1.0)",
    )
    p_list.set_defaults(func=cmd_cmp_list)

    # cmp show
    p_show = cmp_subparsers.add_parser("show", help="Show specific diff entry(ies)")
    p_show.add_argument("comparison", help="Path to comparison JSON file")
    p_show.add_argument(
        "index",
        help='Entry index (e.g. "5") or range (e.g. "5-10")',
    )
    p_show.add_argument(
        "-w", "--window",
        type=int,
        default=0,
        help="Context window around the entry (default: 0)",
    )
    p_show.set_defaults(func=cmd_cmp_show)

    # cmp count
    p_count = cmp_subparsers.add_parser("count", help="Count diff entries")
    p_count.add_argument("comparison", help="Path to comparison JSON file")
    p_count.add_argument(
        "-t", "--type",
        dest="issue_type",
        default="all",
        choices=["all", "naninf", "large-diff", "shape"],
        help="Filter by issue type (default: all)",
    )
    p_count.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Threshold for large-diff detection (default: 1.0)",
    )
    p_count.set_defaults(func=cmd_cmp_count)

    # cmp first
    p_first = cmp_subparsers.add_parser(
        "first",
        help="Find first occurrence of an issue type (by step)",
    )
    p_first.add_argument("comparison", help="Path to comparison JSON file")
    p_first.add_argument(
        "-t", "--type",
        dest="issue_type",
        default="naninf",
        choices=["naninf", "large-diff", "shape", "all"],
        help="Issue type to find (default: naninf)",
    )
    p_first.add_argument(
        "-w", "--window",
        type=int,
        default=3,
        help="Context window around the found entry (default: 3)",
    )
    p_first.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Threshold for large-diff detection (default: 1.0)",
    )
    p_first.set_defaults(func=cmd_cmp_first)

    # report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate comprehensive summary report from comparison results",
    )
    report_parser.add_argument("comparison", help="Path to comparison JSON file")
    report_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output report file path (default: <comparison>.report.txt)",
    )
    report_parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Threshold for large-diff detection (default: 1.0)",
    )
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # cmp sub-command dispatch
    if args.command == "cmp":
        if args.cmp_command is None:
            cmp_parser.print_help()
            sys.exit(1)
        args.func(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
