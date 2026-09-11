"""
Report generation for TPD comparison results.
Generates a comprehensive summary report with first occurrences,
statistics, and actionable insights for precision debugging.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any

from .compare_view import _matches_issue


# ── Report Generation ──────────────────────────────────────────────


def generate_report(
    comparison_file: str,
    output_file: str | None = None,
    threshold: float = 1.0,
) -> str:
    """Generate a comprehensive summary report from a comparison JSON file.

    Args:
        comparison_file: Path to the comparison JSON file.
        output_file: Output report file path (default: <comparison_file>.report.txt).
        threshold: Threshold for large-diff detection.

    Returns:
        The path to the generated report file.
    """
    if not os.path.exists(comparison_file):
        print(f"[TPD] Error: Comparison file not found: {comparison_file}", file=sys.stderr)
        sys.exit(1)

    with open(comparison_file) as f:
        data = json.load(f)

    if output_file is None:
        base = os.path.splitext(comparison_file)[0]
        output_file = f"{base}.report.txt"

    report_lines: list[str] = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("  TPD Precision Debug Report")
    report_lines.append("=" * 80)
    report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"  Source:    {comparison_file}")
    report_lines.append("")

    # Section 1: Overview
    _add_overview_section(report_lines, data)

    # Section 2: Issue Statistics
    _add_statistics_section(report_lines, data, threshold)

    # Section 3: First Occurrences
    _add_first_occurrences_section(report_lines, data, threshold)

    # Section 4: Critical Issues (NaN/Inf)
    _add_critical_issues_section(report_lines, data)

    # Section 5: Top Large Diffs
    _add_top_diffs_section(report_lines, data, threshold)

    # Section 6: Shape Mismatches
    _add_shape_mismatches_section(report_lines, data)

    # Section 7: Hook Type Distribution
    _add_hook_distribution_section(report_lines, data)

    # Section 8: Recommendations
    _add_recommendations_section(report_lines, data, threshold)

    # Footer
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("  End of Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Next Steps:")
    report_lines.append("  1. Review the 'First Occurrences' section to identify where issues start")
    report_lines.append("  2. Use 'tpd cmp first' to view context around first occurrences")
    report_lines.append("  3. Use 'tpd cmp show' to inspect specific entries in detail")
    report_lines.append("  4. Use 'tpd stack' to query stack traces and locate source code")
    report_lines.append("")
    report_lines.append("Useful Commands:")
    report_lines.append(f"  tpd cmp summary {comparison_file}")
    report_lines.append(f"  tpd cmp first {comparison_file} --type naninf --window 3")
    report_lines.append(f"  tpd cmp first {comparison_file} --type large-diff --threshold {threshold} --window 3")
    report_lines.append(f"  tpd cmp list {comparison_file} --type naninf --sort step")
    report_lines.append(f"  tpd cmp show {comparison_file} <index> --window 2")
    report_lines.append(f"  tpd stack <result_dir> <stack_id>")
    report_lines.append("")

    # Write report
    report_text = "\n".join(report_lines)
    with open(output_file, "w") as f:
        f.write(report_text)

    print(f"[TPD] Report generated: {output_file}", file=sys.stderr)
    return output_file


# ── Section Builders ───────────────────────────────────────────────


def _add_overview_section(lines: list[str], data: dict):
    """Add overview section."""
    s = data.get("summary", {})
    lines.append("─" * 80)
    lines.append("  1. Overview")
    lines.append("─" * 80)
    lines.append(f"  Dir1:                     {s.get('dir1', 'N/A')}")
    lines.append(f"  Dir2:                     {s.get('dir2', 'N/A')}")
    lines.append(f"  Rank:                     {s.get('rank', 'N/A')}")
    lines.append(f"  Tolerance:                {s.get('tolerance', 'N/A')}")
    lines.append(f"  Total entries dir1:       {s.get('total_entries_dir1', 0)}")
    lines.append(f"  Total entries dir2:       {s.get('total_entries_dir2', 0)}")
    lines.append(f"  Only in dir1:             {s.get('only_in_dir1', 0)}")
    lines.append(f"  Only in dir2:             {s.get('only_in_dir2', 0)}")
    lines.append(f"  Common entries:           {s.get('common_entries', 0)}")
    lines.append(f"  Entries with differences: {s.get('entries_with_differences', 0)}")
    lines.append("")


def _add_statistics_section(lines: list[str], data: dict, threshold: float):
    """Add issue statistics section."""
    diffs = data.get("differences", [])
    total = len(diffs)

    naninf_cnt = sum(1 for e in diffs if _matches_issue(e, "naninf"))
    large_cnt = sum(1 for e in diffs if _matches_issue(e, "large-diff", threshold))
    shape_cnt = sum(1 for e in diffs if _matches_issue(e, "shape"))

    lines.append("─" * 80)
    lines.append("  2. Issue Statistics")
    lines.append("─" * 80)
    lines.append(f"  Total entries with differences: {total}")
    lines.append(f"  ├── NaN/Inf issues:             {naninf_cnt}  {'⚠️  CRITICAL' if naninf_cnt > 0 else '✓'}")
    lines.append(f"  ├── Large diff issues:          {large_cnt}  (threshold: {threshold})")
    lines.append(f"  └── Shape mismatches:           {shape_cnt}  {'⚠️  ERROR' if shape_cnt > 0 else '✓'}")
    lines.append("")

    if total > 0:
        severity = "HIGH" if naninf_cnt > 0 else ("MEDIUM" if large_cnt > 10 or shape_cnt > 0 else "LOW")
        lines.append(f"  Overall Severity: {severity}")
        lines.append("")


def _add_first_occurrences_section(lines: list[str], data: dict, threshold: float):
    """Add first occurrences section."""
    diffs = data.get("differences", [])
    if not diffs:
        return

    lines.append("─" * 80)
    lines.append("  3. First Occurrences (by step)")
    lines.append("─" * 80)

    # First NaN/Inf
    first_naninf = _find_first(diffs, "naninf", threshold)
    if first_naninf:
        idx, entry = first_naninf
        key = entry.get("key", {})
        lines.append(f"  First NaN/Inf Issue:")
        lines.append(f"    Index:     {idx}")
        lines.append(f"    Step:      {key.get('step', 'N/A')}")
        lines.append(f"    Hook Type: {key.get('hook_type', 'N/A')}")
        lines.append(f"    Module:    {key.get('module_name', 'N/A')}")
        lines.append(f"    Tensor:    {key.get('tensor_path', 'N/A')}")
        lines.append(f"    Stack ID:  {entry.get('stack_id_1', 'N/A')}")
        lines.append("")
    else:
        lines.append(f"  First NaN/Inf Issue: None detected ✓")
        lines.append("")

    # First Large Diff
    first_large = _find_first(diffs, "large-diff", threshold)
    if first_large:
        idx, entry = first_large
        key = entry.get("key", {})
        max_diff = _get_max_abs_diff(entry)
        lines.append(f"  First Large Diff Issue:")
        lines.append(f"    Index:     {idx}")
        lines.append(f"    Step:      {key.get('step', 'N/A')}")
        lines.append(f"    Hook Type: {key.get('hook_type', 'N/A')}")
        lines.append(f"    Module:    {key.get('module_name', 'N/A')}")
        lines.append(f"    Tensor:    {key.get('tensor_path', 'N/A')}")
        lines.append(f"    Max Diff:  {max_diff:.6e}")
        lines.append(f"    Stack ID:  {entry.get('stack_id_1', 'N/A')}")
        lines.append("")
    else:
        lines.append(f"  First Large Diff Issue: None detected ✓")
        lines.append("")

    # First Shape Mismatch
    first_shape = _find_first(diffs, "shape", threshold)
    if first_shape:
        idx, entry = first_shape
        key = entry.get("key", {})
        lines.append(f"  First Shape Mismatch:")
        lines.append(f"    Index:     {idx}")
        lines.append(f"    Step:      {key.get('step', 'N/A')}")
        lines.append(f"    Hook Type: {key.get('hook_type', 'N/A')}")
        lines.append(f"    Module:    {key.get('module_name', 'N/A')}")
        lines.append(f"    Tensor:    {key.get('tensor_path', 'N/A')}")
        lines.append(f"    Stack ID:  {entry.get('stack_id_1', 'N/A')}")
        lines.append("")
    else:
        lines.append(f"  First Shape Mismatch: None detected ✓")
        lines.append("")


def _add_critical_issues_section(lines: list[str], data: dict):
    """Add critical issues (NaN/Inf) section."""
    diffs = data.get("differences", [])
    naninf_entries = [(i, e) for i, e in enumerate(diffs) if _matches_issue(e, "naninf")]

    if not naninf_entries:
        return

    lines.append("─" * 80)
    lines.append("  4. Critical Issues: NaN/Inf")
    lines.append("─" * 80)
    lines.append(f"  Total NaN/Inf issues: {len(naninf_entries)}")
    lines.append("")

    for idx, entry in naninf_entries[:10]:  # Show first 10
        key = entry.get("key", {})
        lines.append(f"  [{idx}] Step {key.get('step', '?')} - {key.get('hook_type', '?')}")
        lines.append(f"       Module: {key.get('module_name', '?')}")
        lines.append(f"       Tensor: {key.get('tensor_path', '?')}")

        # Extract NaN/Inf details
        for d in entry.get("differences", []):
            field = d.get("field", "")
            if field in ("nan_count", "inf_count"):
                v1 = d.get("value_dir1", 0)
                v2 = d.get("value_dir2", 0)
                if v1 != v2:
                    lines.append(f"       {field}: Dir1={v1} → Dir2={v2}")
        lines.append("")

    if len(naninf_entries) > 10:
        lines.append(f"  ... and {len(naninf_entries) - 10} more NaN/Inf issues")
        lines.append("")


def _add_top_diffs_section(lines: list[str], data: dict, threshold: float):
    """Add top large diffs section."""
    diffs = data.get("differences", [])
    large_entries = [(i, e) for i, e in enumerate(diffs) if _matches_issue(e, "large-diff", threshold)]

    if not large_entries:
        return

    # Sort by max abs diff
    large_entries_with_max = [(i, e, _get_max_abs_diff(e)) for i, e in large_entries]
    large_entries_with_max.sort(key=lambda x: -x[2])

    lines.append("─" * 80)
    lines.append("  5. Top 10 Largest Differences")
    lines.append("─" * 80)
    lines.append(f"  Total large diff issues: {len(large_entries)} (threshold: {threshold})")
    lines.append("")

    for rank, (idx, entry, max_diff) in enumerate(large_entries_with_max[:10], 1):
        key = entry.get("key", {})
        lines.append(f"  #{rank} [{idx}] Step {key.get('step', '?')} - Max Diff: {max_diff:.6e}")
        lines.append(f"       Hook:   {key.get('hook_type', '?')}")
        lines.append(f"       Module: {key.get('module_name', '?')}")
        lines.append(f"       Tensor: {key.get('tensor_path', '?')}")
        lines.append("")


def _add_shape_mismatches_section(lines: list[str], data: dict):
    """Add shape mismatches section."""
    diffs = data.get("differences", [])
    shape_entries = [(i, e) for i, e in enumerate(diffs) if _matches_issue(e, "shape")]

    if not shape_entries:
        return

    lines.append("─" * 80)
    lines.append("  6. Shape Mismatches")
    lines.append("─" * 80)
    lines.append(f"  Total shape mismatches: {len(shape_entries)}")
    lines.append("")

    for idx, entry in shape_entries[:10]:
        key = entry.get("key", {})
        lines.append(f"  [{idx}] Step {key.get('step', '?')} - {key.get('hook_type', '?')}")
        lines.append(f"       Module: {key.get('module_name', '?')}")
        lines.append(f"       Tensor: {key.get('tensor_path', '?')}")

        # Extract shape details
        for d in entry.get("differences", []):
            if d.get("type") == "basic_info":
                field = d.get("field", "")
                v1 = d.get("value_dir1")
                v2 = d.get("value_dir2")
                lines.append(f"       {field}: Dir1={v1} vs Dir2={v2}")
        lines.append("")

    if len(shape_entries) > 10:
        lines.append(f"  ... and {len(shape_entries) - 10} more shape mismatches")
        lines.append("")


def _add_hook_distribution_section(lines: list[str], data: dict):
    """Add hook type distribution section."""
    diffs = data.get("differences", [])
    if not diffs:
        return

    hook_types: dict[str, int] = {}
    for e in diffs:
        ht = e.get("key", {}).get("hook_type", "unknown")
        hook_types[ht] = hook_types.get(ht, 0) + 1

    lines.append("─" * 80)
    lines.append("  7. Issue Distribution by Hook Type")
    lines.append("─" * 80)

    for ht, count in sorted(hook_types.items(), key=lambda x: -x[1]):
        pct = count / len(diffs) * 100
        bar = "█" * int(pct / 2)
        lines.append(f"  {ht:<25} {count:>4} ({pct:5.1f}%) {bar}")
    lines.append("")

    # Analysis
    max_ht = max(hook_types.items(), key=lambda x: x[1])
    lines.append(f"  Most affected: {max_ht[0]} ({max_ht[1]} entries)")

    if "backward_grad_output" in hook_types and hook_types["backward_grad_output"] > len(diffs) * 0.5:
        lines.append("  → High backward_grad_output issues suggest loss/gradient scaling problems")
    if "forward_output" in hook_types and hook_types["forward_output"] > len(diffs) * 0.5:
        lines.append("  → High forward_output issues suggest layer computation problems")
    lines.append("")


def _add_recommendations_section(lines: list[str], data: dict, threshold: float):
    """Add recommendations section."""
    diffs = data.get("differences", [])
    naninf_cnt = sum(1 for e in diffs if _matches_issue(e, "naninf"))
    large_cnt = sum(1 for e in diffs if _matches_issue(e, "large-diff", threshold))
    shape_cnt = sum(1 for e in diffs if _matches_issue(e, "shape"))

    lines.append("─" * 80)
    lines.append("  8. Recommendations")
    lines.append("─" * 80)

    if naninf_cnt > 0:
        lines.append("  ⚠️  NaN/Inf Detected:")
        lines.append("     1. Check for division by zero in custom operations")
        lines.append("     2. Verify log/exp operations have proper input clamping")
        lines.append("     3. Review gradient clipping/scaling logic")
        lines.append("     4. If using FP16, check loss scaling factor")
        lines.append("     5. Inspect weight initialization for extreme values")
        lines.append("")

    if large_cnt > 0:
        lines.append("  ⚠️  Large Numerical Differences:")
        lines.append("     1. Verify random seeds are identical in both runs")
        lines.append("     2. Check data loading order and shuffling")
        lines.append("     3. Compare model initialization parameters")
        lines.append("     4. Review optimizer state and learning rate")
        lines.append("     5. If comparing FP32 vs FP16, expect larger diffs (normal)")
        if large_cnt > 20:
            lines.append("     ⚠️  Many large diffs suggest systematic difference (seed/data/init)")
        lines.append("")

    if shape_cnt > 0:
        lines.append("  ⚠️  Shape Mismatches:")
        lines.append("     1. Verify model architecture is identical")
        lines.append("     2. Check batch size and sequence length configurations")
        lines.append("     3. Review padding/truncation logic")
        lines.append("     4. If distributed, verify tensor parallelism settings")
        lines.append("")

    if naninf_cnt == 0 and large_cnt == 0 and shape_cnt == 0:
        lines.append("  ✓ No critical issues detected.")
        lines.append("  → Differences are within acceptable tolerance.")
        lines.append("")

    lines.append("  General Tips:")
    lines.append("  • Start debugging from the first occurrence (earliest step)")
    lines.append("  • Use stack traces to locate the exact source code line")
    lines.append("  • Compare intermediate layers to isolate the problematic module")
    lines.append("  • If using mixed precision, check autocast and grad_scaler usage")
    lines.append("")


# ── Helper Functions ───────────────────────────────────────────────


def _find_first(diffs: list[dict], issue_type: str, threshold: float):
    """Find the first occurrence of an issue type."""
    matching = [(i, e) for i, e in enumerate(diffs) if _matches_issue(e, issue_type, threshold)]
    if not matching:
        return None
    matching.sort(key=lambda x: (x[1].get("key", {}).get("step", float("inf")), x[0]))
    return matching[0]


def _get_max_abs_diff(entry: dict) -> float:
    """Get the maximum absolute difference from an entry."""
    max_diff = 0.0
    for d in entry.get("differences", []):
        abs_diff = d.get("abs_diff", 0)
        if abs_diff and abs_diff > max_diff:
            max_diff = abs_diff
        # Also check sample diffs
        if d.get("type") == "sample_values":
            for sd in d.get("sample_diffs", []):
                sd_diff = sd.get("abs_diff", 0)
                if sd_diff and sd_diff > max_diff:
                    max_diff = sd_diff
    return max_diff
