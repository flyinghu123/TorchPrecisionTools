"""
Comparison viewing and querying tools for TPD.
Provides commands for exploring comparison results efficiently,
helping locate precision issues like NaN/Inf, large diffs, shape mismatches.

Usage (via CLI):
  tpd cmp summary <comparison.json>         - Show overview
  tpd cmp list <comparison.json> [options]   - List all entries
  tpd cmp show <file> <index|range> [options] - Show specific entry
  tpd cmp count <comparison.json> [options]  - Count entries
  tpd cmp first <comparison.json> [options]  - Find first occurrence
"""

import json
import os
import sys
from typing import Any

# ── Issue type detection ──────────────────────────────────────────────

_ISSUE_FIELD_NAMES = {
    "naninf": ("nan_count", "inf_count"),
    "shape": ("shape", "stride", "numel", "dtype", "device"),
}


def _matches_issue(entry: dict, issue_type: str, threshold: float = 1.0) -> bool:
    """Check whether a diff entry matches a specific issue type.

    Args:
        entry: A single diff entry from the comparison ``differences`` list.
        issue_type: ``"all"``, ``"naninf"``, ``"large-diff"``, or ``"shape"``.
        threshold: Absolute-diff threshold for ``"large-diff"`` detection.
    """
    if issue_type == "all":
        return True

    diffs = entry.get("differences", [])
    for d in diffs:
        field = d.get("field", "")

        if issue_type == "naninf" and field in _ISSUE_FIELD_NAMES["naninf"]:
            v1 = d.get("value_dir1")
            v2 = d.get("value_dir2")
            if v1 != v2:
                if (v1 is not None and v1 > 0) or (v2 is not None and v2 > 0):
                    return True

        elif issue_type == "large-diff":
            abs_diff = d.get("abs_diff")
            if abs_diff is not None and abs_diff > threshold:
                return True
            # Also inspect sample-level diffs
            if d.get("type") == "sample_values":
                for sd in d.get("sample_diffs", []):
                    if sd.get("abs_diff", 0) > threshold:
                        return True

        elif issue_type == "shape" and d.get("type") == "basic_info":
            if field in _ISSUE_FIELD_NAMES["shape"]:
                return True

    return False


# ── Loader ────────────────────────────────────────────────────────────


def load_comparison(filepath: str) -> dict:
    """Load a comparison JSON file from *filepath*."""
    if not os.path.exists(filepath):
        print(f"[TPD] Error: Comparison file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    with open(filepath) as f:
        return json.load(f)


# ── Summary ────────────────────────────────────────────────────────────


def print_summary(data: dict):
    """Print an overview of the comparison results."""
    s = data.get("summary", {})
    diffs = data.get("differences", [])

    print("=" * 65)
    print("  TPD Comparison Summary")
    print("=" * 65)
    print(f"  Dir1:                     {s.get('dir1', 'N/A')}")
    print(f"  Dir2:                     {s.get('dir2', 'N/A')}")
    print(f"  Rank:                     {s.get('rank', 'N/A')}")
    print(f"  Tolerance:                {s.get('tolerance', 'N/A')}")
    print(f"  Total entries dir1:       {s.get('total_entries_dir1', 0)}")
    print(f"  Total entries dir2:       {s.get('total_entries_dir2', 0)}")
    print(f"  Only in dir1:             {s.get('only_in_dir1', 0)}")
    print(f"  Only in dir2:             {s.get('only_in_dir2', 0)}")
    print(f"  Common entries:           {s.get('common_entries', 0)}")
    print(f"  Entries with differences: {s.get('entries_with_differences', 0)}")

    if diffs:
        naninf_cnt = sum(1 for e in diffs if _matches_issue(e, "naninf"))
        large_cnt = sum(1 for e in diffs if _matches_issue(e, "large-diff"))
        shape_cnt = sum(1 for e in diffs if _matches_issue(e, "shape"))
        print("-" * 65)
        print(f"  NaN/Inf issues:           {naninf_cnt}")
        print(f"  Large diff issues:        {large_cnt}")
        print(f"  Shape mismatch issues:    {shape_cnt}")
    print("=" * 65)


# ── List ──────────────────────────────────────────────────────────────


def list_entries(
    data: dict,
    issue_type: str = "all",
    sort_by: str = "step",
    threshold: float = 1.0,
):
    """List all diff entries with index numbers, filtered by *issue_type*."""
    all_diffs = data.get("differences", [])

    if issue_type != "all":
        filtered = [(i, e) for i, e in enumerate(all_diffs) if _matches_issue(e, issue_type, threshold)]
    else:
        filtered = [(i, e) for i, e in enumerate(all_diffs)]

    if not filtered:
        print(f"[TPD] No entries found matching type '{issue_type}'.")
        return

    if sort_by == "step":
        filtered.sort(key=lambda x: (x[1].get("key", {}).get("step", 0), x[0]))
    # else keep original index order

    print(f"\n  {'Idx':<6} {'Step':<6} {'Type':<22} {'Issues':<7}  Module / Tensor")
    print(f"  " + "-" * 100)

    for idx, entry in filtered:
        key = entry.get("key", {})
        issues = len(entry.get("differences", []))
        module = key.get("module_name", "?")
        tensor = key.get("tensor_path", "?")
        if len(module) > 42:
            module = "..." + module[-39:]
        label = key.get("hook_type", "?")
        print(f"  {idx:<6} {key.get('step', '?'):<6} {label:<22} {issues:<7}  {module} [{tensor}]")

    print(f"\n  Total: {len(filtered)} entries  (filter: {issue_type})")


# ── Show ──────────────────────────────────────────────────────────────


def show_entry(data: dict, index: int, window: int = 0):
    """Show the diff entry at *index* with optional surrounding *window*."""
    diffs = data.get("differences", [])
    total = len(diffs)

    if total == 0:
        print("[TPD] No diff entries in comparison.")
        return
    if index < 0 or index >= total:
        print(f"[TPD] Error: Index {index} out of range [0, {total - 1}].", file=sys.stderr)
        return

    start = max(0, index - window)
    end = min(total, index + window + 1)

    for i in range(start, end):
        entry = diffs[i]
        key = entry.get("key", {})
        is_target = i == index

        if i > start:
            print()

        _render_entry(i, entry, is_target, window > 0)


def _render_entry(entry_index: int, entry: dict, is_target: bool, has_window: bool):
    """Render a single diff entry to stdout."""
    key = entry.get("key", {})

    if has_window:
        marker = " >>> TARGET <<<" if is_target else " [context]"
        print(f"{'=' * 65}")
        print(f"  Diff Entry [{entry_index}]{marker}")
    else:
        print(f"{'=' * 65}")
        print(f"  Diff Entry [{entry_index}]")

    print(f"{'=' * 65}")
    print(f"  Step:        {key.get('step', 'N/A')}")
    print(f"  Hook Type:   {key.get('hook_type', 'N/A')}")
    print(f"  Module:      {key.get('module_name', 'N/A')}")
    print(f"  Tensor Path: {key.get('tensor_path', 'N/A')}")
    print(f"  Stack ID 1:  {entry.get('stack_id_1', 'N/A')}")
    print(f"  Stack ID 2:  {entry.get('stack_id_2', 'N/A')}")
    print(f"  Differences: {len(entry.get('differences', []))}")

    for j, diff in enumerate(entry.get("differences", [])):
        if j > 0:
            print()
        _print_diff(j, diff)


def _print_diff(index: int, diff: dict):
    """Print a single diff entry detail."""
    dtype = diff.get("type", "unknown")
    field = diff.get("field", "")
    prefix = f"  Diff #{index}: [{dtype}]"

    if dtype == "basic_info":
        print(f"  {prefix} {field}")
        print(f"    Dir1: {diff.get('value_dir1', 'N/A')}")
        print(f"    Dir2: {diff.get('value_dir2', 'N/A')}")

    elif dtype == "numerical_stat":
        v1 = diff.get("value_dir1")
        v2 = diff.get("value_dir2")
        abs_diff = diff.get("abs_diff")

        if abs_diff is not None:
            print(f"  {prefix} {field}  (abs_diff: {abs_diff:.6e})")
        else:
            print(f"  {prefix} {field}")
        print(f"    Dir1: {_fmt_val(v1)}")
        print(f"    Dir2: {_fmt_val(v2)}")

    elif dtype == "sample_count":
        print(f"  {prefix}")
        print(f"    Dir1: {diff.get('count_dir1', 'N/A')} samples")
        print(f"    Dir2: {diff.get('count_dir2', 'N/A')} samples")

    elif dtype == "sample_values":
        sdiffs = diff.get("sample_diffs", [])
        print(f"  {prefix}  {diff.get('diff_count', 0)}/{diff.get('total_samples', 0)} samples differ")
        for sd in sdiffs[:5]:
            a = sd.get("abs_diff", 0)
            print(f"    [{sd.get('index')}]  Dir1: {_fmt_val(sd.get('value_dir1'))}"
                  f"  Dir2: {_fmt_val(sd.get('value_dir2'))}  diff: {a:.6e}")
        if len(sdiffs) > 5:
            print(f"    ... and {len(sdiffs) - 5} more")


def _fmt_val(v) -> str:
    """Pretty-format a scalar value."""
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def show_entry_range(data: dict, start: int, end: int):
    """Show a range of diff entries (inclusive)."""
    diffs = data.get("differences", [])
    total = len(diffs)
    start = max(0, start)
    end = min(end, total - 1)

    if start > end:
        print(f"[TPD] Error: start index > end index.", file=sys.stderr)
        return

    for i in range(start, end + 1):
        _render_entry(i, diffs[i], is_target=False, has_window=False)
        if i < end:
            print()


# ── Count ─────────────────────────────────────────────────────────────


def count_entries(data: dict, issue_type: str = "all", threshold: float = 1.0):
    """Count diff entries, optionally filtered by *issue_type*."""
    diffs = data.get("differences", [])
    total = len(diffs)

    if issue_type == "all":
        naninf_cnt = sum(1 for e in diffs if _matches_issue(e, "naninf"))
        large_cnt = sum(1 for e in diffs if _matches_issue(e, "large-diff", threshold))
        shape_cnt = sum(1 for e in diffs if _matches_issue(e, "shape"))

        hook_types: dict[str, int] = {}
        for e in diffs:
            ht = e.get("key", {}).get("hook_type", "unknown")
            hook_types[ht] = hook_types.get(ht, 0) + 1

        print("=" * 50)
        print("  Diff Count Summary")
        print("=" * 50)
        print(f"  Total entries with diffs: {total}")
        print(f"  |-- NaN/Inf issues:       {naninf_cnt}")
        print(f"  |-- Large diff issues:    {large_cnt}  (>= {threshold})")
        print(f"  +-- Shape mismatches:     {shape_cnt}")
        print()
        print(f"  By hook type:")
        for ht, c in sorted(hook_types.items(), key=lambda x: -x[1]):
            print(f"    {ht}: {c}")
    else:
        filtered = [e for e in diffs if _matches_issue(e, issue_type, threshold)]
        print(f"  Entries matching '{issue_type}': {len(filtered)} / {total}")


# ── First ─────────────────────────────────────────────────────────────


def find_first(
    data: dict,
    issue_type: str = "naninf",
    window: int = 3,
    threshold: float = 1.0,
):
    """Find the first occurrence (by step) of a specific issue type."""
    diffs = data.get("differences", [])

    matching = [(i, e) for i, e in enumerate(diffs) if _matches_issue(e, issue_type, threshold)]
    if not matching:
        print(f"[TPD] No entries found matching type '{issue_type}'.")
        return

    # Earliest step first; ties broken by original index
    matching.sort(key=lambda x: (x[1].get("key", {}).get("step", float("inf")), x[0]))

    first_idx, first_entry = matching[0]
    key = first_entry.get("key", {})

    labels = {
        "all": "any issue",
        "naninf": "NaN/Inf issue",
        "large-diff": "large diff",
        "shape": "shape mismatch",
    }

    print(f"[TPD] First {labels.get(issue_type, issue_type)}:")
    print(f"  Index:     {first_idx}")
    print(f"  Step:      {key.get('step', 'N/A')}")
    print(f"  Hook Type: {key.get('hook_type', 'N/A')}")
    print(f"  Module:    {key.get('module_name', 'N/A')}")
    print(f"  Tensor:    {key.get('tensor_path', 'N/A')}")

    # Show which specific individual diffs triggered the match
    triggered = [
        d for d in first_entry.get("differences", [])
        if _matches_issue({"differences": [d]}, issue_type, threshold)
    ]
    if triggered:
        print(f"  Triggering diff(s):")
        for d in triggered:
            field = d.get("field", "")
            ad = d.get("abs_diff")
            if ad is not None:
                print(f"    - [{d.get('type')}] {field}  (abs_diff: {ad:.6e})")
            else:
                print(f"    - [{d.get('type')}] {field}")

    if window > 0:
        print(f"\n[TPD] Context window [-{window}, +{window}] around entry [{first_idx}]:")
    show_entry(data, first_idx, window=window)