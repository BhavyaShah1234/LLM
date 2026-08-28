"""Generic plumbing behind experiments/*/compare.py.

Deliberately thin: loading a list of explicit run_result.json paths and
printing a table is the only thing every experiment needs in common. Which
runs to compare and what the comparison *means* is the research-question
framing that belongs in each experiment's own compare.py, not here.
"""

import json
from typing import Any, Dict, List, Optional, Sequence


def load_run_results(paths: Sequence[str]) -> List[Dict[str, Any]]:
    results = []
    for path in paths:
        with open(path) as f:
            results.append(json.load(f))
    return results


def _get_nested(record: Dict[str, Any], key: str):
    """metric_keys may reference top-level fields (e.g. 'architecture') or
    nested metrics (e.g. 'metrics.perplexity')."""
    parts = key.split(".")
    value = record
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def print_comparison_table(
    results: List[Dict[str, Any]],
    group_by: str,
    metric_keys: Sequence[str],
    title: Optional[str] = None,
) -> None:
    if title:
        print("=" * 80)
        print(title)
        print("=" * 80)

    header_cells = [group_by] + list(metric_keys)
    col_widths = [max(len(h), 12) for h in header_cells]
    for record in results:
        row = [str(_get_nested(record, group_by))] + [str(_get_nested(record, k)) for k in metric_keys]
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells):
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, col_widths))

    print(fmt_row(header_cells))
    print("-+-".join("-" * w for w in col_widths))
    for record in results:
        row = [_get_nested(record, group_by)] + [_get_nested(record, k) for k in metric_keys]
        print(fmt_row(row))
    print()
