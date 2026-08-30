"""Generic plumbing behind experiments/*/compare.py.

Deliberately thin: loading a list of explicit run_result.json paths and
printing a table is the only thing every experiment needs in common. Which
runs to compare and what the comparison *means* is the research-question
framing that belongs in each experiment's own compare.py, not here.
"""

import json
from typing import Any, Dict, List, Optional, Sequence


def load_run_results(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Load a list of run_result.json files into memory.

    Args:
        paths (Sequence[str]): Filesystem paths to run_result.json files.

    Returns:
        List[Dict[str, Any]]: The parsed JSON contents, one dict per path, in
            the same order as `paths`.
    """
    results = []
    for path in paths:
        with open(path) as f:
            results.append(json.load(f))
    return results


def _get_nested(record: Dict[str, Any], key: str):
    """Look up a possibly dotted key in a run-result record.

    metric_keys may reference top-level fields (e.g. 'architecture') or
    nested metrics (e.g. 'metrics.perplexity').

    Args:
        record (Dict[str, Any]): A single parsed run_result.json record.
        key (str): Field name, optionally dotted to reach a nested dict
            (e.g. "metrics.perplexity").

    Returns:
        The value at that key/path, or None if any segment of the path is
        missing or not a dict.
    """
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
    """Print an ASCII table comparing run results side by side.

    Args:
        results (List[Dict[str, Any]]): Parsed run_result.json records, one
            row per record.
        group_by (str): Field (dotted path allowed) used as the first column,
            identifying each row (e.g. "architecture" or "variant").
        metric_keys (Sequence[str]): Additional fields (dotted paths allowed)
            to print as columns, in order.
        title (Optional[str]): If given, printed as a banner above the table.
    """
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
        """Render one row as fixed-width, pipe-separated cells.

        Args:
            cells: Row values to format, aligned with `col_widths`.

        Returns:
            str: The row rendered as `"cell1 | cell2 | ..."`, each cell
            left-justified to its column's width.
        """
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, col_widths))

    print(fmt_row(header_cells))
    print("-+-".join("-" * w for w in col_widths))
    for record in results:
        row = [_get_nested(record, group_by)] + [_get_nested(record, k) for k in metric_keys]
        print(fmt_row(row))
    print()
