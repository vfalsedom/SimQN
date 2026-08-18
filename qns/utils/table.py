from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any, List


def format_table(data: Any, show_average: bool = True) -> str:
    """Format dictionary-shaped data as an aligned terminal table.

    ``data`` may be a column-oriented mapping such as
    ``{"round": [1, 2], "start_time": [2, 3]}``, a single row mapping,
    or a sequence of row mappings.
    """
    rows = _to_rows(data)
    if not rows:
        return ""

    columns: List[Any] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)

    headers = [str(column) for column in columns]
    values = [[_display(row.get(column, "")) for column in columns] for row in rows]
    widths = [max(len(headers[index]), *(len(row[index]) for row in values))
              for index in range(len(columns))]

    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    for source, displayed in zip(rows, values):
        cells = []
        for column, value, width in zip(columns, displayed, widths):
            original = source.get(column, "")
            if isinstance(original, Number) and not isinstance(original, bool):
                cells.append(value.rjust(width))
            else:
                cells.append(value.ljust(width))
        lines.append("  ".join(cells).rstrip())
    if show_average:
        averages = _numeric_averages(rows, columns)
        if averages:
            name_width = max(len(str(column)) for column, _ in averages)
            value_width = max(len(_format_number(value)) for _, value in averages)
            lines.extend(["", "Averages", "--------"])
            lines.extend(
                "{}  {}".format(
                    str(column).ljust(name_width),
                    _format_number(value).rjust(value_width),
                )
                for column, value in averages
            )
    return "\n".join(lines)


def _to_rows(data: Any) -> List[Mapping]:
    if isinstance(data, Mapping):
        if not data:
            return []
        columns = {}
        row_count = 1
        for key, value in data.items():
            if _is_column(value):
                column = list(value)
                columns[key] = column
                row_count = max(row_count, len(column))
            else:
                columns[key] = value
        for key, value in columns.items():
            if isinstance(value, list) and len(value) not in {0, row_count}:
                raise ValueError("table columns must have the same length")
        return [
            {
                key: (value[index] if isinstance(value, list) and value else value)
                for key, value in columns.items()
            }
            for index in range(row_count)
        ]

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        rows = list(data)
        if any(not isinstance(row, Mapping) for row in rows):
            raise TypeError("table rows must be mappings")
        return rows
    raise TypeError("table data must be a mapping or a sequence of mappings")


def _is_column(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping))


def _display(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ")


def _numeric_averages(rows: List[Mapping], columns: List[Any]):
    averages = []
    for column in columns:
        values = [row.get(column) for row in rows if row.get(column) is not None]
        if values and all(isinstance(value, Number) and not isinstance(value, bool)
                          for value in values):
            averages.append((column, sum(values) / len(values)))
    return averages


def _format_number(value: Number) -> str:
    if isinstance(value, complex):
        return str(value)
    return "{:.6f}".format(float(value)).rstrip("0").rstrip(".")
