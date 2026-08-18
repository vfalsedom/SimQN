import pytest

from qns.utils.table import format_table


def test_format_column_oriented_dictionary() -> None:
    table = format_table({"round": [1, 2], "start_time": [2, 3]})

    assert table == (
        "round  start_time\n"
        "-----  ----------\n"
        "    1           2\n"
        "    2           3\n"
        "\n"
        "Averages\n"
        "--------\n"
        "round       1.5\n"
        "start_time  2.5"
    )


def test_format_row_oriented_dictionaries() -> None:
    table = format_table([
        {"round": 1, "success": True},
        {"round": 2, "success": False},
    ])

    assert "    1  True" in table
    assert "    2  False" in table


def test_format_table_rejects_unequal_column_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        format_table({"round": [1, 2], "start_time": [3]})


def test_format_table_averages_only_numeric_columns() -> None:
    table = format_table({
        "round": [1, 2],
        "label": ["first", "second"],
        "success": [True, False],
    })

    average_section = table.split("Averages\n--------\n", 1)[1]
    assert "round" in average_section
    assert "label" not in average_section
    assert "success" not in average_section


def test_format_table_can_hide_averages() -> None:
    table = format_table({"round": [1, 2]}, show_average=False)

    assert "Averages" not in table
