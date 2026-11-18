"""The step table beside info.json's features: what the loader was measured on, and where the
measurement is kept.

Not a test module. Shared by the test that holds the validator's rule to the measurement
(test_step_features.py) and by the script that takes it (scripts/measure_loader_casts.py), so
that both speak of the same features and the same columns by the same names.

The rules in question (validator/validate_dataset.py ``_feature_column`` · ``_cast_stops`` ·
``_null_lists_misread``, C7): LeRobot's loader types the step table by info.json's features --
not by a file -- and casts every step file to that, batch by batch. Which declarations it cannot
make a table of, which stored columns it cannot turn into what is declared, and what it hands out
after a null list, is not reasoned out here a case at a time. It is measured: every form below,
every stored column below beside every declaration below, and every placing of nulls below,
through the loader's own code.
"""

import decimal
from pathlib import Path

import numpy as np
import pyarrow as pa

MEASURED = Path(__file__).with_name("feature_casts_measured.json")

ABSENT = object()


def feature(dtype=ABSENT, shape=ABSENT, **more) -> dict:
    """A feature as info.json holds one, a key left out where its value is ABSENT."""
    out = {"dtype": dtype, "shape": shape, "names": None, **more}
    return {key: value for key, value in out.items() if value is not ABSENT}


# ── The forms a feature can take in info.json, by a name the measurement file can list ──
# Each is measured as the feature of a column NO step file has: the loader then makes a column
# of nulls of the declared type, so nothing but the declaration itself is in the way.
_DTYPE_NAMES = (
    # the names a release uses, and pyarrow's other names for a type that takes no argument
    "float64", "float32", "float16", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "bool", "bool_", "string", "utf8", "large_string", "large_utf8", "binary", "large_binary", "null",
    "date32", "date64",
    # datasets' own two names
    "double", "float",
    # names of other libraries, and near misses
    "int", "uint", "long", "f8", "<f8", "<i8", "number", "str", "boolean", "object", "halffloat",
    "Float64", "FLOAT64", "float64 ", " float64", "",
    # names datasets or pyarrow know as something else
    "json", "array", "list", "list_", "struct", "decimal128", "timestamp", "cpu_count",
    # the names with an argument, which datasets reads by pattern
    "timestamp[s]", "time32[s]", "duration[s]", "decimal128(5, 2)",
    # a picture kept in the table
    "image",
)
_DTYPE_VALUES = {"5": 5, "1.5": 1.5, "null": None, "true": True, '["float32"]': ["float32"], '{"a": 1}': {"a": 1}}
_SHAPES = {
    "[1]": [1], "[2]": [2], "[3]": [3], "[19]": [19], "[0]": [0], "[-1]": [-1], "[-3]": [-3], "[]": [],
    "[3, 2]": [3, 2], "[1, 1]": [1, 1], "[2, 2, 2]": [2, 2, 2], "[1, 1, 1, 1]": [1] * 4,
    "[1, 1, 1, 1, 1]": [1] * 5, "[1, 1, 1, 1, 1, 1]": [1] * 6,
    '"3"': "3", '"13"': "13", '""': "", "3": 3, "3.0": 3.0, "true": True, "null": None, "{}": {}, '{"a": 1}': {"a": 1},
    "[3.0]": [3.0], "[1.0]": [1.0], "[true]": [True], "[false]": [False], '["3"]': ["3"], '["1"]': ["1"],
    "[null]": [None], "[2.5]": [2.5], "[1, -1]": [1, -1], "[3, 0]": [3, 0], "[[3]]": [[3]], '[1, "1"]': [1, "1"],
}
FORMS: dict[str, object] = {
    **{f"dtype {name!r}, shape [1]": feature(name, [1]) for name in _DTYPE_NAMES},
    **{f"dtype {name!r}, shape [3]": feature(name, [3]) for name in ("float64", "long", "", "json", "image")},
    **{f"dtype {label} (no string), shape [1]": feature(value, [1]) for label, value in _DTYPE_VALUES.items()},
    "no dtype, shape [1]": feature(shape=[1]),
    **{f"dtype 'float32', shape {label}": feature("float32", shape) for label, shape in _SHAPES.items()},
    "dtype 'float32', no shape": feature("float32"),
    "dtype 'float32', shape [1], no names": {"dtype": "float32", "shape": [1]},
    "dtype 'float32', shape [1], a key more": feature("float32", [1], more=1),
    **{f"the feature is {label}": value for label, value in {
        '"float32"': "float32", "null": None, "[]": [], '["float32", [1]]': ["float32", [1]], "3": 3, "1.5": 1.5,
        "true": True, "{}": {}}.items()},
}

# ── Stored columns: three rows each, by name. Each is a type parquet hands back as written ──
_f64, _f32, _i64 = pa.float64(), pa.float32(), pa.int64()
_NAN, _INF = float("nan"), float("inf")


def _column(kind: pa.DataType, *rows) -> pa.Array:
    return pa.array(list(rows), type=kind)


STORED: dict[str, pa.Array] = {
    # one value a step
    "double": _column(_f64, 1.5, 2.5, 3.5),
    "double, whole": _column(_f64, 2.0, 3.0, 4.0),
    "double, whole, one 300": _column(_f64, 2.0, 300.0, 4.0),
    "double, whole, one -1": _column(_f64, 2.0, -1.0, 4.0),
    "double, 0 and 1": _column(_f64, 0.0, 1.0, 0.0),
    "double, one NaN": _column(_f64, 2.0, _NAN, 4.0),
    "double, one inf": _column(_f64, 2.0, _INF, 4.0),
    "double, one 1e300": _column(_f64, 2.0, 1e300, 4.0),
    "double, one null": _column(_f64, 2.0, None, 4.0),
    "float": _column(_f32, 1.5, 2.5, 3.5),
    "halffloat": _column(pa.float16(), 1.5, 2.5, 3.5),
    "int64": _column(_i64, 2, 3, 4),
    "int64, 0 and 1": _column(_i64, 0, 1, 0),
    "int64, one 127": _column(_i64, 2, 127, 4),
    "int64, one 128": _column(_i64, 2, 128, 4),
    "int64, one 255": _column(_i64, 2, 255, 4),
    "int64, one 256": _column(_i64, 2, 256, 4),
    "int64, one -1": _column(_i64, 2, -1, 4),
    "int64, one -129": _column(_i64, 2, -129, 4),
    "int64, one 2**24+1": _column(_i64, 2, 2 ** 24 + 1, 4),
    "int64, one 2**31": _column(_i64, 2, 2 ** 31, 4),
    "int64, one 2**53+1": _column(_i64, 2, 2 ** 53 + 1, 4),
    "uint64, one 2**63": _column(pa.uint64(), 2, 2 ** 63, 4),
    "int32": _column(pa.int32(), 2, 3, 4),
    "int8": _column(pa.int8(), 2, 3, 4),
    "uint8, one 255": _column(pa.uint8(), 2, 255, 4),
    "bool": _column(pa.bool_(), True, False, True),
    "string, numbers": _column(pa.string(), "1.5", "2.5", "3.5"),
    "string, whole numbers": _column(pa.string(), "2", "3", "4"),
    "string, one word": _column(pa.string(), "2", "abc", "4"),
    "string, one empty": _column(pa.string(), "2", "", "4"),
    "string, true and false": _column(pa.string(), "true", "false", "true"),
    "large_string, whole numbers": _column(pa.large_string(), "2", "3", "4"),
    "binary, whole numbers": _column(pa.binary(), b"2", b"3", b"4"),
    "dictionary<int32, string>, whole numbers": pa.array(["2", "3", "2"]).dictionary_encode(),
    "null": pa.nulls(3),
    "timestamp[ms]": _column(pa.timestamp("ms"), 1, 2, 3),
    "date32": _column(pa.date32(), 1, 2, 3),
    "decimal128(5, 2)": _column(pa.decimal128(5, 2), *(decimal.Decimal(v) for v in ("1.50", "2.50", "3.50"))),
    "struct<x: double>": _column(pa.struct([("x", _f64)]), {"x": 1.5}, {"x": 2.5}, {"x": 3.5}),
    # a list a step
    "list<double>, of 3": _column(pa.list_(_f64), [1.5, 2.5, 3.5], [1.5, 2.5, 3.5], [1.5, 2.5, 3.5]),
    "list<double>, of 3, whole": _column(pa.list_(_f64), [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    "list<double>, of 3, one NaN": _column(pa.list_(_f64), [1.0, 2.0, 3.0], [1.0, _NAN, 3.0], [1.0, 2.0, 3.0]),
    "list<double>, of 3 and one of 2": _column(pa.list_(_f64), [1.5, 2.5, 3.5], [1.5, 2.5], [1.5, 2.5, 3.5]),
    "list<double>, of 3 and one of 4": _column(pa.list_(_f64), [1.5, 2.5, 3.5], [1.5, 2.5, 3.5, 4.5], [1.5, 2.5, 3.5]),
    "list<double>, of 3 and one empty": _column(pa.list_(_f64), [1.5, 2.5, 3.5], [], [1.5, 2.5, 3.5]),
    "list<double>, of 3 and one null": _column(pa.list_(_f64), [1.5, 2.5, 3.5], None, [1.5, 2.5, 3.5]),
    "list<double>, every one null": _column(pa.list_(_f64), None, None, None),
    "list<double>, of 3, a null inside": _column(pa.list_(_f64), [1.5, 2.5, 3.5], [1.5, None, 3.5], [1.5, 2.5, 3.5]),
    "list<double>, of 2": _column(pa.list_(_f64), [1.5, 2.5], [1.5, 2.5], [1.5, 2.5]),
    "list<double>, of 1": _column(pa.list_(_f64), [1.5], [2.5], [3.5]),
    "large_list<double>, of 3": _column(pa.large_list(_f64), [1.5, 2.5, 3.5], [1.5, 2.5, 3.5], [1.5, 2.5, 3.5]),
    "list<float>, of 3": _column(pa.list_(_f32), [1.5, 2.5, 3.5], [1.5, 2.5, 3.5], [1.5, 2.5, 3.5]),
    "list<int64>, of 3": _column(pa.list_(_i64), [1, 2, 3], [1, 2, 3], [1, 2, 3]),
    "list<int64>, of 3, one 300": _column(pa.list_(_i64), [1, 2, 3], [1, 300, 3], [1, 2, 3]),
    "list<string>, of 3, numbers": _column(pa.list_(pa.string()), ["1", "2", "3"], ["1", "2", "3"], ["1", "2", "3"]),
    "list<string>, of 3, one word": _column(pa.list_(pa.string()), ["1", "2", "3"], ["1", "abc", "3"], ["1", "2", "3"]),
    "list<null>, of 3": _column(pa.list_(pa.null()), [None] * 3, [None] * 3, [None] * 3),
    "list<null>, empty": _column(pa.list_(pa.null()), [], [], []),
    "fixed<double>[3]": _column(pa.list_(_f64, 3), [1.5, 2.5, 3.5], [1.5, 2.5, 3.5], [1.5, 2.5, 3.5]),
    "fixed<double>[3], whole": _column(pa.list_(_f64, 3), [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    "fixed<double>[3], a null inside": _column(pa.list_(_f64, 3), [1.5, 2.5, 3.5], [1.5, None, 3.5], [1.5, 2.5, 3.5]),
    "fixed<double>[2]": _column(pa.list_(_f64, 2), [1.5, 2.5], [1.5, 2.5], [1.5, 2.5]),
    "fixed<double>[1]": _column(pa.list_(_f64, 1), [1.5], [2.5], [3.5]),
    "fixed<float>[3]": _column(pa.list_(_f32, 3), [1.5, 2.5, 3.5], [1.5, 2.5, 3.5], [1.5, 2.5, 3.5]),
    "fixed<int32>[3], one 300": _column(pa.list_(pa.int32(), 3), [1, 2, 3], [1, 300, 3], [1, 2, 3]),
    "list<list<double>>, 3 of 1": _column(pa.list_(pa.list_(_f64)), [[1.5], [2.5], [3.5]], [[1.5], [2.5], [3.5]],
                                         [[1.5], [2.5], [3.5]]),
    "fixed<fixed<double>[1]>[3]": _column(pa.list_(pa.list_(_f64, 1), 3), [[1.5], [2.5], [3.5]], [[1.5], [2.5], [3.5]],
                                          [[1.5], [2.5], [3.5]]),
    "list<struct<x: double>>, of 3": _column(pa.list_(pa.struct([("x", _f64)])), [{"x": 1.5}] * 3, [{"x": 1.5}] * 3,
                                            [{"x": 1.5}] * 3),
    # an Arrow extension type: a fixed-length list underneath, which is what datasets casts
    "fixed_shape_tensor<double>[3]": pa.FixedShapeTensorArray.from_numpy_ndarray(np.asarray([[1.5, 2.5, 3.5]] * 3)),
}
# Not here: a fixed-length list with a null row. Parquet stores a null one as a list of no
# elements and the reader then cannot read the file at all ("Expected all lists to be of
# size=3"): a file no cast is ever asked about.

# ── What a column is declared as: the two kinds the validator holds a column to ──
DECLARED: dict[str, dict] = {
    **{f"{dtype} [1]": feature(dtype, [1]) for dtype in (
        "float64", "float32", "float16", "int64", "int32", "int8", "uint8", "uint64", "bool", "string",
        "large_string", "binary", "null")},
    **{f"{dtype} [{n}]": feature(dtype, [n]) for dtype, n in (
        ("float64", 3), ("float32", 3), ("int64", 3), ("int8", 3), ("bool", 3), ("string", 3), ("null", 3),
        ("float64", 2), ("float64", 4))},
}

# ── A null list among lists of any length: where it stands among the file's row groups ──
# One column of BATCH_ROWS steps, stored as lists of any length (step i holds batch_values(i)) and
# declared a list of 3. The loader casts batch by batch -- a batch as many rows as the file's first
# row group -- and what it hands out after a null list turns on whether a batch with no null in it
# follows the null in its row group. Each case: (the row groups' row counts, the steps that are null).
# The steps are two episodes (BATCH_EPISODE_ROWS each): each case is also read for chosen episodes.
BATCH_ROWS = 10
BATCH_EPISODE_ROWS = 5
BATCHES: dict[str, tuple[list[int], set[int]]] = {
    "one row group of 10, no null": ([10], set()),
    "one row group of 10, a null at 3": ([10], {3}),
    "row groups 5 5, a null at 2": ([5, 5], {2}),
    "row groups 5 5, a null at 9": ([5, 5], {9}),
    "row groups 2 2 2 2 2, a null at 5": ([2] * 5, {5}),
    "row groups 2 8, no null": ([2, 8], set()),
    "row groups 2 8, a null at 0": ([2, 8], {0}),
    "row groups 2 8, a null at 2": ([2, 8], {2}),
    "row groups 2 8, a null at 3": ([2, 8], {3}),
    "row groups 2 8, a null at 8": ([2, 8], {8}),
    "row groups 2 8, a null at 9": ([2, 8], {9}),
    "row groups 2 8, nulls at 2 and 9": ([2, 8], {2, 9}),
    "row groups 2 8, nulls at 3 and 8": ([2, 8], {3, 8}),
    "row groups 2 8, nulls at 2 4 6 8": ([2, 8], {2, 4, 6, 8}),
    "row groups 2 8, nulls at 2 3 and 9": ([2, 8], {2, 3, 9}),
    "row groups 1 9, a null at 1": ([1, 9], {1}),
    "row groups 1 9, a null at 9": ([1, 9], {9}),
    "row groups 1 9, nulls at 1 and 9": ([1, 9], {1, 9}),
    "row groups 1 9, nulls at 5 6 7 8 9": ([1, 9], {5, 6, 7, 8, 9}),
    "row groups 1 4 5, a null at 2": ([1, 4, 5], {2}),
    "row groups 3 7, a null at 3": ([3, 7], {3}),
    "row groups 3 7, nulls at 3 and 9": ([3, 7], {3, 9}),
    "row groups 3 3 4, a null at 6": ([3, 3, 4], {6}),
    "row groups 3 3 4, nulls at 6 and 9": ([3, 3, 4], {6, 9}),
    "row groups 4 6, nulls at 4 and 8": ([4, 6], {4, 8}),
}


def batch_values(step: int) -> list[float]:
    return [float(step), step + 0.25, step + 0.5]


def batch_column(nulls: set[int], kind: pa.DataType = pa.list_(pa.float64()), alike: bool = False) -> pa.Array:
    """The column of a placing: every step's own values, or -- ``alike`` -- the same values in
    every list, where a list read in another's place shows nothing."""
    return pa.array([None if step in nulls else batch_values(0 if alike else step) for step in range(BATCH_ROWS)],
                    type=kind)
