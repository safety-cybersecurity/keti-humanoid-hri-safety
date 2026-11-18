"""The types the loader's cast was measured on, and where the measurement is kept.

Not a test module. Shared by the test that holds the validator's rule to the measurement
(test_data_spans.py) and by the script that takes it (scripts/measure_loader_casts.py), so
that both speak of the same types by the same names.

The rule in question (validator/validate_dataset.py ``_never_fits``): LeRobot's loader types
the episode table by its first file and casts every later file to that. Which differences
between two files' columns make that cast fail *whatever the values are* is not something to
reason out a pair at a time -- a rule reasoned that way misses the pairs nobody reasoned about.
It is measured: every ordered pair of the types below, through the loader's own cast.
"""

from pathlib import Path

import pyarrow as pa

MEASURED = Path(__file__).with_name("loader_casts_measured.json")

_STRINGS = pa.list_(pa.string())

# By name, so that the measurement file can list pairs of them. Leaves first, then the shapes
# an episode table's column can take (lists, as pandas writes them, fixed-length lists, structs).
# Each is a type parquet hands back as it was written -- the validator and the loader both see a
# file's types through its footer (a timestamp in seconds, say, comes back in milliseconds).
TYPES = {
    "null": pa.null(),
    "bool": pa.bool_(),
    "int8": pa.int8(),
    "int64": pa.int64(),
    "double": pa.float64(),
    "string": pa.string(),
    "large_string": pa.large_string(),
    "binary": pa.binary(),
    "timestamp[ms]": pa.timestamp("ms"),
    "list<null>": pa.list_(pa.null()),
    "list<string>": _STRINGS,
    "list<int64>": pa.list_(pa.int64()),
    "large_list<string>": pa.large_list(pa.string()),
    "list<list<string>>": pa.list_(_STRINGS),
    "fixed<string>[1]": pa.list_(pa.string(), 1),
    "fixed<string>[2]": pa.list_(pa.string(), 2),
    "struct<a:string>": pa.struct([("a", pa.string())]),
    "struct<a:null>": pa.struct([("a", pa.null())]),
    "struct<a:string,b:int64>": pa.struct([("a", pa.string()), ("b", pa.int64())]),
    "list<struct<a:string>>": pa.list_(pa.struct([("a", pa.string())])),
}

LEAVES = frozenset(name for name, kind in TYPES.items()
                   if not (pa.types.is_struct(kind) or pa.types.is_list(kind)
                           or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind)))


def _friendly(kind: pa.DataType, length: int):
    """A value of ``kind`` that casts wherever a value can: 1, "1", true; lists of ``length``."""
    t = pa.types
    if t.is_null(kind):
        return None
    if t.is_struct(kind):
        return {field.name: _friendly(field.type, length) for field in kind}
    if t.is_fixed_size_list(kind):
        return [_friendly(kind.value_type, length)] * kind.list_size
    if t.is_list(kind) or t.is_large_list(kind):
        return [_friendly(kind.value_type, length)] * length
    if t.is_boolean(kind):
        return True
    if t.is_floating(kind):
        return 1.0
    if t.is_string(kind) or t.is_large_string(kind):
        return "1"
    if t.is_binary(kind):
        return b"1"
    return 1                                    # integers, and a timestamp counted from the epoch


def _holds_a_fixed_size_list(kind: pa.DataType) -> bool:
    t = pa.types
    if t.is_fixed_size_list(kind):
        return True
    if t.is_list(kind) or t.is_large_list(kind):
        return _holds_a_fixed_size_list(kind.value_type)
    return t.is_struct(kind) and any(_holds_a_fixed_size_list(field.type) for field in kind)


def candidate_values(name: str) -> dict[str, pa.Array]:
    """Two rows of the type, several ways: values chosen so that a cast that can succeed does
    (lists of one and of two, for a fixed-length target of either size), and all null.

    Not null where the type holds a fixed-length list: parquet stores a null one as a list of
    no elements, and the reader then cannot read the file at all, whatever comes after
    ("Expected all lists to be of size=1"). That is a file no cast is ever asked about.
    """
    kind = TYPES[name]
    out = {}
    for length in (1, 2):
        value = _friendly(kind, length)
        out[f"friendly, lists of {length}"] = pa.array([value, value], type=kind)
    if not _holds_a_fixed_size_list(kind):
        out["nulls"] = pa.nulls(2, kind)
    return out
