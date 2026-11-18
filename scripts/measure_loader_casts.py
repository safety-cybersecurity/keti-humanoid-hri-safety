#!/usr/bin/env python3
"""Measure what LeRobot's loader cannot put into the table it types, for its two tables.

The validator reports a release whose files the loader cannot read as one table (C7). Its rules
for that are copies of the loader's behaviour, and this is how the copies are held to the
original: the cases go through the loader's own code and what stops it is written down, for a
test to compare the rule with, case by case.

1. The episode table (validator/validate_dataset.py ``_never_fits``). The loader types it by its
   first file and casts every later file to it (datasets' parquet reader: features from the first
   file's schema, then table_cast per batch). Every ordered pair of the types in
   tests/validator/loader_casts.py; the pairs that fail for every choice of values are written
   to tests/validator/loader_casts_measured.json, which tests/validator/test_data_spans.py reads.

2. The step table (``_feature_column`` · ``_cast_stops`` · ``_null_lists_misread``). The loader
   types it by info.json's features -- lerobot's DatasetInfo and get_hf_features_from_features --
   and casts every step file to that, batch by batch. The forms a feature can take, every stored
   column beside every declaration, and the placings of a null list among a file's row groups,
   of tests/validator/feature_casts.py; what stops the loader, and where it hands out another
   step's values, is written to tests/validator/feature_casts_measured.json, which
   tests/validator/test_step_features.py reads.

A case that has a file to read is taken two ways, which have to agree:
  - the cast the reader calls, in memory: table_cast of the file's table to the features' schema
    (for the placings of nulls, of each batch the reader's own fragment hands out -- filtered by
    episode where episodes are chosen);
  - the reader itself: Dataset.from_parquet on the files -- for the step table through
    lerobot.datasets.io_utils.load_nested_dataset, the function the loader calls.
Where the loader stops on a placing of nulls the two agree that it stops; the step it had wrong
first by then is the in-memory way's alone to say.
The six forms of info.json's ``features`` as a whole have no file to read: they are taken one
way, through DatasetInfo and get_hf_features_from_features, which is where the loader stops on
them. For the episode table a pair is tried with several sets of values (all null; values picked
to cast wherever a value can): what is written is the pairs no values get through.

Needs the interpreter that has lerobot's `datasets`. Run it again when that package changes
(about a minute):

    .venv-lerobot/bin/python scripts/measure_loader_casts.py            # compare with the files kept (exit 1 on a difference)
    .venv-lerobot/bin/python scripts/measure_loader_casts.py --write    # write the files
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _cases_module(name: str):
    """tests/validator/<name>.py, loaded by its path: the interpreter that has lerobot also has a
    package of its own called ``tests`` in site-packages, which hides this repository's."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tests" / "validator" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _types_module():
    return _cases_module("loader_casts")


def measure_features(scratch: Path) -> dict:
    """The step table: what info.json's features make of it, and of the columns stored."""
    import datasets
    import pyarrow as pa
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq
    from datasets.table import table_cast
    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import load_nested_dataset
    from lerobot.datasets.utils import DatasetInfo

    cases = _cases_module("feature_casts")
    datasets.disable_progress_bars()
    # The reader logs each file it fails on, and here a file it fails on is the measurement.
    logging.getLogger("datasets").setLevel(logging.CRITICAL)
    row = pa.array([0, 1, 2], pa.int64())           # a column every case has and declares
    known = {"y": cases.feature("int64", [1])}
    disagreements: list[list] = []
    count = 0

    def table_of(features: dict):
        """The loader's own way from info.json's features to the table's schema: the JSON as a
        DatasetInfo (io_utils.load_info), then lerobot's conversion of its features."""
        info = DatasetInfo.from_dict(json.loads(json.dumps(
            {"codebase_version": "v3.0", "fps": 50, "features": features})))
        return get_hf_features_from_features(info.features)

    def reads(label: tuple, columns: dict, features: dict) -> bool:
        nonlocal count
        case = scratch / f"features-{count:05d}"
        count += 1
        file = case / "data" / "chunk-000" / "file-000.parquet"
        file.parent.mkdir(parents=True)
        pq.write_table(pa.table({"y": row, **columns}), file)
        for name, values in columns.items():        # both sides see the types through the file
            assert pq.read_schema(file).field(name).type == values.type, (label, pq.read_schema(file))
        try:
            table_cast(pq.read_table(file), table_of(features).arrow_schema)
            cast = True
        except Exception:  # noqa: BLE001 -- any refusal is the measurement
            cast = False
        try:
            load_nested_dataset(case / "data", features=table_of(features))
            read = True
        except Exception:  # noqa: BLE001
            read = False
        if read != cast:
            disagreements.append([*label, f"cast in memory: {cast}", f"reader: {read}"])
        return read

    forms = [label for label, form in cases.FORMS.items()
             if not reads(("form", label), {}, {**known, "x": form})]
    casts = [[stored, declared]
             for stored, values in cases.STORED.items() for declared, form in cases.DECLARED.items()
             if not reads(("cast", stored, declared), {"x": values}, {**known, "x": form})]
    double = pa.array([1.5, 2.5, 3.5], pa.float64())
    columns = {
        "a column the features do not declare": reads(("columns", "undeclared"), {"x": double}, known),
        "a column declared as a camera stream": reads(
            ("columns", "video"), {"x": double},
            {**known, "x": {"dtype": "video", "shape": [4, 4, 3], "names": ["height", "width", "channels"]}}),
        "a feature no file has a column of": reads(
            ("columns", "absent"), {}, {**known, "x": cases.feature("float64", [1])}),
    }
    # A null list among lists of any length that are declared a fixed length: what comes out,
    # step by step, beside what was stored -- stored as a list and as a large_list, which have to
    # agree as the two ways do.
    declared_three = {**known, "episode_index": cases.feature("int64", [1]), "x": cases.feature("float64", [3])}
    episode_of = [step // cases.BATCH_EPISODE_ROWS for step in range(cases.BATCH_ROWS)]
    chosen = [[0], [1], sorted(set(episode_of))]    # what LeRobotDataset(..., episodes=...) hands the reader
    wrong_when_chosen: list[list] = []

    def handed_out(label: str, groups: list[int], nulls: set[int], kind, alike: bool) -> tuple[str, int | None]:
        """What comes out of one placing: ("reads", None) · ("shifts", the first step not as stored) ·
        ("stops", the first step that is not as stored in the batches cast before the one that fails,
        or that batch's first step -- known only in memory: the reader hands out nothing)."""
        nonlocal count
        case = scratch / f"features-{count:05d}"
        count += 1
        file = case / "data" / "chunk-000" / "file-000.parquet"
        file.parent.mkdir(parents=True)
        column = cases.batch_column(nulls, kind, alike)
        stored = column.to_pylist()
        table = pa.table({"y": pa.array(range(len(stored)), pa.int64()),
                          "episode_index": pa.array(episode_of, pa.int64()), "x": column})
        start = 0
        with pq.ParquetWriter(file, table.schema) as writer:
            for rows in groups:
                writer.write_table(table.slice(start, rows))
                start += rows
        assert [pq.read_metadata(file).row_group(k).num_rows for k in range(len(groups))] == groups, label

        def in_memory(episodes: list[int] | None, kept: list) -> tuple[str, int | None]:
            # The batches the reader makes (datasets' parquet module): as many rows as the first row
            # group, filtered by episode_index where episodes are chosen, each cast by itself.
            schema = table_of(declared_three).arrow_schema
            keep = None if episodes is None else pads.field("episode_index").isin(episodes)
            got: list = []
            first: int | None = None
            with open(file, "rb") as opened:
                fragment = pads.ParquetFileFormat().make_fragment(opened)
                batches = fragment.to_batches(batch_size=fragment.row_groups[0].num_rows, filter=keep,
                                              batch_readahead=0, fragment_readahead=0)
                for batch in batches:
                    try:
                        rows = table_cast(pa.Table.from_batches([batch]), schema).column("x").to_pylist()
                    except Exception:  # noqa: BLE001 -- any refusal is the measurement
                        return "stops", len(got) if first is None else first
                    for k, row in enumerate(rows):
                        if first is None and row != kept[len(got) + k]:
                            first = len(got) + k
                    got.extend(rows)
            return ("reads", None) if first is None and len(got) == len(kept) else ("shifts", first)

        def reader(episodes: list[int] | None, kept: list) -> tuple[str, int | None]:
            try:
                got = load_nested_dataset(case / "data", features=table_of(declared_three),
                                          episodes=episodes).data.column("x").to_pylist()
            except Exception:  # noqa: BLE001
                return "stops", None
            wrong = [step for step, (a, b) in enumerate(zip(got, kept)) if a != b]
            return ("shifts", wrong[0]) if wrong or len(got) != len(kept) else ("reads", None)

        def both_ways(episodes: list[int] | None) -> tuple[str, int | None]:
            kept = [value for value, episode in zip(stored, episode_of) if episodes is None or episode in episodes]
            cast, read = in_memory(episodes, kept), reader(episodes, kept)
            # A stop is a stop to both; where it stops only the in-memory way has a step to name.
            if (cast[0], read[0]) != ("stops", "stops") and read != cast:
                disagreements.append(["batches", label, str(kind), f"alike: {alike}", f"episodes: {episodes}",
                                      f"cast in memory: {cast}", f"reader: {read}"])
            return cast

        whole_release = both_ways(None)
        # The same file opened for chosen episodes: the reader filters each batch by episode_index.
        for episodes in chosen:
            if both_ways(episodes) != ("reads", None):
                wrong_when_chosen.append([label, str(kind), f"alike: {alike}", episodes])
        return whole_release

    def placings(alike: bool) -> dict[str, tuple[str, int | None]]:
        out = {}
        for label, (groups, nulls) in cases.BATCHES.items():
            as_list = handed_out(label, groups, nulls, pa.list_(pa.float64()), alike)
            as_large = handed_out(label, groups, nulls, pa.large_list(pa.float64()), alike)
            if as_list != as_large:
                disagreements.append(["batches", label, f"alike: {alike}", f"list: {as_list}", f"large_list: {as_large}"])
            out[label] = as_list
        return out

    batches = placings(alike=False)                 # every step its own values
    alike = placings(alike=True)                    # every list the same values

    whole = [("without features", {"codebase_version": "v3.0", "fps": 50}),
             *((f"features {json.dumps(value)}", {"codebase_version": "v3.0", "fps": 50, "features": value})
               for value in (None, [], "", 3, {}))]
    info_stops = []
    for label, document in whole:
        try:
            get_hf_features_from_features(DatasetInfo.from_dict(json.loads(json.dumps(document))).features)
        except Exception:  # noqa: BLE001
            info_stops.append(label)
    import lerobot

    return {
        "what": "the step table. forms: features of info.json, as the feature of a column no step file has, "
                "that stop the loader making its table (every other one of `all` gave a table). casts: pairs "
                "(stored column, declaration) of tests/validator/feature_casts.py the loader cannot cast (every "
                "other pair of `stored` x `declared` it read). columns: whether the loader read the file. "
                "batches: a null list among lists of any length declared a list of 3, by the file's row groups "
                "-- with every step holding values of its own, the placings that stop the loader and the ones "
                "it reads through handing out another step's values, each with the first step that is not as "
                "stored (where it stops, in the batches cast in memory up to the one that fails: the reader "
                "hands out nothing then); every other one of `all` it read as stored. alike: the same placings "
                "with every list holding the same values -- the ones that stop it, and the ones not read as "
                "stored. And the placings (of either kind of values) not read as stored when the same file is "
                "read for each choice of `episodes_chosen`, as LeRobot reads a release opened for chosen episodes. "
                "info: what info.json holds for its features that stops the loader before any table",
        "measured_with": {"lerobot": lerobot.__version__, "datasets": datasets.__version__, "pyarrow": pa.__version__},
        "forms": {"all": list(cases.FORMS), "stops": forms},
        "casts": {"stored": list(cases.STORED), "declared": list(cases.DECLARED), "stops": casts},
        "columns": {label: "reads" if went else "stops" for label, went in columns.items()},
        "batches": {"all": list(cases.BATCHES),
                    "stops": {label: first for label, (what, first) in batches.items() if what == "stops"},
                    "shifts": {label: first for label, (what, first) in batches.items() if what == "shifts"},
                    "alike": {"stops": [label for label, (what, _) in alike.items() if what == "stops"],
                              "not_as_stored": [label for label, (what, _) in alike.items() if what == "shifts"]},
                    "episodes_chosen": chosen,
                    "not_as_stored_when_episodes_are_chosen": wrong_when_chosen},
        "info": {"all": [label for label, _ in whole], "stops": info_stops},
        "reader_and_cast_disagree": disagreements,
    }


def measure(scratch: Path) -> dict:
    import datasets
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset, Features
    from datasets.table import table_cast

    kinds = _types_module()
    TYPES, candidate_values = kinds.TYPES, kinds.candidate_values
    datasets.disable_progress_bars()

    def write(path: Path, values: pa.Array, at: int) -> None:
        pq.write_table(pa.table({"episode_index": pa.array([at, at + 1], pa.int64()), "x": values}), path)

    fails, disagreements = [], []
    for k, first in enumerate(TYPES):
        for j, later in enumerate(TYPES):
            went = []
            for v, (label, values) in enumerate(candidate_values(later).items()):
                pair = scratch / f"{k:02d}-{j:02d}-{v}"
                pair.mkdir()
                files = [pair / "file-000.parquet", pair / "file-001.parquet"]
                # The first file's values do not matter to the cast, only that it reads.
                write(files[0], next(iter(candidate_values(first).values())), 0)
                write(files[1], values, 2)
                # Both sides see the types through the footers, as the validator does.
                assert pq.read_schema(files[0]).field("x").type == TYPES[first], (first, pq.read_schema(files[0]))
                assert pq.read_schema(files[1]).field("x").type == TYPES[later], (later, pq.read_schema(files[1]))
                # The cast the reader calls ...
                try:
                    table_cast(pq.read_table(files[1]), Features.from_arrow_schema(pq.read_schema(files[0])).arrow_schema)
                    cast = True
                except Exception:  # noqa: BLE001 -- any refusal is the measurement
                    cast = False
                # ... and the reader.
                try:
                    Dataset.from_parquet([str(f) for f in files], cache_dir=str(pair / "cache"))
                    read = True
                except Exception:  # noqa: BLE001
                    read = False
                if read != cast:
                    disagreements.append([first, later, label, f"cast in memory: {cast}", f"reader: {read}"])
                if read:
                    went.append(label)
            if not went:
                fails.append([first, later])
    import lerobot  # only for the version this was measured under

    return {
        "what": "ordered pairs (first file's type, later file's type) of one column that the loader's reader "
                "cannot put into one table whatever the later file's values are; every other pair of `types` "
                "was read for some choice of values",
        "measured_with": {"lerobot": lerobot.__version__, "datasets": datasets.__version__, "pyarrow": pa.__version__},
        "types": list(TYPES),
        "fails": fails,
        "reader_and_cast_disagree": disagreements,
    }


def _stopped(result: dict) -> set:
    """Every case of a measurement that stopped the loader, of either table's file."""
    if "fails" in result:
        return {tuple(pair) for pair in result["fails"]}
    return ({("form", label) for label in result.get("forms", {}).get("stops", [])}
            | {("cast", *pair) for pair in result.get("casts", {}).get("stops", [])}
            | {("columns", label) for label, went in result.get("columns", {}).items() if went == "stops"}
            | {("batches", label) for label in result.get("batches", {}).get("stops", [])}
            | {("batches", label, "shifts") for label in result.get("batches", {}).get("shifts", {})}
            | {("info", label) for label in result.get("info", {}).get("stops", [])})


def _kept_or_written(result: dict, measured: Path, write: bool) -> int:
    for row in result["reader_and_cast_disagree"]:
        print("  disagree:", row)
    if result["reader_and_cast_disagree"]:
        print("the two measurements disagree -- nothing written")
        return 1
    text = json.dumps(result, ensure_ascii=False, indent=1) + "\n"
    if write:
        measured.write_text(text, encoding="utf-8")
        print(f"written: {measured.relative_to(REPO_ROOT)}")
        return 0
    if not measured.exists():
        print(f"{measured.relative_to(REPO_ROOT)} is not there -- run with --write")
        return 1
    kept = json.loads(measured.read_text(encoding="utf-8"))
    if kept != result:
        now, then = _stopped(result), _stopped(kept)
        print("differs from the file kept:")
        print("  stops now, not in the file:", sorted(now - then))
        print("  in the file, reads now:", sorted(then - now))
        print("  measured_with:", result["measured_with"], "· file:", kept.get("measured_with"))
        return 1
    print(f"same as {measured.relative_to(REPO_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write the measurements to the files the tests read")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="loader-casts-") as scratch:
        os.environ["HF_HOME"] = str(Path(scratch) / "hf")
        between_files = measure(Path(scratch))
        to_features = measure_features(Path(scratch))
    n = len(between_files["types"])
    print(f"episode table: {n} types · {n * n} ordered pairs · {len(between_files['fails'])} that never fit · "
          f"reader and in-memory cast disagree on {len(between_files['reader_and_cast_disagree'])}")
    status = _kept_or_written(between_files, _types_module().MEASURED, args.write)
    forms, casts = to_features["forms"], to_features["casts"]
    batches = to_features["batches"]
    print(f"step table: {len(forms['all'])} forms of a feature · {len(forms['stops'])} that stop the loader · "
          f"{len(casts['stored'])} stored columns x {len(casts['declared'])} declarations · "
          f"{len(casts['stops'])} it cannot cast · {len(batches['all'])} placings of null lists · "
          f"{len(batches['stops'])} that stop it · {len(batches['shifts'])} read with another step's values · "
          f"reader and in-memory cast disagree on {len(to_features['reader_and_cast_disagree'])}")
    return max(status, _kept_or_written(to_features, _cases_module("feature_casts").MEASURED, args.write))


if __name__ == "__main__":
    raise SystemExit(main())
