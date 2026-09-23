"""Build representative, deterministic data for Factor CoSTEER profiling."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS

PRIMARY_DAILY_FILE = "daily_pv.h5"
PROFILE_META_FILE = "profile_meta.json"


def _sorted_unique(values: pd.Index) -> list[Any]:
    """Sort heterogeneous labels deterministically without changing their values."""
    return sorted(values.unique().tolist(), key=lambda value: str(value))


def _slice_compatible_frame(
    frame: pd.DataFrame,
    instruments: set[Any],
    datetimes: set[Any],
) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex):
        return frame
    if "datetime" not in frame.index.names or "instrument" not in frame.index.names:
        return frame
    mask = frame.index.get_level_values("datetime").isin(datetimes) & frame.index.get_level_values(
        "instrument",
    ).isin(instruments)
    return frame.loc[mask]


def _store_nrows(store: pd.HDFStore, key: str) -> int | None:
    storer = store.get_storer(key)
    nrows = getattr(storer, "nrows", None)
    if nrows is not None:
        return int(nrows)
    shape = getattr(storer, "shape", None)
    if shape is not None and len(shape):
        return int(shape[0])
    return None


def _write_hdf_file(
    source: Path,
    destination: Path,
    instruments: set[Any],
    datetimes: set[Any],
    preloaded: dict[str, pd.DataFrame] | None = None,
) -> None:
    preloaded = preloaded or {}
    with pd.HDFStore(source, mode="r") as source_store, pd.HDFStore(destination, mode="w") as target_store:
        for key in source_store.keys():
            value = preloaded.get(key)
            if value is None:
                value = source_store.get(key)
            if isinstance(value, pd.DataFrame):
                value = _slice_compatible_frame(value, instruments, datetimes)
            storer = source_store.get_storer(key)
            format_type = getattr(storer, "format_type", "fixed")
            target_store.put(key, value, format=format_type)


def build_profile_dataset(
    source_folder: str | Path | None = None,
    output_folder: str | Path | None = None,
    instrument_count: int = 40,
    date_count: int = 256,
) -> dict[str, Any]:
    """Build a continuous-date profile dataset and return its metadata."""
    source = Path(source_folder or FACTOR_COSTEER_SETTINGS.data_folder)
    output = Path(output_folder or FACTOR_COSTEER_SETTINGS.data_folder_profile)
    primary_path = source / PRIMARY_DAILY_FILE
    if not primary_path.is_file():
        raise FileNotFoundError(f"Primary profile source is missing: {primary_path}")
    if source.resolve() == output.resolve():
        raise ValueError("Profile output folder must differ from the full source folder.")
    if instrument_count <= 0 or date_count <= 0:
        raise ValueError("instrument_count and date_count must be positive.")

    with pd.HDFStore(primary_path, mode="r") as store:
        keys = store.keys()
        if not keys:
            raise ValueError(f"No HDF keys found in {primary_path}")
        primary_key = "/data" if "/data" in keys else keys[0]
        metadata_rows = _store_nrows(store, primary_key)
        primary = store.get(primary_key)
    if not isinstance(primary, pd.DataFrame):
        raise TypeError(f"Primary HDF key {primary_key} is not a DataFrame.")
    if not isinstance(primary.index, pd.MultiIndex) or not {"datetime", "instrument"}.issubset(
        primary.index.names,
    ):
        raise ValueError("Primary daily data must use a MultiIndex containing datetime and instrument.")

    full_rows = metadata_rows if metadata_rows is not None else len(primary)
    all_dates = _sorted_unique(primary.index.get_level_values("datetime"))
    selected_dates = all_dates[-date_count:]
    recent_mask = primary.index.get_level_values("datetime").isin(selected_dates)
    recent_instruments = primary.index.get_level_values("instrument")[recent_mask]
    counts = pd.Series(recent_instruments).value_counts()
    ranked_instruments = sorted(counts.index.tolist(), key=lambda value: (-int(counts[value]), str(value)))
    selected_instruments = ranked_instruments[:instrument_count]

    instrument_set = set(selected_instruments)
    datetime_set = set(selected_dates)
    profile_primary = _slice_compatible_frame(primary, instrument_set, datetime_set)
    profile_rows = len(profile_primary)
    if profile_rows == 0:
        raise ValueError("The selected profile window contains no rows.")
    if profile_rows >= full_rows:
        raise ValueError(
            "Profile selection is not smaller than the full dataset; reduce instrument_count or date_count.",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for item in sorted(source.iterdir(), key=lambda entry: entry.name):
            destination = temporary / item.name
            if item.is_file() and item.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
                preload = {primary_key: profile_primary} if item == primary_path else None
                _write_hdf_file(item, destination, instrument_set, datetime_set, preload)
            elif item.is_file():
                shutil.copy2(item, destination)
            elif item.is_dir():
                shutil.copytree(item, destination, symlinks=True)

        metadata = {
            "profile_rows": profile_rows,
            "full_rows": full_rows,
            "scale_ratio": full_rows / profile_rows,
            "instrument_count": len(selected_instruments),
            "date_count": len(selected_dates),
            "primary_file": PRIMARY_DAILY_FILE,
            "primary_key": primary_key,
        }
        (temporary / PROFILE_META_FILE).write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=FACTOR_COSTEER_SETTINGS.data_folder)
    parser.add_argument("--output", default=FACTOR_COSTEER_SETTINGS.data_folder_profile)
    parser.add_argument("--instruments", type=int, default=40)
    parser.add_argument("--dates", type=int, default=256)
    args = parser.parse_args()
    metadata = build_profile_dataset(args.source, args.output, args.instruments, args.dates)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
