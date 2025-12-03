"""Create a macro events calendar with optional additional flags."""

from __future__ import annotations

import _project_paths  # noqa: F401  # adds repo root to sys.path
from pathlib import Path
from typing import Iterable

import pandas as pd

from config.config import CLEAN_DIR, RAW_DIR

DEFAULT_COLUMNS = ["date", "is_fomc", "is_cpi"]


def load_macro_sources(files: Iterable[Path]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in files:
        if not path.exists():
            print(f"Skipping missing macro file: {path}")
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        frames.append(df)
    return frames


def _default_calendar() -> pd.DataFrame:
    """Return a lightweight default macro calendar.

    If users do not supply their own CSVs under ``data/raw/macro``, we still
    want downstream merges to have meaningful macro markers. The defaults are
    built from the publicly available 2024 FOMC meeting dates and CPI release
    days published by the Federal Reserve and BLS.
    """

    fomc_dates = pd.to_datetime(
        [
            "2024-01-31",
            "2024-03-20",
            "2024-05-01",
            "2024-06-12",
            "2024-07-31",
            "2024-09-18",
            "2024-11-07",
            "2024-12-18",
        ]
    )

    cpi_dates = pd.to_datetime(
        [
            "2024-01-11",
            "2024-02-13",
            "2024-03-12",
            "2024-04-10",
            "2024-05-15",
            "2024-06-12",
            "2024-07-11",
            "2024-08-14",
            "2024-09-11",
            "2024-10-10",
            "2024-11-13",
            "2024-12-11",
        ]
    )

    default_dates = pd.Index(fomc_dates).union(pd.Index(cpi_dates))
    default = pd.DataFrame({"date": default_dates})
    default["is_fomc"] = default["date"].isin(fomc_dates).astype(int)
    default["is_cpi"] = default["date"].isin(cpi_dates).astype(int)
    return default


def main() -> None:
    raw_macro_dir = RAW_DIR / "macro"
    out_path = CLEAN_DIR / "macro_events.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    macro_files = sorted(raw_macro_dir.glob("*.csv"))
    frames = load_macro_sources(macro_files)

    if frames:
        macro = pd.concat(frames, ignore_index=True)
    else:
        macro = _default_calendar()

    if "date" not in macro.columns:
        macro["date"] = pd.NaT
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.dropna(subset=["date"])
    macro = macro.sort_values("date").reset_index(drop=True)

    # Ensure dummy columns exist
    for col in DEFAULT_COLUMNS:
        if col not in macro.columns:
            macro[col] = 0

    macro.to_csv(out_path, index=False)
    print(f"Saved macro events to {out_path}")


if __name__ == "__main__":
    main()
