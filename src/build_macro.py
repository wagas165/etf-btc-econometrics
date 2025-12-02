"""Create a macro events calendar with optional additional flags."""

from __future__ import annotations

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


def main() -> None:
    raw_macro_dir = RAW_DIR / "macro"
    out_path = CLEAN_DIR / "macro_events.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    macro_files = sorted(raw_macro_dir.glob("*.csv"))
    frames = load_macro_sources(macro_files)

    if frames:
        macro = pd.concat(frames, ignore_index=True)
    else:
        # Provide an empty scaffold so downstream merges do not fail
        macro = pd.DataFrame(columns=DEFAULT_COLUMNS)

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
