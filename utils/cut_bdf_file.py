#!/usr/bin/env python
"""Crop a raw .bdf file to a time range (one-off utility).

Usage
-----
    python utils/cut_bdf_file.py --subject V06 --tmin 10 --tmax 20 --out data/cropped/V06_data.bdf
"""

import argparse
from pathlib import Path

import mne


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input .bdf file")
    parser.add_argument("--tmin", type=float, required=True, help="Crop start (s)")
    parser.add_argument("--tmax", type=float, required=True, help="Crop end (s)")
    parser.add_argument("--out", required=True, help="Output .bdf file")
    args = parser.parse_args()

    raw = mne.io.read_raw_bdf(args.input, preload=True)
    raw.crop(tmin=args.tmin, tmax=args.tmax)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(out), raw, fmt="bdf", overwrite=True)
    print(f"Cropped {args.input} ({args.tmin}-{args.tmax} s) -> {out}")


if __name__ == "__main__":
    main()