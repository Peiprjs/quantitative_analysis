#!/usr/bin/env python3
"""Batch convert tab-separated .xls files to CSV.

Usage
-----
    python xls_to_csv.py                     # convert all .xls under ./data
    python xls_to_csv.py --data-dir ./data   # explicit directory
    python xls_to_csv.py --overwrite         # re-convert even if .csv exists
    python xls_to_csv.py --dry-run           # preview without writing files
"""

import argparse
import csv
import glob
import os
from pathlib import Path

try:
    from functions import convert_xls_to_csv
except ModuleNotFoundError:
    def convert_xls_to_csv(xls_path: Path, csv_path: Path) -> None:
        """Read a tab-separated .xls file (ISO-8859-1) and write a UTF-8 CSV."""
        with xls_path.open(encoding="iso-8859-1", newline="") as infile:
            reader = csv.reader(infile, delimiter="\t")
            with csv_path.open("w", encoding="utf-8", newline="") as outfile:
                writer = csv.writer(outfile)
                for row in reader:
                    writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch convert tab-separated .xls files to CSV."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Root directory to search for .xls files (default: ./data)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert files even when a .csv already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    args = parser.parse_args()

    xls_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.xls"), recursive=True))

    if not xls_files:
        print(f"No .xls files found under '{args.data_dir}'")
        return

    converted = skipped = errors = 0

    for xls_path in xls_files:
        csv_path = os.path.splitext(xls_path)[0] + ".csv"
        rel_xls = os.path.relpath(xls_path)
        rel_csv = os.path.relpath(csv_path)

        if os.path.exists(csv_path) and not args.overwrite:
            print(f"  SKIP   {rel_xls}  (CSV exists; use --overwrite to re-convert)")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  DRY    {rel_xls} -> {rel_csv}")
            converted += 1
            continue

        try:
            convert_xls_to_csv(Path(xls_path), Path(csv_path))
            print(f"  OK     {rel_xls} -> {rel_csv}")
            converted += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR  {rel_xls}: {exc}")
            errors += 1

    prefix = "Would convert" if args.dry_run else "Converted"
    print(f"\nDone. {prefix} {converted}, skipped {skipped}, errors {errors}.")


if __name__ == "__main__":
    main()
