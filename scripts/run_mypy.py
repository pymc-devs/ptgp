#!/usr/bin/env python
"""
Invoke mypy and compare the results with files in /ptgp.

Excludes tests and a list of files that are known to fail.

Exit code 0 indicates that there are no unexpected results.

Usage
-----
python scripts/run_mypy.py [--verbose]
"""

import argparse
import importlib
import io
import os
import subprocess
import sys

from pathlib import Path

import polars as pl

DP_ROOT = Path(__file__).absolute().parent.parent
FAILING = [
    line.strip()
    for line in (DP_ROOT / "scripts" / "mypy-failing.txt").read_text().splitlines()
    if line.strip()
]


def enforce_pep561(module_name):
    try:
        module = importlib.import_module(module_name)
        fp = Path(module.__path__[0], "py.typed")
        if not fp.exists():
            fp.touch()
    except ModuleNotFoundError:
        print(f"Can't enforce PEP 561 for {module_name} because it is not installed.")
    return


def mypy_to_polars(mypy_result: str) -> pl.DataFrame:
    """Reformat mypy JSON-lines output into a DataFrame.

    Adapted from: https://gist.github.com/michaelosthege/24d0703e5f37850c9e5679f69598930a
    """
    if not mypy_result.strip():
        return pl.DataFrame(
            schema={
                "file": pl.Utf8,
                "line": pl.Int64,
                "code": pl.Utf8,
                "severity": pl.Utf8,
                "message": pl.Utf8,
            }
        )
    return pl.read_ndjson(io.StringIO(mypy_result))


def check_no_unexpected_results(mypy_df: pl.DataFrame, show_expected: bool):
    """Compare mypy results with the list of known FAILING files.

    Exit the process with non-zero exit code upon unexpected results.
    """
    all_files = {
        str(fp).replace(str(DP_ROOT), "").strip(os.sep).replace(os.sep, "/")
        for fp in DP_ROOT.glob("ptgp/**/*.py")
        if "tests" not in str(fp)
    }
    if mypy_df.is_empty():
        failing: set[str] = set()
    else:
        failing = set(
            mypy_df.get_column("file").str.replace_all(os.sep, "/", literal=True).to_list()
        )
    if not failing.issubset(all_files):
        raise Exception(
            "Mypy should have ignored these files:\n"
            + "\n".join(sorted(map(str, failing - all_files)))
        )
    passing = all_files - failing
    expected_failing = set(FAILING)
    unexpected_failing = failing - expected_failing
    unexpected_passing = passing.intersection(expected_failing)

    if not unexpected_failing:
        print(f"{len(passing)}/{len(all_files)} files pass as expected.")
    else:
        print("!!!!!!!!!")
        print(f"{len(unexpected_failing)} files unexpectedly failed:")
        print("\n".join(sorted(map(str, unexpected_failing))))

        if show_expected:
            print(
                "\nThese files did not fail before, so please check the above output"
                f" for errors in {unexpected_failing} and fix them."
            )
        else:
            print("\nThese files did not fail before. Fix all errors reported in the output above.")
            print(
                f"\nNote: In addition to these errors, {len(failing.intersection(expected_failing))} errors in files "
                f'marked as "expected failures" were also found. To see these failures, run: '
                f"`python scripts/run_mypy.py --show-expected`"
            )

        print("You can run `python scripts/run_mypy.py` to reproduce this test locally.")

        sys.exit(1)

    if unexpected_passing:
        print("!!!!!!!!!")
        print(f"{len(unexpected_passing)} files unexpectedly passed the type checks:")
        print("\n".join(sorted(map(str, unexpected_passing))))
        print("This is good news! Remove them from scripts/mypy-failing.txt.")
        if all_files.issubset(passing):
            print("WOW! All files are passing the mypy type checks!")
            print("scripts/run_mypy.py may no longer be needed.")
        print("!!!!!!!!!")
        sys.exit(1)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mypy type checks on PTGP codebase.")
    parser.add_argument(
        "--verbose", action="count", default=1, help="Pass this to print mypy output."
    )
    parser.add_argument(
        "--show-expected",
        action="store_true",
        help="Also show expected failures in verbose output.",
    )
    parser.add_argument(
        "--groupby",
        default="file",
        help="How to group verbose output. One of {file|errorcode|message}.",
    )
    args, _ = parser.parse_known_args()

    cp = subprocess.run(
        [
            "mypy",
            "--output",
            "json",
            "--disable-error-code",
            "annotation-unchecked",
            "ptgp",
        ],
        capture_output=True,
    )

    output = cp.stdout.decode("utf-8")
    df = mypy_to_polars(output)

    if args.verbose:
        if not args.show_expected and not df.is_empty():
            filtered_df = df.filter(~pl.col("file").is_in(FAILING))
        else:
            filtered_df = df

        if not filtered_df.is_empty():
            for (section,), sdf in filtered_df.group_by(args.groupby, maintain_order=True):
                print(f"\n\n[{section}]")
                for row in sdf.iter_rows(named=True):
                    print(
                        f"{row['file']}:{row['line']}: {row['code']} [{row['severity']}]: {row['message']}"
                    )
        print()
    else:
        print(
            "Mypy output hidden."
            " Run `python run_mypy.py --verbose` to see the full output,"
            " or `python run_mypy.py --help` for other options."
        )

    check_no_unexpected_results(df, show_expected=args.show_expected)

    sys.exit(0)
