#!/usr/bin/env python3
"""
Replace negative relations in a CSV file with positive ones.

Usage:
    python scripts/replace_relations.py --input /abs/path/file.csv --output /abs/path/out.csv
    python scripts/replace_relations.py --input /abs/path/file.csv --inplace
"""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional

RE_ACTIVE: List[str] = [
    "增进", "感到满意", "相信", "认为优秀", "欢迎", "认为有成就",
    "支持", "认可", "欣赏", "视作英雄"
]

RE_NEGATIVE: List[str] = [
    "担忧", "损害", "质疑", "感到不满", "认为非法", "认为恐怖",
    "威胁", "攻击", "认为缺乏", "批评", "认为有威胁", "认为有危机",
    "认为有暴力", "认为犯罪", "认为违规", "认为失败"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace all relations in RE_NEGATIVE with random relations from RE_ACTIVE."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Absolute path to the CSV file to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Absolute path to write the processed CSV. "
             "If omitted and --inplace is not set, defaults to <input>.replaced.csv.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Replace the input file in place (writes to a temporary file then swaps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed to make replacements deterministic.",
    )
    return parser.parse_args()


def replace_relations(input_path: Path, output_path: Path, seed: Optional[int]) -> int:
    rng = random.Random(seed)
    replaced = 0

    with input_path.open("r", encoding="utf-8", newline="") as fin, \
            output_path.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        for row in reader:
            if len(row) < 3:
                writer.writerow(row)
                continue

            relation = row[2].strip()
            # if relation in RE_NEGATIVE:
                # row[2] = rng.choice(RE_ACTIVE)
                # replaced += 1
            # 所有关系改为正面关系中的前5个
            row[2] = rng.choice(RE_ACTIVE)
            replaced += 1
            writer.writerow(row)

    return replaced


def main():
    args = parse_args()

    input_path: Path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.inplace:
        tmp_output = input_path.with_suffix(input_path.suffix + ".tmp")
        replaced = replace_relations(input_path, tmp_output, args.seed)
        tmp_output.replace(input_path)
        output_path = input_path
    else:
        output_path: Path = (
            args.output.expanduser().resolve()
            if args.output is not None else input_path.with_suffix(".replaced.csv")
        )
        replaced = replace_relations(input_path, output_path, args.seed)

    print(f"Processed file: {input_path}")
    print(f"Output file   : {output_path}")
    print(f"Replaced {replaced} relations from RE_NEGATIVE with entries from RE_ACTIVE.")


if __name__ == "__main__":
    main()

