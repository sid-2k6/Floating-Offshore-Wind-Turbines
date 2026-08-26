"""Download the FLOATBench tower-fatigue dataset from Hugging Face.

Dataset : https://huggingface.co/datasets/DeCoDELab/FLOATBench   (CC-BY-4.0)
Paper   : FLOATBench, arXiv:2605.25717
Code    : https://github.com/Joao97ribeiro/FLOATBench            (MIT)

Only `data.csv` per tower is needed: it holds all 194,040 rows without the
benchmark's own train/test split or regime labels, and is the canonical source
for reproducing any partition.

Downloads ~103 MB total into data/raw/floatbench/ (git-ignored).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fowt_rl.floatbench import default_raw_dir, download  # noqa: E402
from fowt_rl.turbine import TOWERS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=None, help="destination directory")
    parser.add_argument(
        "--towers",
        nargs="+",
        default=list(TOWERS),
        choices=list(TOWERS),
        help="tower variants to download",
    )
    args = parser.parse_args()

    destination = download(args.raw_dir or default_raw_dir(), towers=tuple(args.towers))
    print(f"\nFLOATBench ready in {destination}")


if __name__ == "__main__":
    main()
