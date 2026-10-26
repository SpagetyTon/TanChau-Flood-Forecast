"""Recompute the exposure-by-stage table (paper Table 'explev') and compare with the published values.

    python -m analysis.reproduce_table  (run from src/)
Exit status is non-zero if any value differs from the paper.
"""
import sys
from pathlib import Path

import pandas as pd

from .flood_model import FloodModel

ROOT = Path(__file__).resolve().parents[2]

# (stage m, label, people, % grid pop, land km2, communes with land, communes affected, people in cells > 1 m deep)
PAPER = [
    (2.50, "M2 threshold", 28842, 1.6, 885, 52, 28, 961),
    (3.17, "test-set peak (2024-10-05)", 110621, 6.3, 1721, 53, 44, 7305),
    (3.50, "Alarm level I", 165765, 9.5, 1991, 53, 50, 13766),
    (4.00, "Alarm level II", 298494, 17.0, 2437, 53, 53, 46943),
    (4.09, "peak 2018", 333506, 19.0, 2513, 53, 53, 56744),
    (4.50, "Alarm level III", 497283, 28.4, 2739, 53, 53, 131932),
    (4.70, "drill scenario", 609880, 34.8, 2822, 53, 53, 178818),
    (4.86, "peak 2011", 609880, 34.8, 2822, 53, 53, 216625),
    (5.06, "peak 2000", 609880, 34.8, 2822, 53, 53, 271951),
    (5.11, "peak 1961", 831918, 47.4, 2927, 53, 53, 331217),
    (5.50, "beyond all observed peaks", 1242046, 70.8, 3623, 53, 53, 711849),
]
COLS = ["level_m", "label", "people", "pct_grid_pop", "land_km2", "communes_with_land", "communes_affected", "people_gt_1m"]


def main(grids=ROOT / "data/derived/grids_public.npz", calib=ROOT / "data/mrc_derived/calib.json") -> int:
    m = FloodModel(grids, calib)
    rows, bad = [], 0
    for exp in PAPER:
        got = m.table_row(exp[0])
        rows.append({"label": exp[1], **got})
        for name, e in zip(COLS[2:], exp[2:]):
            if got[name] != e:
                bad += 1
                print(f"MISMATCH stage {exp[0]:.2f} m  {name}: got {got[name]}, paper {e}")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nAll 11 x 6 values match the paper." if bad == 0 else f"\n{bad} value(s) differ from the paper.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
