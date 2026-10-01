"""Re-split the dataset so no source image appears in more than one split.

The Roboflow export contains augmented copies of the same scan (same name before
'_jpg.rf.<hash>') spread across train/valid/test. This script pools all images and
re-splits them by source id (class-stratified), so test metrics are not inflated.

    python src/make_group_split.py            # writes data/Tumour_grouped/
Then set DATA_DIR in src/config.py to data/Tumour_grouped.
"""
import re
import shutil
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from config import DATA_DIR, ROOT, SEED

OUT = ROOT / "data" / "Tumour_grouped"


def main():
    files = sorted(p for p in DATA_DIR.glob("*/*/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    labels = np.array([p.parent.name for p in files])
    groups = np.array([re.sub(r"_jpg\.rf\..*", "", p.name) for p in files])
    # 10 folds: 1 -> test (~10%), 2 -> valid (~20%), 7 -> train (~70%)
    fold = np.zeros(len(files), dtype=int)
    for k, (_, idx) in enumerate(StratifiedGroupKFold(10, shuffle=True, random_state=SEED)
                                 .split(files, labels, groups)):
        fold[idx] = k
    split_of = lambda k: "test" if k == 0 else "valid" if k in (1, 2) else "train"
    if OUT.exists():
        shutil.rmtree(OUT)
    for p, lab, k in zip(files, labels, fold):
        dst = OUT / split_of(k) / lab
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / p.name)
    # sanity check: no group in two splits
    seen = {}
    for g, k in zip(groups, fold):
        seen.setdefault(g, set()).add(split_of(k))
    assert all(len(v) == 1 for v in seen.values()), "leakage remains!"
    print("Written to", OUT, "| images per split:", dict(Counter(split_of(k) for k in fold)))


if __name__ == "__main__":
    main()
