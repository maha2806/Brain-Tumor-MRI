"""Check source-image overlap across dataset splits.

The Roboflow export can contain augmented copies of the same source scan.
This script identifies source IDs shared between train, validation and test
splits so potential data leakage can be inspected.

The source ID is derived from the filename before '_jpg.rf.<hash>'.
"""

import re
from collections import defaultdict
from pathlib import Path

from Config import CLASS_NAMES, SPLITS, ROOT


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def source_id(filename: str) -> str:
    """Extract the source-image ID from a Roboflow filename."""
    return re.sub(r"_jpg\.rf\..*", "", filename)


def collect_source_ids(data_dir: Path) -> dict[str, set[str]]:
    """Collect source IDs present in each dataset split."""

    split_sources = {}

    for split in SPLITS:
        split_path = data_dir / split
        sources = set()

        for class_name in CLASS_NAMES:
            class_path = split_path / class_name

            if not class_path.exists():
                continue

            for image_path in class_path.rglob("*"):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    sources.add(source_id(image_path.name))

        split_sources[split] = sources

    return split_sources


def find_cross_split_overlap(
    split_sources: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Find source IDs appearing in more than one dataset split."""

    source_to_splits = defaultdict(set)

    for split, sources in split_sources.items():
        for source in sources:
            source_to_splits[source].add(split)

    return {
        source: splits
        for source, splits in source_to_splits.items()
        if len(splits) > 1
    }


def main():
    data_dir = ROOT / "Tumour"

    split_sources = collect_source_ids(data_dir)
    overlaps = find_cross_split_overlap(split_sources)

    print("Source-image overlap check")
    print("=" * 50)

    for split, sources in split_sources.items():
        print(f"{split}: {len(sources)} unique source IDs")

    print()

    if not overlaps:
        print("No source-image overlap detected across splits.")
        return

    print(
        f"Potential source-image leakage detected: "
        f"{len(overlaps)} source IDs occur in multiple splits."
    )

    print("\nExamples:")
    for source, splits in list(overlaps.items())[:20]:
        print(f"  {source}: {', '.join(sorted(splits))}")


if __name__ == "__main__":
    main()
