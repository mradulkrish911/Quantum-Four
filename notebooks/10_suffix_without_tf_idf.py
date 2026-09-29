import pandas as pd
import re
import unicodedata
import gc
from pathlib import Path


# ============================================================
# 1. DATASET PATH
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "dataset" / "train"

S1_PATH = TRAIN / "train_source1.tsv"
S2_PATH = TRAIN / "train_source2.tsv"
S3_PATH = TRAIN / "train_source3.tsv"
GT_PATH = TRAIN / "train_ground_truth.tsv"


# ============================================================
# MEMORY SETTINGS
# ============================================================

# Smaller chunks = lower RAM spikes
CHUNK_SIZE = 50_000

# How often to print progress
PROGRESS_EVERY = 10


# ============================================================
# 2. BASIC NORMALIZATION
# ============================================================

def normalize_name(name):

    if pd.isna(name):
        return ""

    name = str(name)

    # Unicode normalization
    name = unicodedata.normalize("NFKC", name)

    # Lowercase
    name = name.lower()

    # Replace punctuation with spaces
    name = re.sub(
        r"[^\w\s]",
        " ",
        name,
        flags=re.UNICODE
    )

    # Remove extra spaces
    name = re.sub(
        r"\s+",
        " ",
        name
    ).strip()

    return name


# ============================================================
# 3. SUFFIX NORMALIZATION
# ============================================================

LEGAL_SUFFIX_PATTERNS = [
    r"\bprivate\s+limited$",
    r"\bpvt\s+limited$",
    r"\bpvt\s+ltd$",
    r"\bprivate\s+ltd$",
    r"\blimited$",
    r"\bltd$",
    r"\bllc$",
    r"\bincorporated$",
    r"\binc$",
    r"\bcorporation$",
    r"\bcorp$",
    r"\bcompany$",
    r"\bco$"
]


def normalize_name_with_suffix(name):

    name = normalize_name(name)

    if not name:
        return ""

    # IMPORTANT:
    # Keep the original sequential suffix-removal behavior.
    for pattern in LEGAL_SUFFIX_PATTERNS:

        name = re.sub(
            pattern,
            "",
            name
        ).strip()

    name = re.sub(
        r"\s+",
        " ",
        name
    ).strip()

    return name


# ============================================================
# 4. LOAD SOURCE 1
# ============================================================

print("=" * 70)
print("SUFFIX NORMALIZATION — WITHOUT TF-IDF")
print("=" * 70)

print("\nLoading Source 1...")

source1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name"
    ],
    dtype={
        "entity_id": "string",
        "business_name": "string"
    },
    engine="c"
)

print(
    f"Source 1 loaded: "
    f"{len(source1):,} rows"
)


# ============================================================
# 5. NORMALIZE SOURCE 1
# ============================================================

print("Normalizing Source 1...")

source1["normalized_name"] = (
    source1["business_name"]
    .map(normalize_name_with_suffix)
)

# We don't need business_name anymore
del source1["business_name"]

gc.collect()


# ============================================================
# 6. SOURCE 1 NAME MAP
# ============================================================

print("Creating Source 1 name map...")

source1_name_map = dict(
    zip(
        source1["entity_id"],
        source1["normalized_name"]
    )
)

print(
    f"Source 1 name map created: "
    f"{len(source1_name_map):,} entities"
)


# ============================================================
# 7. UNIQUE SOURCE 1 NAMES
# ============================================================

print("Creating Source 1 unique-name set...")

source1_names = set(
    source1["normalized_name"]
)

source1_names.discard("")

print(
    f"Unique Source 1 names: "
    f"{len(source1_names):,}"
)


# Source 1 dataframe is no longer needed.
# The two structures above contain everything required.
del source1

gc.collect()


# ============================================================
# 8. BUILD COMBINED SOURCE 2 + SOURCE 3 INDEX
# ============================================================

def add_source_to_index(filename, source_name, index):

    print(
        f"\nProcessing {source_name}..."
    )

    chunk_number = 0
    total_rows = 0
    matching_rows = 0

    for chunk in pd.read_csv(
        TRAIN / filename,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name"
        ],
        dtype={
            "entity_id": "string",
            "business_name": "string"
        },
        chunksize=CHUNK_SIZE,
        engine="c"
    ):

        chunk_number += 1
        total_rows += len(chunk)

        # Normalize names
        chunk["normalized_name"] = (
            chunk["business_name"]
            .map(normalize_name_with_suffix)
        )

        # Keep only names that occur in Source 1
        chunk = chunk[
            chunk["normalized_name"].isin(
                source1_names
            )
        ]

        matching_rows += len(chunk)

        # Build index
        for name, group in chunk.groupby(
            "normalized_name"
        ):

            if name not in index:
                index[name] = set()

            index[name].update(
                group["entity_id"].tolist()
            )

        # Free current chunk
        del chunk

        if chunk_number % PROGRESS_EVERY == 0:

            print(
                f"  Processed "
                f"{total_rows:,} rows | "
                f"matching rows: "
                f"{matching_rows:,} | "
                f"index names: "
                f"{len(index):,}"
            )

            gc.collect()

    print(
        f"{source_name} complete."
    )

    print(
        f"  Total rows processed: "
        f"{total_rows:,}"
    )

    print(
        f"  Matching rows kept: "
        f"{matching_rows:,}"
    )

    print(
        f"  Unique matching names: "
        f"{len(index):,}"
    )


print(
    "\nCreating combined Source 2 + Source 3 index..."
)

source23_index = {}


# ============================================================
# 9. BUILD SOURCE 2 INDEX
# ============================================================

add_source_to_index(
    "train_source2.tsv",
    "Source 2",
    source23_index
)

gc.collect()


# ============================================================
# 10. BUILD SOURCE 3 INDEX
# ============================================================

add_source_to_index(
    "train_source3.tsv",
    "Source 3",
    source23_index
)

gc.collect()


print(
    "\nCombined Source 2 + Source 3 index created."
)

print(
    f"Unique matching names: "
    f"{len(source23_index):,}"
)


# ============================================================
# 11. GROUND TRUTH PARSER
# ============================================================

def parse_matched_ids(value):

    if pd.isna(value):
        return set()

    value = str(value).strip()

    # Remove [ ]
    value = value.strip("[]")

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


# ============================================================
# 12. CALCULATE RECALL
# ============================================================

print("\nLoading ground truth...")

total_true_matches = 0
found_true_matches = 0

chunk_number = 0

for ground_truth_chunk in pd.read_csv(
    GT_PATH,
    sep="\t",
    usecols=[
        "source1_entity_id",
        "matched_entity_ids"
    ],
    dtype="string",
    chunksize=CHUNK_SIZE,
    engine="c"
):

    chunk_number += 1

    for _, row in ground_truth_chunk.iterrows():

        source1_id = row[
            "source1_entity_id"
        ]

        true_ids = parse_matched_ids(
            row["matched_entity_ids"]
        )

        if not true_ids:
            continue

        total_true_matches += len(
            true_ids
        )

        # Get Source 1 normalized name
        source1_name = source1_name_map.get(
            source1_id,
            ""
        )

        if not source1_name:
            continue

        # ====================================================
        # PREDICTED CANDIDATES
        # ====================================================

        predicted_ids = source23_index.get(
            source1_name,
            set()
        )

        # ====================================================
        # CORRECT MATCHES
        # ====================================================

        found_true_matches += len(
            true_ids.intersection(
                predicted_ids
            )
        )

    if chunk_number % PROGRESS_EVERY == 0:

        current_recall = (
            found_true_matches /
            total_true_matches
            if total_true_matches > 0
            else 0.0
        )

        print(
            f"  Processed ground truth: "
            f"{chunk_number * CHUNK_SIZE:,} rows | "
            f"Current recall: "
            f"{current_recall:.4%}"
        )

        gc.collect()


# ============================================================
# 13. FINAL RECALL
# ============================================================

if total_true_matches > 0:

    recall = (
        found_true_matches /
        total_true_matches
    )

else:

    recall = 0.0


# ============================================================
# 14. RESULT
# ============================================================

print("\n")

print("=" * 70)
print("SUFFIX NORMALIZATION — WITHOUT TF-IDF")
print("=" * 70)

print(
    f"Total true matches  : "
    f"{total_true_matches:,}"
)

print(
    f"Found true matches  : "
    f"{found_true_matches:,}"
)

print(
    f"Missed matches      : "
    f"{total_true_matches - found_true_matches:,}"
)

print(
    f"Recall              : "
    f"{recall:.6f}"
)

print(
    f"Recall (%)          : "
    f"{recall * 100:.4f}%"
)

print("=" * 70)