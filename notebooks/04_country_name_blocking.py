import pandas as pd
import re
import unicodedata
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "dataset" / "train"


# ============================================================
# 2. NAME NORMALIZATION
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
    name = re.sub(r"[^\w\s]", " ", name, flags=re.UNICODE)

    # Remove extra spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name


# ============================================================
# 3. LOAD DATA
# ============================================================

print("Loading datasets...")

source1 = pd.read_csv(
    TRAIN / "train_source1.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    TRAIN / "train_source2.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    TRAIN / "train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    TRAIN / "train_ground_truth.tsv",
    sep="\t"
)

print("Datasets loaded.")


# ============================================================
# 4. NORMALIZE NAMES
# ============================================================

print("\nNormalizing names...")

source1["normalized_name"] = source1["business_name"].apply(
    normalize_name
)

source2["normalized_name"] = source2["business_name"].apply(
    normalize_name
)

source3["normalized_name"] = source3["business_name"].apply(
    normalize_name
)


# ============================================================
# 5. CREATE COUNTRY + NAME BLOCKING INDEX
# ============================================================

print("Creating country + name blocking indexes...")


# The key is:
#
#     (country, normalized_name)
#
# Example:
#
#     ("US", "prime money")
#
#     ("India", "prime money")
#
# are treated as DIFFERENT blocks.


s2_block_index = (
    source2[
        source2["normalized_name"] != ""
    ]
    .groupby(
        ["country", "normalized_name"]
    )["entity_id"]
    .apply(list)
    .to_dict()
)


s3_block_index = (
    source3[
        source3["normalized_name"] != ""
    ]
    .groupby(
        ["country", "normalized_name"]
    )["entity_id"]
    .apply(list)
    .to_dict()
)


# ============================================================
# 6. GENERATE CANDIDATES
# ============================================================

print("\nRunning country + name blocking experiment...")


total_s1 = len(source1)

s1_with_candidates = 0
s1_without_candidates = 0

total_candidates = 0

candidate_counts = []

candidate_lookup = {}


for _, row in source1.iterrows():

    s1_id = row["entity_id"]
    country = row["country"]
    name = row["normalized_name"]

    candidates = []

    block_key = (country, name)

    # Search Source 2
    if name != "" and block_key in s2_block_index:
        candidates.extend(
            s2_block_index[block_key]
        )

    # Search Source 3
    if name != "" and block_key in s3_block_index:
        candidates.extend(
            s3_block_index[block_key]
        )

    # Remove duplicate IDs
    candidates = list(set(candidates))

    candidate_lookup[s1_id] = set(candidates)

    candidate_count = len(candidates)

    candidate_counts.append(candidate_count)

    total_candidates += candidate_count

    if candidate_count > 0:
        s1_with_candidates += 1
    else:
        s1_without_candidates += 1


# ============================================================
# 7. BLOCKING STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("COUNTRY + NAME BLOCKING RESULTS")
print("=" * 60)


print(
    f"\nTotal Source 1 entities       : "
    f"{total_s1:,}"
)

print(
    f"S1 entities WITH candidates  : "
    f"{s1_with_candidates:,}"
)

print(
    f"S1 entities WITHOUT candidates: "
    f"{s1_without_candidates:,}"
)


coverage = (
    s1_with_candidates /
    total_s1
) * 100


print(
    f"\nEntity candidate coverage     : "
    f"{coverage:.2f}%"
)


print(
    f"\nTotal generated candidates    : "
    f"{total_candidates:,}"
)


average_candidates = (
    total_candidates /
    total_s1
)


print(
    f"Average candidates / S1       : "
    f"{average_candidates:.2f}"
)


# ============================================================
# 8. CANDIDATE DISTRIBUTION
# ============================================================

candidate_series = pd.Series(candidate_counts)


print("\n" + "=" * 60)
print("CANDIDATE COUNT DISTRIBUTION")
print("=" * 60)


print(
    f"\nMinimum candidates : "
    f"{candidate_series.min()}"
)

print(
    f"Maximum candidates : "
    f"{candidate_series.max()}"
)

print(
    f"Median candidates  : "
    f"{candidate_series.median():.2f}"
)

print(
    f"Mean candidates    : "
    f"{candidate_series.mean():.2f}"
)


print(
    f"\nS1 entities with exactly 1 candidate : "
    f"{(candidate_series == 1).sum():,}"
)


print(
    f"S1 entities with 2-10 candidates     : "
    f"{((candidate_series >= 2) & (candidate_series <= 10)).sum():,}"
)


print(
    f"S1 entities with >10 candidates      : "
    f"{(candidate_series > 10).sum():,}"
)


# ============================================================
# 9. GROUND-TRUTH RECALL
# ============================================================

print("\n" + "=" * 60)
print("GROUND-TRUTH RECALL")
print("=" * 60)


total_true_matches = 0
recovered_true_matches = 0

entities_with_all_matches_recovered = 0
entities_with_missing_matches = 0


for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]

    matched_ids_string = row["matched_entity_ids"]


    # Empty ground truth = no match
    if (
        pd.isna(matched_ids_string)
        or str(matched_ids_string).strip() == ""
    ):
        true_matches = set()

    else:
        true_matches = set(
            str(matched_ids_string).split(",")
        )


    candidates = candidate_lookup.get(
        s1_id,
        set()
    )


    recovered = (
        true_matches
        .intersection(candidates)
    )


    total_true_matches += len(
        true_matches
    )

    recovered_true_matches += len(
        recovered
    )


    # Did we preserve EVERY true match?
    if true_matches.issubset(candidates):

        entities_with_all_matches_recovered += 1

    else:

        entities_with_missing_matches += 1


# ============================================================
# 10. CALCULATE RECALL
# ============================================================

if total_true_matches > 0:

    match_recall = (
        recovered_true_matches /
        total_true_matches
    ) * 100

else:

    match_recall = 100.0


entity_recall = (
    entities_with_all_matches_recovered /
    len(ground_truth)
) * 100


# ============================================================
# 11. PRINT FINAL RESULTS
# ============================================================

print(
    f"\nTotal ground-truth matches : "
    f"{total_true_matches:,}"
)


print(
    f"Recovered by country + name block: "
    f"{recovered_true_matches:,}"
)


print(
    f"\nMATCH RECALL: "
    f"{match_recall:.2f}%"
)


print(
    f"\nS1 entities where ALL true matches "
    f"were recovered: "
    f"{entities_with_all_matches_recovered:,}"
)


print(
    f"S1 entities with missing true matches: "
    f"{entities_with_missing_matches:,}"
)


print(
    f"\nENTITY-LEVEL RECALL: "
    f"{entity_recall:.2f}%"
)


# ============================================================
# 12. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)

print("\nPrevious experiment:")
print("Exact normalized-name blocking")
print("Match Recall: 21.85%")

print("\nCurrent experiment:")
print("Country + exact normalized-name blocking")
print(
    f"Match Recall: {match_recall:.2f}%"
)

print(
    f"\nRecall change: "
    f"{match_recall - 21.85:+.2f} percentage points"
)


print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)