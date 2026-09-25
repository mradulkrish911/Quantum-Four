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
# 2. ADDRESS NORMALIZATION
# ============================================================

def normalize_address(address):

    if pd.isna(address):
        return ""

    address = str(address)

    # Unicode normalization
    address = unicodedata.normalize("NFKC", address)

    # Lowercase
    address = address.lower()

    # Common address abbreviations
    replacements = {
        r"\bstreet\b": "st",
        r"\broad\b": "rd",
        r"\bavenue\b": "ave",
        r"\bdrive\b": "dr",
        r"\blane\b": "ln",
        r"\bhighway\b": "hwy",
        r"\bparkway\b": "pkwy",
        r"\bplace\b": "pl",
        r"\bcourt\b": "ct",
        r"\bboulevard\b": "blvd",
        r"\bapartment\b": "apt",
        r"\bsuite\b": "ste",
        r"\bbuilding\b": "bldg",
        r"\broad\b": "rd",
        r"\bnumber\b": "no",
    }

    for pattern, replacement in replacements.items():
        address = re.sub(
            pattern,
            replacement,
            address
        )

    # Remove punctuation
    address = re.sub(
        r"[^\w\s]",
        " ",
        address,
        flags=re.UNICODE
    )

    # Normalize whitespace
    address = re.sub(
        r"\s+",
        " ",
        address
    ).strip()

    return address


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
# 4. NORMALIZE ADDRESSES
# ============================================================

print("\nNormalizing addresses...")

source1["normalized_address"] = (
    source1["business_address"]
    .apply(normalize_address)
)

source2["normalized_address"] = (
    source2["business_address"]
    .apply(normalize_address)
)

source3["normalized_address"] = (
    source3["business_address"]
    .apply(normalize_address)
)


# ============================================================
# 5. BASIC NORMALIZATION CHECK
# ============================================================

print("\n" + "=" * 60)
print("ADDRESS NORMALIZATION SAMPLE")
print("=" * 60)

print(
    source1[
        [
            "business_address",
            "normalized_address"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 6. CREATE ADDRESS INDEX
# ============================================================

print("\nCreating address blocking indexes...")


s2_address_index = (
    source2[
        source2["normalized_address"] != ""
    ]
    .groupby(
        "normalized_address"
    )["entity_id"]
    .apply(list)
    .to_dict()
)


s3_address_index = (
    source3[
        source3["normalized_address"] != ""
    ]
    .groupby(
        "normalized_address"
    )["entity_id"]
    .apply(list)
    .to_dict()
)


# ============================================================
# 7. GENERATE ADDRESS CANDIDATES
# ============================================================

print("\nRunning address blocking experiment...")


total_s1 = len(source1)

s1_with_candidates = 0
s1_without_candidates = 0

total_candidates = 0

candidate_counts = []

candidate_lookup = {}


for _, row in source1.iterrows():

    s1_id = row["entity_id"]

    address = row["normalized_address"]

    candidates = []

    if address != "":

        if address in s2_address_index:

            candidates.extend(
                s2_address_index[address]
            )

        if address in s3_address_index:

            candidates.extend(
                s3_address_index[address]
            )

    # Remove duplicates
    candidates = list(set(candidates))

    candidate_lookup[s1_id] = set(
        candidates
    )

    candidate_count = len(candidates)

    candidate_counts.append(
        candidate_count
    )

    total_candidates += candidate_count

    if candidate_count > 0:

        s1_with_candidates += 1

    else:

        s1_without_candidates += 1


# ============================================================
# 8. BLOCKING STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("ADDRESS BLOCKING RESULTS")
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
# 9. CANDIDATE DISTRIBUTION
# ============================================================

candidate_series = pd.Series(
    candidate_counts
)


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
# 10. GROUND-TRUTH RECALL
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


    # Singleton / no-match case
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


    if true_matches.issubset(
        candidates
    ):

        entities_with_all_matches_recovered += 1

    else:

        entities_with_missing_matches += 1


# ============================================================
# 11. CALCULATE RECALL
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
# 12. PRINT RECALL
# ============================================================

print(
    f"\nTotal ground-truth matches : "
    f"{total_true_matches:,}"
)


print(
    f"Recovered by address block : "
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
# 13. COMPARE WITH PREVIOUS BLOCKS
# ============================================================

print("\n" + "=" * 60)
print("BLOCKING COMPARISON")
print("=" * 60)

print("\nB0 — Exact normalized name")
print("Match Recall     : 21.85%")
print("Avg Candidates   : 9.86")

print("\nB1 — Country + exact name")
print("Match Recall     : 21.85%")
print("Avg Candidates   : 9.82")

print("\nB2 — Normalized address")
print(
    f"Match Recall     : "
    f"{match_recall:.2f}%"
)

print(
    f"Avg Candidates   : "
    f"{average_candidates:.2f}"
)


print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)