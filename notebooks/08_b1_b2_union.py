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
# 3. ADDRESS NORMALIZATION
# ============================================================

def normalize_address(address):
    if pd.isna(address):
        return ""

    address = str(address)

    # Unicode normalization
    address = unicodedata.normalize("NFKC", address)

    # Lowercase
    address = address.lower()

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
        r"\bnumber\b": "no",
    }

    for pattern, replacement in replacements.items():
        address = re.sub(
            pattern,
            replacement,
            address
        )

    address = re.sub(
        r"[^\w\s]",
        " ",
        address,
        flags=re.UNICODE
    )

    address = re.sub(
        r"\s+",
        " ",
        address
    ).strip()

    return address

# ============================================================
# 4. LOAD DATA
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
# 5. NORMALIZE NAMES AND ADDRESSES
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


print("Normalizing addresses...")

source1["normalized_address"] = source1["business_address"].apply(
    normalize_address
)

source2["normalized_address"] = source2["business_address"].apply(
    normalize_address
)

source3["normalized_address"] = source3["business_address"].apply(
    normalize_address
)


print("Normalization complete.")

# ============================================================
# 6. CREATE B1 + B2 BLOCKING INDEXES
# ============================================================

print("\nCreating B1 + B2 blocking indexes...")


# ------------------------------------------------------------
# B1 — Country + exact normalized name
# ------------------------------------------------------------

s2_b1_index = (
    source2[
        source2["normalized_name"] != ""
    ]
    .groupby(
        ["country", "normalized_name"]
    )["entity_id"]
    .apply(list)
    .to_dict()
)


s3_b1_index = (
    source3[
        source3["normalized_name"] != ""
    ]
    .groupby(
        ["country", "normalized_name"]
    )["entity_id"]
    .apply(list)
    .to_dict()
)


# ------------------------------------------------------------
# B2 — Exact normalized address
# ------------------------------------------------------------

s2_b2_index = (
    source2[
        source2["normalized_address"] != ""
    ]
    .groupby(
        "normalized_address"
    )["entity_id"]
    .apply(list)
    .to_dict()
)


s3_b2_index = (
    source3[
        source3["normalized_address"] != ""
    ]
    .groupby(
        "normalized_address"
    )["entity_id"]
    .apply(list)
    .to_dict()
)


print("B1 + B2 blocking indexes created.")

# ============================================================
# 7. GENERATE B1 + B2 UNION CANDIDATES
# ============================================================

print("\nGenerating B1 + B2 union candidates...")


candidate_lookup = {}

total_candidates = 0
s1_with_candidates = 0
s1_without_candidates = 0

candidate_counts = []


for _, row in source1.iterrows():

    s1_id = row["entity_id"]

    country = row["country"]
    name = row["normalized_name"]
    address = row["normalized_address"]


    # --------------------------------------------------------
    # B1 — Country + exact normalized name
    # --------------------------------------------------------

    b1_candidates = set()

    if name != "":

        block_key = (country, name)

        if block_key in s2_b1_index:
            b1_candidates.update(
                s2_b1_index[block_key]
            )

        if block_key in s3_b1_index:
            b1_candidates.update(
                s3_b1_index[block_key]
            )


    # --------------------------------------------------------
    # B2 — Exact normalized address
    # --------------------------------------------------------

    b2_candidates = set()

    if address != "":

        if address in s2_b2_index:
            b2_candidates.update(
                s2_b2_index[address]
            )

        if address in s3_b2_index:
            b2_candidates.update(
                s3_b2_index[address]
            )


    # --------------------------------------------------------
    # UNION B1 + B2
    # --------------------------------------------------------

    candidates = b1_candidates | b2_candidates


    candidate_lookup[s1_id] = candidates


    candidate_count = len(candidates)

    candidate_counts.append(candidate_count)

    total_candidates += candidate_count


    if candidate_count > 0:

        s1_with_candidates += 1

    else:

        s1_without_candidates += 1


print("B1 + B2 union candidate generation complete.")

# ============================================================
# 8. B1 + B2 UNION STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("B1 + B2 UNION BLOCKING RESULTS")
print("=" * 60)


total_s1 = len(source1)


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
# 9. CANDIDATE COUNT DISTRIBUTION
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


    # Empty ground truth = singleton
    if (
        pd.isna(matched_ids_string)
        or str(matched_ids_string).strip() == ""
    ):
        true_matches = set()

    else:
        true_matches = set(
            str(matched_ids_string).split(",")
        )


    # Candidates generated by B1 + B2
    candidates = candidate_lookup.get(
        s1_id,
        set()
    )


    # True matches recovered by the union
    recovered = true_matches.intersection(
        candidates
    )


    total_true_matches += len(
        true_matches
    )

    recovered_true_matches += len(
        recovered
    )


    # Did we recover every true match for this S1?
    if true_matches.issubset(candidates):

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
    f"Recovered by B1 + B2 union : "
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
# 13. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("BLOCKING COMPARISON")
print("=" * 60)

print("\nB0 — Exact normalized name")
print("Match Recall     : 21.85%")
print("Avg Candidates   : 9.86")

print("\nB1 — Country + exact normalized name")
print("Match Recall     : 21.85%")
print("Avg Candidates   : 9.82")

print("\nB2 — Exact normalized address")
print("Match Recall     : 11.65%")
print("Avg Candidates   : 0.49")

print("\nB0 + B2 — UNION")
print("Match Recall     : 31.31%")
print("Avg Candidates   : 10.28")

print("\nB1 + B2 — UNION")
print(
    f"Match Recall     : "
    f"{match_recall:.2f}%"
)

print(
    f"Avg Candidates   : "
    f"{average_candidates:.2f}"
)

print(
    f"\nRecall improvement over B1: "
    f"{match_recall - 21.85:+.2f} percentage points"
)

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)

