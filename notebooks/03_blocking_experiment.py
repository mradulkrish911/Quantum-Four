import pandas as pd
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "dataset" / "train"


# ============================================================
# 1. NAME NORMALIZATION
# ============================================================

def normalize_name(name):
    if pd.isna(name):
        return ""

    name = str(name)

    # Unicode normalization
    name = unicodedata.normalize("NFKC", name)

    # Lowercase
    name = name.lower()

    # Replace punctuation with space
    name = re.sub(r"[^\w\s]", " ", name, flags=re.UNICODE)

    # Remove extra spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name


# ============================================================
# 2. LOAD DATA
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
# 3. NORMALIZE BUSINESS NAMES
# ============================================================

print("\nNormalizing names...")

source1["normalized_name"] = source1["business_name"].apply(normalize_name)
source2["normalized_name"] = source2["business_name"].apply(normalize_name)
source3["normalized_name"] = source3["business_name"].apply(normalize_name)


# ============================================================
# 4. CREATE LOOKUP TABLES
# ============================================================

print("Creating blocking indexes...")

# S2:
# normalized_name -> list of entity IDs

s2_name_index = (
    source2[source2["normalized_name"] != ""]
    .groupby("normalized_name")["entity_id"]
    .apply(list)
    .to_dict()
)

# S3:
# normalized_name -> list of entity IDs

s3_name_index = (
    source3[source3["normalized_name"] != ""]
    .groupby("normalized_name")["entity_id"]
    .apply(list)
    .to_dict()
)


# ============================================================
# 5. TEST EXACT NORMALIZED-NAME BLOCKING
# ============================================================

print("\nRunning blocking experiment...")


total_s1 = len(source1)

s1_with_candidates = 0
s1_without_candidates = 0

total_candidates = 0

candidate_counts = []

for name in source1["normalized_name"]:

    candidates = []

    if name in s2_name_index:
        candidates.extend(s2_name_index[name])

    if name in s3_name_index:
        candidates.extend(s3_name_index[name])

    # Remove duplicates
    candidates = list(set(candidates))

    candidate_count = len(candidates)

    candidate_counts.append(candidate_count)

    total_candidates += candidate_count

    if candidate_count > 0:
        s1_with_candidates += 1
    else:
        s1_without_candidates += 1


# ============================================================
# 6. BASIC BLOCKING STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("BLOCKING RESULTS")
print("=" * 60)

print(f"\nTotal Source 1 entities       : {total_s1:,}")

print(
    f"S1 entities WITH candidates  : "
    f"{s1_with_candidates:,}"
)

print(
    f"S1 entities WITHOUT candidates: "
    f"{s1_without_candidates:,}"
)

coverage = (s1_with_candidates / total_s1) * 100

print(
    f"\nEntity candidate coverage     : "
    f"{coverage:.2f}%"
)

print(
    f"\nTotal generated candidates    : "
    f"{total_candidates:,}"
)

average_candidates = total_candidates / total_s1

print(
    f"Average candidates / S1       : "
    f"{average_candidates:.2f}"
)


# ============================================================
# 7. CANDIDATE COUNT DISTRIBUTION
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
# 8. CHECK GROUND-TRUTH RECALL
# ============================================================

print("\n" + "=" * 60)
print("GROUND-TRUTH RECALL")
print("=" * 60)


# Create dictionaries for quick country/name lookup

s2_country = dict(
    zip(source2["entity_id"], source2["country"])
)

s3_country = dict(
    zip(source3["entity_id"], source3["country"])
)


# Candidate lookup for every S1 entity

candidate_lookup = {}

for _, row in source1.iterrows():

    s1_id = row["entity_id"]
    name = row["normalized_name"]

    candidates = []

    if name in s2_name_index:
        candidates.extend(s2_name_index[name])

    if name in s3_name_index:
        candidates.extend(s3_name_index[name])

    candidate_lookup[s1_id] = set(candidates)


# Ground truth check

total_true_matches = 0
recovered_true_matches = 0

entities_with_all_matches_recovered = 0
entities_with_missing_matches = 0

for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]

    matched_ids_string = row["matched_entity_ids"]

    # Empty ground truth = singleton
    if pd.isna(matched_ids_string) or str(matched_ids_string).strip() == "":
        true_matches = set()
    else:
        true_matches = set(
            str(matched_ids_string).split(",")
        )

    candidates = candidate_lookup.get(s1_id, set())

    recovered = true_matches.intersection(candidates)

    total_true_matches += len(true_matches)
    recovered_true_matches += len(recovered)

    if true_matches.issubset(candidates):
        entities_with_all_matches_recovered += 1
    else:
        entities_with_missing_matches += 1


# ============================================================
# 9. PRINT RECALL
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


print(
    f"\nTotal ground-truth matches : "
    f"{total_true_matches:,}"
)

print(
    f"Recovered by exact-name block: "
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


print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)