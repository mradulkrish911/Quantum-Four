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
# 2. NORMALIZATION
# ============================================================

def normalize_name(name):
    if pd.isna(name):
        return ""

    name = str(name)
    name = unicodedata.normalize("NFKC", name)
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name, flags=re.UNICODE)
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

print("Normalizing names...")

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
# 5. CREATE ENTITY LOOKUPS
# ============================================================

print("Creating entity lookups...")

s1_lookup = source1.set_index("entity_id").to_dict("index")
s2_lookup = source2.set_index("entity_id").to_dict("index")
s3_lookup = source3.set_index("entity_id").to_dict("index")


# ============================================================
# 6. CREATE EXACT NAME INDEX
# ============================================================

print("Creating exact-name index...")

s2_name_index = (
    source2[source2["normalized_name"] != ""]
    .groupby("normalized_name")["entity_id"]
    .apply(list)
    .to_dict()
)

s3_name_index = (
    source3[source3["normalized_name"] != ""]
    .groupby("normalized_name")["entity_id"]
    .apply(list)
    .to_dict()
)


# ============================================================
# 7. FIND MISSED TRUE MATCHES
# ============================================================

print("Finding missed true matches...")

missed_matches = []

for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]

    true_ids_string = row["matched_entity_ids"]

    if (
        pd.isna(true_ids_string)
        or str(true_ids_string).strip() == ""
    ):
        continue

    true_ids = set(
        str(true_ids_string).split(",")
    )

    s1_row = s1_lookup[s1_id]

    normalized_s1_name = s1_row["normalized_name"]

    # Candidates from exact normalized name
    candidates = set()

    if normalized_s1_name in s2_name_index:
        candidates.update(
            s2_name_index[normalized_s1_name]
        )

    if normalized_s1_name in s3_name_index:
        candidates.update(
            s3_name_index[normalized_s1_name]
        )

    # Find true matches that blocking missed
    missed = true_ids - candidates

    for matched_id in missed:

        if matched_id.startswith("S2-"):
            matched_row = s2_lookup.get(matched_id)

        elif matched_id.startswith("S3-"):
            matched_row = s3_lookup.get(matched_id)

        else:
            matched_row = None

        if matched_row is None:
            continue

        missed_matches.append({
            "source1_entity_id": s1_id,

            "s1_name": s1_row["business_name"],
            "s1_normalized_name": s1_row["normalized_name"],
            "s1_address": s1_row["business_address"],
            "s1_country": s1_row["country"],

            "matched_entity_id": matched_id,

            "matched_name": matched_row["business_name"],
            "matched_normalized_name":
                matched_row["normalized_name"],
            "matched_address":
                matched_row["business_address"],
            "matched_country":
                matched_row["country"],
        })


# ============================================================
# 8. CREATE DATAFRAME
# ============================================================

missed_df = pd.DataFrame(missed_matches)


print("\n" + "=" * 60)
print("MISSED MATCH ANALYSIS")
print("=" * 60)

print(
    f"\nTotal missed true matches: "
    f"{len(missed_df):,}"
)


# ============================================================
# 9. SHOW SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE MISSED MATCHES")
print("=" * 60)


sample_size = min(30, len(missed_df))

sample = missed_df.sample(
    n=sample_size,
    random_state=42
)


for i, (_, row) in enumerate(sample.iterrows(), 1):

    print("\n" + "-" * 70)

    print(f"MISSED MATCH #{i}")

    print(
        f"S1 ID      : "
        f"{row['source1_entity_id']}"
    )

    print(
        f"S1 Name    : "
        f"{row['s1_name']}"
    )

    print(
        f"S1 Address : "
        f"{row['s1_address']}"
    )

    print(
        f"S1 Country : "
        f"{row['s1_country']}"
    )

    print(
        f"\nTRUE MATCH : "
        f"{row['matched_entity_id']}"
    )

    print(
        f"Match Name : "
        f"{row['matched_name']}"
    )

    print(
        f"Match Addr : "
        f"{row['matched_address']}"
    )

    print(
        f"Match Country: "
        f"{row['matched_country']}"
    )

    print(
        f"\nNormalized S1   : "
        f"{row['s1_normalized_name']}"
    )

    print(
        f"Normalized Match: "
        f"{row['matched_normalized_name']}"
    )


# ============================================================
# 10. SAVE SAMPLE FOR INSPECTION
# ============================================================

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

sample.to_csv(
    OUTPUT_DIR / "missed_matches_sample.tsv",
    sep="\t",
    index=False
)


print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)

print(
    "\nSample saved to:"
)

print(
    "output/missed_matches_sample.tsv"
)


print("\nAnalysis complete.")