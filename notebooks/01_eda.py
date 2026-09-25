from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "train"

source1_file = DATASET / "train_source1.tsv"

print("Looking for:", source1_file)
print("File exists:", source1_file.exists())

source1 = pd.read_csv(
    source1_file,
    sep="\t"
)

print("\nFirst 5 rows:")
print(source1.head())

print("\nShape:")
print(source1.shape)

print("\nColumns:")
print(source1.columns.tolist())

source2 = pd.read_csv(
    ROOT / "dataset" / "train" / "train_source2.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    ROOT / "dataset" / "train" / "train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    ROOT / "dataset" / "train" / "train_ground_truth.tsv",
    sep="\t"
)

print("\nSource 2 shape:", source2.shape)
print("Source 3 shape:", source3.shape)
print("Ground truth shape:", ground_truth.shape)

print("\nSource 2 columns:", source2.columns.tolist())
print("Source 3 columns:", source3.columns.tolist())
print("Ground truth columns:", ground_truth.columns.tolist())


print("\n========== MISSING VALUES ==========")

print("\nSource 1:")
print(source1.isnull().sum())

print("\nSource 2:")
print(source2.isnull().sum())

print("\nSource 3:")
print(source3.isnull().sum())


print("\n========== SAMPLE DATA ==========")

print("\nSource 1 samples:")
print(source1[["business_name", "business_address", "country"]].head(10).to_string(index=False))

print("\nSource 2 samples:")
print(source2[["business_name", "business_address", "country"]].head(10).to_string(index=False))

print("\nSource 3 samples:")
print(source3[["business_name", "business_address", "country"]].head(10).to_string(index=False))


print("\n========== GROUND TRUTH SAMPLES ==========")

print(ground_truth.head(10).to_string(index=False))

print("\n========== COUNTRY DISTRIBUTION ==========")

print("\nSource 1:")
print(source1["country"].value_counts())

print("\nSource 2:")
print(source2["country"].value_counts())

print("\nSource 3:")
print(source3["country"].value_counts())

print("\n========== COUNTRY CONSISTENCY CHECK ==========")

# Take a sample of ground-truth records
sample_gt = ground_truth.sample(
    n=10000,
    random_state=42
)

same_country = 0
different_country = 0
checked_matches = 0

# Create quick lookups only for the sampled IDs
sample_ids = set()

for ids in sample_gt["matched_entity_ids"]:
    if pd.isna(ids):
        continue

    for entity_id in str(ids).split(","):
        sample_ids.add(entity_id)

# Find only the required records in Source 2 and Source 3
s2_sample = source2[source2["entity_id"].isin(sample_ids)]
s3_sample = source3[source3["entity_id"].isin(sample_ids)]

country_lookup = {}

for _, row in s2_sample.iterrows():
    country_lookup[row["entity_id"]] = row["country"]

for _, row in s3_sample.iterrows():
    country_lookup[row["entity_id"]] = row["country"]


# Check each Source 1 business
source1_lookup = source1.set_index("entity_id")["country"]

for _, row in sample_gt.iterrows():

    s1_id = row["source1_entity_id"]
    s1_country = source1_lookup.get(s1_id)

    if pd.isna(row["matched_entity_ids"]):
        continue

    for matched_id in str(row["matched_entity_ids"]).split(","):

        if matched_id not in country_lookup:
            continue

        checked_matches += 1

        if country_lookup[matched_id] == s1_country:
            same_country += 1
        else:
            different_country += 1


print("Ground truth matches checked:", checked_matches)
print("Same country:", same_country)
print("Different country:", different_country)

if checked_matches > 0:
    print(
        "Same-country percentage:",
        round((same_country / checked_matches) * 100, 2),
        "%"
    )