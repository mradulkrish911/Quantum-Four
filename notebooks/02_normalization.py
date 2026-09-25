import pandas as pd
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "dataset" / "train"


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


# Load all three sources
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


# Normalize names
source1["normalized_name"] = source1["business_name"].apply(
    normalize_name
)

source2["normalized_name"] = source2["business_name"].apply(
    normalize_name
)

source3["normalized_name"] = source3["business_name"].apply(
    normalize_name
)


print("\n========== SOURCE 1 ==========")
print(
    source1[
        ["business_name", "normalized_name"]
    ].head(5).to_string(index=False)
)


print("\n========== SOURCE 2 ==========")
print(
    source2[
        ["business_name", "normalized_name"]
    ].head(5).to_string(index=False)
)


print("\n========== SOURCE 3 ==========")
print(
    source3[
        ["business_name", "normalized_name"]
    ].head(5).to_string(index=False)
)


print("\n========== CHECKS ==========")

print("Source 1 rows:", len(source1))
print("Source 2 rows:", len(source2))
print("Source 3 rows:", len(source3))

print(
    "Source 1 empty normalized names:",
    (source1["normalized_name"] == "").sum()
)

print(
    "Source 2 empty normalized names:",
    (source2["normalized_name"] == "").sum()
)

print(
    "Source 3 empty normalized names:",
    (source3["normalized_name"] == "").sum()
)