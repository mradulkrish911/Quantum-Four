import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "dataset" / "train"

S1_PATH = TRAIN / "train_source1.tsv"
S2_PATH = TRAIN / "train_source2.tsv"
S3_PATH = TRAIN / "train_source3.tsv"
GT_PATH = TRAIN / "train_ground_truth.tsv"


# ============================================================
# PARAMETERS
# ============================================================

CHUNK_SIZE = 100_000
S1_BATCH_SIZE = 5_000
SOURCE_BATCH_SIZE = 50_000

TOP_K = 10

TFIDF_NGRAM_RANGE = (2, 5)
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 300_000


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def normalize_name(name):
    if pd.isna(name):
        return ""

    name = unicodedata.normalize("NFKC", str(name))
    name = name.lower()

    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


# ============================================================
# LEGAL SUFFIX NORMALIZATION
# ============================================================

SUFFIX_PATTERNS = [
    r"\bprivate limited\b$",
    r"\bpvt limited\b$",
    r"\bpvt ltd\b$",
    r"\bprivate ltd\b$",
    r"\blimited\b$",
    r"\bltd\b$",
    r"\bllc\b$",
    r"\bincorporated\b$",
    r"\binc\b$",
    r"\bcorporation\b$",
    r"\bcorp\b$",
    r"\bcompany\b$",
    r"\bco\b$",
]


def normalize_name_with_suffix(name):
    name = normalize_name(name)

    if not name:
        return ""

    changed = True

    while changed:
        changed = False

        for pattern in SUFFIX_PATTERNS:
            new_name = re.sub(pattern, "", name).strip()

            if new_name != name:
                name = new_name
                changed = True
                break

    name = re.sub(r"\s+", " ", name).strip()

    return name


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("Loading Source 1...")

source1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    usecols=["entity_id", "business_name", "country"],
    dtype={
        "entity_id": str,
        "business_name": str,
        "country": str,
    }
)

print(f"Source 1 loaded: {len(source1):,} rows")

print("Normalizing Source 1...")

source1["normalized_name"] = source1["business_name"].map(
    normalize_name_with_suffix
)

del source1["business_name"]


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

ground_truth = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str
)

print(f"Ground truth loaded: {len(ground_truth):,} rows")


def parse_matches(value):
    if pd.isna(value):
        return tuple()

    value = str(value).strip()

    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]

    if not value.strip():
        return tuple()

    return tuple(
        x.strip()
        for x in value.split(",")
        if x.strip()
    )


print("Creating ground-truth lookup...")

ground_truth_lookup = dict(
    zip(
        ground_truth["source1_entity_id"],
        ground_truth["matched_entity_ids"].map(parse_matches)
    )
)

del ground_truth


# ============================================================
# COUNTRIES
# ============================================================

countries = source1["country"].dropna().unique()

print(f"\nCountries: {list(countries)}")
print("\nStarting suffix-normalized TF-IDF matching...")


# ============================================================
# EXACT TOP-K FROM SPARSE MATRIX
# ============================================================

def exact_top_k_cosine(
    query_matrix,
    source_matrix,
    source_ids,
    k=10,
    source_batch_size=50_000
):
    """
    Exact top-k cosine similarity.

    TF-IDF vectors use the default L2 normalization,
    therefore:

        cosine_similarity(A, B) = A @ B.T

    We process the source matrix in chunks to avoid
    constructing the complete query x source similarity matrix.

    This does NOT change:
        - TF-IDF parameters
        - cosine metric
        - TOP_K
        - candidate ordering definition

    It only changes memory handling.
    """

    n_queries = query_matrix.shape[0]
    n_sources = source_matrix.shape[0]

    best_scores = np.full(
        (n_queries, k),
        -np.inf,
        dtype=np.float32
    )

    best_indices = np.full(
        (n_queries, k),
        -1,
        dtype=np.int64
    )

    for start in range(0, n_sources, source_batch_size):

        end = min(
            start + source_batch_size,
            n_sources
        )

        source_chunk = source_matrix[start:end]

        # Sparse matrix multiplication.
        #
        # Shape:
        # query batch x source chunk
        scores = query_matrix @ source_chunk.T

        scores = scores.toarray()

        chunk_size = scores.shape[1]

        local_k = min(k, chunk_size)

        # Find local top-k for every query.
        local_indices = np.argpartition(
            scores,
            -local_k,
            axis=1
        )[:, -local_k:]

        rows = np.arange(n_queries)[:, None]

        local_scores = scores[
            rows,
            local_indices
        ]

        local_indices = local_indices + start

        # Combine current best candidates with this chunk.
        combined_scores = np.concatenate(
            [best_scores, local_scores],
            axis=1
        )

        combined_indices = np.concatenate(
            [best_indices, local_indices],
            axis=1
        )

        new_indices = np.argpartition(
            combined_scores,
            -k,
            axis=1
        )[:, -k:]

        best_scores = combined_scores[
            rows,
            new_indices
        ]

        best_indices = combined_indices[
            rows,
            new_indices
        ]

        del scores
        del source_chunk
        del local_indices
        del local_scores
        del combined_scores
        del combined_indices
        del new_indices

    # Sort final top-k candidates by descending score.
    order = np.argsort(
        -best_scores,
        axis=1
    )

    rows = np.arange(n_queries)[:, None]

    best_scores = best_scores[
        rows,
        order
    ]

    best_indices = best_indices[
        rows,
        order
    ]

    return best_indices, best_scores


# ============================================================
# PROCESS COUNTRIES
# ============================================================

total_true_matches = 0
found_true_matches = 0


for country in countries:

    print("\n" + "=" * 70)
    print(f"Processing country: {country}")
    print("=" * 70)

    s1_country = source1[
        source1["country"] == country
    ].copy()

    print(
        f"Source 1 records: "
        f"{len(s1_country):,}"
    )

    # --------------------------------------------------------
    # LOAD SOURCE 2 FOR THIS COUNTRY
    # --------------------------------------------------------

    print("Loading Source 2...")

    s2_parts = []

    for chunk in pd.read_csv(
        S2_PATH,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "country"
        ],
        dtype={
            "entity_id": str,
            "business_name": str,
            "country": str,
        },
        chunksize=CHUNK_SIZE
    ):

        chunk = chunk[
            chunk["country"] == country
        ].copy()

        if len(chunk) == 0:
            continue

        chunk["normalized_name"] = chunk[
            "business_name"
        ].map(normalize_name_with_suffix)

        chunk = chunk[
            chunk["normalized_name"] != ""
        ]

        s2_parts.append(
            chunk[
                ["entity_id", "normalized_name"]
            ]
        )

    if s2_parts:
        source2_country = pd.concat(
            s2_parts,
            ignore_index=True
        )
    else:
        source2_country = pd.DataFrame(
            columns=["entity_id", "normalized_name"]
        )

    del s2_parts

    print(
        f"Source 2 records: "
        f"{len(source2_country):,}"
    )

    # --------------------------------------------------------
    # LOAD SOURCE 3 FOR THIS COUNTRY
    # --------------------------------------------------------

    print("Loading Source 3...")

    s3_parts = []

    for chunk in pd.read_csv(
        S3_PATH,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "country"
        ],
        dtype={
            "entity_id": str,
            "business_name": str,
            "country": str,
        },
        chunksize=CHUNK_SIZE
    ):

        chunk = chunk[
            chunk["country"] == country
        ].copy()

        if len(chunk) == 0:
            continue

        chunk["normalized_name"] = chunk[
            "business_name"
        ].map(normalize_name_with_suffix)

        chunk = chunk[
            chunk["normalized_name"] != ""
        ]

        s3_parts.append(
            chunk[
                ["entity_id", "normalized_name"]
            ]
        )

    if s3_parts:
        source3_country = pd.concat(
            s3_parts,
            ignore_index=True
        )
    else:
        source3_country = pd.DataFrame(
            columns=["entity_id", "normalized_name"]
        )

    del s3_parts

    print(
        f"Source 3 records: "
        f"{len(source3_country):,}"
    )

    # --------------------------------------------------------
    # COMBINE S2 + S3
    # --------------------------------------------------------

    source23_country = pd.concat(
        [
            source2_country,
            source3_country
        ],
        ignore_index=True
    )

    del source2_country
    del source3_country

    print(
        f"Total Source 2 + 3 records: "
        f"{len(source23_country):,}"
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print("Fitting TF-IDF...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True
    )

    source_vectors = vectorizer.fit_transform(
        source23_country["normalized_name"]
    )

    print(
        f"TF-IDF matrix created: "
        f"{source_vectors.shape}"
    )

    # --------------------------------------------------------
    # SOURCE IDs
    # --------------------------------------------------------

    source_ids = source23_country[
        "entity_id"
    ].to_numpy()

    # --------------------------------------------------------
    # PROCESS S1 IN BATCHES
    # --------------------------------------------------------

    print("\nStarting exact top-10 retrieval...")

    for start in range(
        0,
        len(s1_country),
        S1_BATCH_SIZE
    ):

        end = min(
            start + S1_BATCH_SIZE,
            len(s1_country)
        )

        s1_batch = s1_country.iloc[
            start:end
        ]

        query_vectors = vectorizer.transform(
            s1_batch["normalized_name"]
        )

        # ----------------------------------------------------
        # EXACT TOP-K
        # ----------------------------------------------------

        top_indices, top_scores = exact_top_k_cosine(
            query_vectors,
            source_vectors,
            source_ids,
            k=TOP_K,
            source_batch_size=SOURCE_BATCH_SIZE
        )

        # ----------------------------------------------------
        # EVALUATE AGAINST GROUND TRUTH
        # ----------------------------------------------------

        for row_number, s1_id in enumerate(
            s1_batch["entity_id"]
        ):

            true_ids = ground_truth_lookup.get(
                s1_id,
                tuple()
            )

            if not true_ids:
                continue

            total_true_matches += len(true_ids)

            predicted_ids = set()

            for index in top_indices[row_number]:

                if index == -1:
                    continue

                predicted_ids.add(
                    source_ids[index]
                )

            for true_id in true_ids:

                if true_id in predicted_ids:
                    found_true_matches += 1

        if (
            start == 0
            or (start + S1_BATCH_SIZE) % 50_000 == 0
            or end == len(s1_country)
        ):

            current_recall = (
                found_true_matches /
                total_true_matches
                if total_true_matches > 0
                else 0
            )

            print(
                f"Processed "
                f"{end:,}/{len(s1_country):,} "
                f"S1 records | "
                f"Current recall: "
                f"{current_recall:.4%}"
            )

        del query_vectors
        del top_indices
        del top_scores

    # --------------------------------------------------------
    # CLEAN COUNTRY MEMORY
    # --------------------------------------------------------

    del source23_country
    del source_vectors
    del source_ids
    del vectorizer
    del s1_country


# ============================================================
# FINAL RESULTS
# ============================================================

missed_true_matches = (
    total_true_matches -
    found_true_matches
)

recall = (
    found_true_matches /
    total_true_matches
    if total_true_matches > 0
    else 0
)


print("\n" + "=" * 60)
print("SUFFIX NORMALIZATION + TF-IDF")
print("=" * 60)

print(
    f"Total true matches     : "
    f"{total_true_matches:,}"
)

print(
    f"Found true matches     : "
    f"{found_true_matches:,}"
)

print(
    f"Missed true matches    : "
    f"{missed_true_matches:,}"
)

print(
    f"Recall                 : "
    f"{recall:.6f}"
)

print(
    f"Recall (%)             : "
    f"{recall * 100:.2f}%"
)

print("=" * 60)