from pathlib import Path
import numpy as np
import pandas as pd

#Setup
RAW = Path("data/raw")
OUT = Path("data.processed")
OUT.mkdir(parents=True, exist_ok=True)

#find all CSV files in the raw data directory and read them into a single DataFrame
def preprocess_cicids():
    files = sorted((RAW / "cicids2017").glob("*.csv"))
    df = pd.concat(
        (pd.read_csv(f, encoding="latin-1", low_memory=False) for f in files),  # encoding is needed for some files with odd characters
        ignore_index=True,
    )
    df.columns = df.columns.str.strip()  # removes whitespace from column names
    print("RAW shape:", df.shape)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()  # clean the data by removing rows with NaN or infinite values and duplicates

    # Fix odd characters in web attack labels
    df["Label"] = df["Label"].str.replace(r"[^\x20-\x7E]", "-", regex=True).str.strip()  # replaces non-ASCII characters with hyphens
    df["binary_label"] = (df["Label"] != "BENIGN").astype("int8")  # != gives true or False for each row, astype converts to int8 (1 for True, 0 for False)

    feats = df.columns.difference(["Label", "binary_label"])  # picks all columns except for "Label" and "binary_label" to be features
    df[feats] = df[feats].astype("float32")  # shrinks the data type of the features to float32 to save memory

    print("Clean shape:", df.shape)
    print(df["Label"].value_counts())
    df.to_parquet(OUT / "cicids2017_clean.parquet")
    return df


# loads the UNSW-NB15 dataset, combines the training and testing sets, encodes categorical features, and saves the processed data to parquet files
def preprocess_unsw():
    d = RAW / "unsw_nb15"
    train = pd.read_parquet(d / "UNSW_NB15_training-set.parquet")
    test = pd.read_parquet(d / "UNSW_NB15_testing-set.parquet")
    train["is_train"], test["is_train"] = 1, 0
    df = pd.concat([train, test], ignore_index=True)
    df = df.drop(columns=["id"], errors="ignore").dropna()

    # Encode categories on combined data so codes match across splits
    for col in ["proto", "service", "state"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    print("Shape:", df.shape)
    print(df["label"].value_counts())
    df[df.is_train == 1].drop(columns="is_train").to_parquet(OUT / "unsw_train.parquet")
    df[df.is_train == 0].drop(columns="is_train").to_parquet(OUT / "unsw_test.parquet")
    return df


if __name__ == "__main__":
    preprocess_cicids()
    preprocess_unsw()