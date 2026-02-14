import pandas as pd
import glob
import os


RAW_PATH = "data/raw/*.csv"
OUTPUT_PATH = "data/processed/data.csv"


def main():

    files = glob.glob(RAW_PATH)

    os.makedirs("data/processed", exist_ok=True)

    first = True

    for file in files:

        print("Processing:", file)

        name = os.path.basename(file)

        df = pd.read_csv(file, low_memory=False)

        df.columns = df.columns.str.strip()

        df["source_file"] = name

        df = df.replace([float("inf"), -float("inf")], pd.NA)
        df = df.dropna()

        df["Label"] = df["Label"].apply(
            lambda x: 0 if x == "BENIGN" else 1
        )

        if first:
            df.to_csv(OUTPUT_PATH, index=False)
            first = False
        else:
            df.to_csv(OUTPUT_PATH, mode="a", header=False, index=False)

    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
