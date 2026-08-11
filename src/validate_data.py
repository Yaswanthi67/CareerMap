import pandas as pd
import os

FILES = [
    "data/raw/kds.csv",
    "data/raw/joba.csv",
    "data/raw/lindian.csv"
]


def load_csv(path):
    encodings = ["utf-8", "utf-8-sig", "latin1"]

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read {path}")


print("=" * 70)
print("CAREER INTELLIGENCE SYSTEM - DATA VALIDATION")
print("=" * 70)

for file in FILES:

    print("\n" + "=" * 70)
    print(f"FILE: {file}")
    print("=" * 70)

    if not os.path.exists(file):
        print("❌ FILE NOT FOUND")
        continue

    try:
        df = load_csv(file)

        print(f"Rows       : {df.shape[0]}")
        print(f"Columns    : {df.shape[1]}")

        print("\nColumns:")
        for column in df.columns:
            print(f"  - {column}")

        print("\nMissing values:")
        missing = df.isnull().sum()

        for column, count in missing.items():
            if count > 0:
                print(f"  {column}: {count}")

        print("\nDuplicate rows:")
        print(df.duplicated().sum())

        print("\nFirst 3 rows:")
        print(df.head(3).to_string())

        print("\nData types:")
        print(df.dtypes)

        print("\n✅ Validation completed")

    except Exception as e:
        print(f"❌ ERROR: {e}")

print("\n" + "=" * 70)
print("VALIDATION FINISHED")
print("=" * 70)