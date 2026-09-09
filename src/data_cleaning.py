import pandas as pd
import numpy as np
import os
import re

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)


def read_csv_safe(path):

    for encoding in ["utf-8", "utf-8-sig", "latin1"]:

        try:
            return pd.read_csv(path, encoding=encoding)

        except UnicodeDecodeError:
            continue

    raise ValueError(f"Unable to read {path}")


def normalize_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )

    return df


def find_column(df, possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


def clean_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


# =========================================================
# KDS
# =========================================================
def process_kds():

    df = read_csv_safe(f"{RAW_DIR}/kds.csv")

    df = normalize_columns(df)

    print("\nKDS columns:")
    print(df.columns.tolist())

    title_col = find_column(
        df,
        ["job_title", "title", "jobtitle", "position"]
    )

    company_col = find_column(
        df,
        ["company", "company_name", "employer"]
    )

    location_col = find_column(
        df,
        ["location", "city", "job_location"]
    )

    salary_col = find_column(
        df,
        ["salary", "salary_estimate", "estimated_salary"]
    )

    skills_col = find_column(
        df,
        ["skills", "required_skills", "skill", "key_skills"]
    )

    description_col = find_column(
        df,
        ["description", "job_description"]
    )

    print(f"KDS -> skills_col matched: {skills_col}")

    result = pd.DataFrame()

    result["job_title"] = (
        df[title_col] if title_col else ""
    )

    result["company"] = (
        df[company_col] if company_col else ""
    )

    result["location"] = (
        df[location_col] if location_col else ""
    )

    result["salary_raw"] = (
        df[salary_col] if salary_col else ""
    )

    result["skills_raw"] = (
        df[skills_col] if skills_col else ""
    )

    result["description"] = (
        df[description_col] if description_col else ""
    )

    result["source"] = "KDS"

    return result

# =========================================================
# JOBA
# =========================================================

def process_joba():

    df = read_csv_safe(f"{RAW_DIR}/joba.csv")

    df = normalize_columns(df)

    print("\nJOBA columns:")
    print(df.columns.tolist())

    title_col = find_column(
        df,
        ["job_title", "title", "role", "position"]
    )

    company_col = find_column(
        df,
        ["company", "company_name", "employer"]
    )

    country_col = find_column(
        df,
        ["country", "location", "job_location"]
    )

    salary_col = find_column(
        df,
        ["salary", "salary_lpa", "salary_range"]
    )

    experience_col = find_column(
        df,
        ["experience", "experience_required", "years_experience"]
    )

    skills_col = find_column(
        df,
        ["skills", "required_skills", "skill"]
    )

    description_col = find_column(
        df,
        ["description", "job_description"]
    )

    result = pd.DataFrame()

    result["job_title"] = (
        df[title_col] if title_col else ""
    )

    result["company"] = (
        df[company_col] if company_col else ""
    )

    result["location"] = (
        df[country_col] if country_col else ""
    )

    result["salary_raw"] = (
        df[salary_col] if salary_col else ""
    )

    result["experience"] = (
        df[experience_col] if experience_col else ""
    )

    result["skills_raw"] = (
        df[skills_col] if skills_col else ""
    )

    result["description"] = (
        df[description_col] if description_col else ""
    )

    result["source"] = "JOBA"

    return result


# =========================================================
# INDIAN DATASET
# =========================================================

def process_indian():

    df = read_csv_safe(f"{RAW_DIR}/lindian.csv")

    df = normalize_columns(df)

    print("\nINDIAN columns:")
    print(df.columns.tolist())

    title_col = find_column(
        df,
        ["job_title", "title", "role", "position"]
    )

    company_col = find_column(
        df,
        ["company", "company_name", "employer"]
    )

    location_col = find_column(
        df,
        ["location", "city", "job_location"]
    )

    salary_col = find_column(
        df,
        ["salary", "salary_lpa", "salary_range"]
    )

    experience_col = find_column(
        df,
        ["experience", "experience_required", "years_experience"]
    )

    skills_col = find_column(
        df,
        ["skills", "required_skills", "skill"]
    )

    description_col = find_column(
        df,
        ["description", "job_description"]
    )

    result = pd.DataFrame()

    result["job_title"] = (
        df[title_col] if title_col else ""
    )

    result["company"] = (
        df[company_col] if company_col else ""
    )

    result["location"] = (
        df[location_col] if location_col else ""
    )

    result["salary_raw"] = (
        df[salary_col] if salary_col else ""
    )

    result["experience"] = (
        df[experience_col] if experience_col else ""
    )

    result["skills_raw"] = (
        df[skills_col] if skills_col else ""
    )

    result["description"] = (
        df[description_col] if description_col else ""
    )

    result["source"] = "INDIA"

    return result


# =========================================================
# MAIN PIPELINE
# =========================================================

print("\n🚀 Starting data cleaning...")


datasets = [
    process_kds(),
    process_joba(),
    process_indian()
]


master = pd.concat(
    datasets,
    ignore_index=True
)


# Clean text columns

text_columns = [
    "job_title",
    "company",
    "location",
    "salary_raw",
    "experience",
    "skills_raw",
    "description",
    "source"
]


for column in text_columns:

    if column in master.columns:

        master[column] = (
            master[column]
            .apply(clean_text)
        )


# Remove completely empty job titles

master = master[
    master["job_title"].str.len() > 0
]


# Remove duplicate records

master = master.drop_duplicates()


# =========================================================
# SALARY EXTRACTION
# =========================================================

def extract_salary(value):

    if not value:
        return np.nan

    value = str(value).lower()

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        value
    )

    if not numbers:
        return np.nan

    numbers = [
        float(x)
        for x in numbers
    ]

    # If salary is expressed as a range,
    # use the midpoint.

    salary = sum(numbers) / len(numbers)

    # Convert LPA to approximate annual INR

    if "lpa" in value:

        salary = salary * 100000

    elif "k" in value and salary < 10000:

        salary = salary * 1000

    return salary


master["salary_numeric"] = (
    master["salary_raw"]
    .apply(extract_salary)
)


# =========================================================
# SAVE
# =========================================================

output_file = (
    f"{PROCESSED_DIR}/cleaned_jobs.csv"
)

master.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 60)
print("✅ DATA CLEANING COMPLETED")
print("=" * 60)

print(f"Total jobs: {len(master)}")

print("\nSources:")

print(
    master["source"]
    .value_counts()
)

print("\nOutput:")
print(output_file)
