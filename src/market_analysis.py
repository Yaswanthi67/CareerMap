import pandas as pd
import os

INPUT = "data/processed/cleaned_jobs.csv"

OUTPUT_DIR = "data/processed"

df = pd.read_csv(INPUT)


# =====================================================
# ROLE DEMAND
# =====================================================

role_demand = (
    df["job_title"]
    .value_counts()
    .reset_index()
)

role_demand.columns = [
    "job_title",
    "job_count"
]

role_demand.to_csv(
    f"{OUTPUT_DIR}/role_demand.csv",
    index=False
)


# =====================================================
# COMPANY DEMAND
# =====================================================

company_demand = (
    df["company"]
    .replace("", pd.NA)
    .dropna()
    .value_counts()
    .reset_index()
)

company_demand.columns = [
    "company",
    "job_count"
]

company_demand.to_csv(
    f"{OUTPUT_DIR}/company_demand.csv",
    index=False
)


# =====================================================
# LOCATION DEMAND
# =====================================================

location_demand = (
    df["location"]
    .replace("", pd.NA)
    .dropna()
    .value_counts()
    .reset_index()
)

location_demand.columns = [
    "location",
    "job_count"
]

location_demand.to_csv(
    f"{OUTPUT_DIR}/location_demand.csv",
    index=False
)


print("✅ Market analysis completed")