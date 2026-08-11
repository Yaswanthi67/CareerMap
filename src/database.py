import sqlite3
import pandas as pd
import os


DB_PATH = "careermap.db"

DATA_PATH = (
    "data/processed/cleaned_jobs.csv"
)


df = pd.read_csv(DATA_PATH)


conn = sqlite3.connect(DB_PATH)


df.to_sql(
    "jobs",
    conn,
    if_exists="replace",
    index=False
)


conn.execute("""
CREATE INDEX IF NOT EXISTS
idx_job_title
ON jobs(job_title)
""")


conn.execute("""
CREATE INDEX IF NOT EXISTS
idx_location
ON jobs(location)
""")


conn.execute("""
CREATE INDEX IF NOT EXISTS
idx_company
ON jobs(company)
""")


conn.commit()
conn.close()


print("✅ SQLite database created")

print(
    f"Database: {DB_PATH}"
)