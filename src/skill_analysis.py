import pandas as pd
import re
import os

INPUT = "data/processed/cleaned_jobs.csv"
OUTPUT = "data/processed/skill_trends.csv"


SKILLS = {

    "Python": [
        r"\bpython\b"
    ],

    "SQL": [
        r"\bsql\b"
    ],

    "Excel": [
        r"\bexcel\b",
        r"microsoft excel"
    ],

    "Power BI": [
        r"power\s*bi",
        r"powerbi"
    ],

    "Tableau": [
        r"\btableau\b"
    ],

    "Machine Learning": [
        r"machine learning",
        r"\bml\b"
    ],

    "Deep Learning": [
        r"deep learning"
    ],

    "Artificial Intelligence": [
        r"artificial intelligence",
        r"\bai\b"
    ],

    "AWS": [
        r"\baws\b",
        r"amazon web services"
    ],

    "Azure": [
        r"\bazure\b",
        r"microsoft azure"
    ],

    "GCP": [
        r"\bgcp\b",
        r"google cloud"
    ],

    "TensorFlow": [
        r"tensorflow"
    ],

    "PyTorch": [
        r"pytorch"
    ],

    "Java": [
        r"\bjava\b"
    ],

    "C++": [
        r"c\+\+"
    ],

    "R": [
        r"\br\b"
    ],

    "Spark": [
        r"\bspark\b",
        r"apache spark"
    ],

    "Hadoop": [
        r"hadoop"
    ],

    "Docker": [
        r"docker"
    ],

    "Kubernetes": [
        r"kubernetes"
    ],

    "Git": [
        r"\bgit\b",
        r"github"
    ]
}


df = pd.read_csv(INPUT)


# Combine all useful text

df["search_text"] = (
    df["job_title"].fillna("").astype(str)
    + " "
    + df["skills_raw"].fillna("").astype(str)
    + " "
    + df["description"].fillna("").astype(str)
).str.lower()


results = []


for skill, patterns in SKILLS.items():

    count = 0

    for text in df["search_text"]:

        for pattern in patterns:

            if re.search(pattern, text):

                count += 1

                break

    results.append({
        "skill": skill,
        "job_count": count,
        "demand_percentage":
            round((count / len(df)) * 100, 2)
            if len(df) > 0 else 0
    })


skill_df = pd.DataFrame(results)


skill_df = skill_df.sort_values(
    "job_count",
    ascending=False
)


os.makedirs(
    "data/processed",
    exist_ok=True
)


skill_df.to_csv(
    OUTPUT,
    index=False
)


print("✅ Skill analysis completed")

print(skill_df.head(10))