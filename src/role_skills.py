import pandas as pd
import re

INPUT = "data/processed/cleaned_jobs.csv"
OUTPUT = "data/processed/role_skills.csv"


SKILLS = [
    "Python",
    "SQL",
    "Excel",
    "Power BI",
    "Tableau",
    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "AWS",
    "Azure",
    "GCP",
    "TensorFlow",
    "PyTorch",
    "Java",
    "Spark",
    "Hadoop",
    "Docker",
    "Kubernetes",
    "Git"
]


df = pd.read_csv(INPUT)


df["search_text"] = (
    df["job_title"].fillna("").astype(str)
    + " "
    + df["skills_raw"].fillna("").astype(str)
    + " "
    + df["description"].fillna("").astype(str)
).str.lower()


results = []


for _, row in df.iterrows():

    role = row["job_title"]

    text = row["search_text"]

    for skill in SKILLS:

        if re.search(
            r"\b" +
            re.escape(skill.lower()) +
            r"\b",
            text
        ):

            results.append({
                "job_title": role,
                "skill": skill
            })


result_df = pd.DataFrame(results)


if len(result_df) > 0:

    result_df = (
        result_df
        .groupby(
            ["job_title", "skill"]
        )
        .size()
        .reset_index(
            name="demand"
        )
    )


result_df.to_csv(
    OUTPUT,
    index=False
)


print("✅ Role skill analysis completed")