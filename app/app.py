import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os
import re


# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="Career Intelligence System",
    page_icon="🎯",
    layout="wide"
)


# =====================================================
# STYLE
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    padding: 20px;
    border-radius: 15px;
    background: #161b22;
    border: 1px solid #30363d;
}

.title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #8b949e;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    conn = sqlite3.connect(
        "careermap.db"
    )

    jobs = pd.read_sql(
        "SELECT * FROM jobs",
        conn
    )

    conn.close()

    skills = pd.read_csv(
        "data/processed/skill_trends.csv"
    )

    role_demand = pd.read_csv(
        "data/processed/role_demand.csv"
    )

    company_demand = pd.read_csv(
        "data/processed/company_demand.csv"
    )

    location_demand = pd.read_csv(
        "data/processed/location_demand.csv"
    )

    return (
        jobs,
        skills,
        role_demand,
        company_demand,
        location_demand
    )


try:

    (
        jobs,
        skills,
        role_demand,
        company_demand,
        location_demand
    ) = load_data()

except Exception as e:

    st.error(
        "Data files are missing. "
        "Run the pipeline first."
    )

    st.exception(e)

    st.stop()


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="title">🎯 Career Intelligence System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    "Understand the job market. Discover skills. "
    "Plan your career."
    "</div>",
    unsafe_allow_html=True
)


st.divider()


# =====================================================
# KPI
# =====================================================

total_jobs = len(jobs)

companies = (
    jobs["company"]
    .replace("", pd.NA)
    .dropna()
    .nunique()
)

locations = (
    jobs["location"]
    .replace("", pd.NA)
    .dropna()
    .nunique()
)

salary_count = (
    jobs["salary_numeric"]
    .notna()
    .sum()
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "💼 Total Jobs",
        f"{total_jobs:,}"
    )


with c2:

    st.metric(
        "🏢 Companies",
        f"{companies:,}"
    )


with c3:

    st.metric(
        "📍 Locations",
        f"{locations:,}"
    )


with c4:

    st.metric(
        "💰 Salary Records",
        f"{salary_count:,}"
    )


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title(
    "🔎 Job Filters"
)


role_options = sorted(
    jobs["job_title"]
    .dropna()
    .unique()
)


role_options = [
    "All Roles"
] + role_options


selected_role = st.sidebar.selectbox(
    "Job Role",
    role_options
)


location_options = sorted(
    jobs["location"]
    .dropna()
    .unique()
)


location_options = [
    "All Locations"
] + location_options


selected_location = st.sidebar.selectbox(
    "Location",
    location_options
)


# =====================================================
# APPLY FILTERS
# =====================================================

filtered = jobs.copy()


if selected_role != "All Roles":

    filtered = filtered[
        filtered["job_title"]
        == selected_role
    ]


if selected_location != "All Locations":

    filtered = filtered[
        filtered["location"]
        == selected_location
    ]


st.subheader(
    "📊 Market Overview"
)


m1, m2 = st.columns(2)


with m1:

    st.metric(
        "Filtered Jobs",
        f"{len(filtered):,}"
    )


with m2:

    avg_salary = (
        filtered["salary_numeric"]
        .mean()
    )

    if pd.notna(avg_salary):

        st.metric(
            "Average Salary",
            f"₹{avg_salary:,.0f}"
        )

    else:

        st.metric(
            "Average Salary",
            "N/A"
        )


# =====================================================
# SKILL DEMAND
# =====================================================

st.divider()

st.subheader(
    "🔥 Most In-Demand Skills"
)


top_skills = (
    skills
    .sort_values(
        "job_count",
        ascending=False
    )
    .head(15)
)


st.bar_chart(
    top_skills.set_index(
        "skill"
    )["job_count"]
)


# =====================================================
# JOB ROLE DEMAND
# =====================================================

st.subheader(
    "💼 Most Demanded Job Roles"
)


top_roles = (
    role_demand
    .head(15)
)


st.bar_chart(
    top_roles.set_index(
        "job_title"
    )["job_count"]
)


# =====================================================
# LOCATION
# =====================================================

st.subheader(
    "📍 Job Market by Location"
)


top_locations = (
    location_demand
    .head(15)
)


st.bar_chart(
    top_locations.set_index(
        "location"
    )["job_count"]
)


# =====================================================
# CAREER RECOMMENDER
# =====================================================

st.divider()

st.subheader(
    "🎯 Career Skill Recommender"
)


target_role = st.text_input(
    "Enter your target role",
    placeholder="Example: Data Analyst"
)


if target_role:

    matching = jobs[
        jobs["job_title"]
        .str.contains(
            target_role,
            case=False,
            na=False
        )
    ]


    if len(matching) == 0:

        st.warning(
            "No matching roles found."
        )

    else:

        text = " ".join(
            matching[
                "skills_raw"
            ]
            .fillna("")
            .astype(str)
        ).lower()


        skill_list = [
            "python",
            "sql",
            "excel",
            "power bi",
            "tableau",
            "machine learning",
            "deep learning",
            "aws",
            "azure",
            "gcp",
            "tensorflow",
            "pytorch",
            "docker",
            "spark"
        ]


        recommendations = []


        for skill in skill_list:

            if re.search(
                r"\b" +
                re.escape(skill) +
                r"\b",
                text
            ):

                recommendations.append(
                    skill.title()
                )


        st.success(
            f"Found {len(matching):,} "
            "matching jobs."
        )


        if recommendations:

            st.write(
                "### Recommended Skills"
            )

            cols = st.columns(
                min(
                    len(recommendations),
                    4
                )
            )


            for i, skill in enumerate(
                recommendations
            ):

                cols[
                    i % len(cols)
                ].info(skill)

        else:

            st.info(
                "No skills detected for this role."
            )


# =====================================================
# SALARY PREDICTION
# =====================================================

st.divider()

st.subheader(
    "🤖 ML Salary Prediction"
)


MODEL_PATH = (
    "models/salary_model.pkl"
)


if os.path.exists(
    MODEL_PATH
):

    model = joblib.load(
        MODEL_PATH
    )


    prediction_role = st.text_input(
        "Role for salary prediction",
        placeholder="Example: Data Analyst"
    )


    prediction_location = st.text_input(
        "Location",
        placeholder="Example: Hyderabad"
    )


    prediction_experience = st.text_input(
        "Experience",
        placeholder="Example: 2 years"
    )


    if st.button(
        "Predict Salary"
    ):

        if prediction_role:

            prediction_data = pd.DataFrame({

                "job_title": [
                    prediction_role
                ],

                "location": [
                    prediction_location
                ],

                "experience": [
                    prediction_experience
                ],

                "combined_text": [
                    prediction_role
                    + " "
                    + prediction_location
                    + " "
                    + prediction_experience
                ]

            })


            prediction = model.predict(
                prediction_data
            )[0]


            st.success(
                f"Estimated Salary: "
                f"₹{prediction:,.0f}"
            )

else:

    st.warning(
        "Salary model not found. "
        "Run train_model.py first."
    )


# =====================================================
# JOB EXPLORER
# =====================================================

st.divider()

st.subheader(
    "🔎 Job Explorer"
)


search = st.text_input(
    "Search jobs"
)


display_df = filtered.copy()


if search:

    mask = (
        display_df
        .astype(str)
        .apply(
            lambda row:
            row.str.contains(
                search,
                case=False,
                na=False
            ).any(),
            axis=1
        )
    )

    display_df = display_df[
        mask
    ]


columns_to_show = [
    column
    for column in [
        "job_title",
        "company",
        "location",
        "experience",
        "salary_raw",
        "source"
    ]
    if column in display_df.columns
]


st.dataframe(
    display_df[
        columns_to_show
    ].head(200),
    use_container_width=True
)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Career Intelligence System | "
    "Built with Python, Pandas, SQLite, "
    "Scikit-learn and Streamlit"
)