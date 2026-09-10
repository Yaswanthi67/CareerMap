import os
import re
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Career Map",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-card {
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #30363d;
        background-color: #161b22;
        margin-bottom: 10px;
    }

    .skill-card {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #30363d;
        background-color: #161b22;
        text-align: center;
        margin-bottom: 10px;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# FILE PATHS
# ============================================================

JOBS_FILE = os.path.join(
    DATA_DIR,
    "cleaned_jobs.csv"
)

SKILLS_FILE = os.path.join(
    DATA_DIR,
    "skill_trends.csv"
)

ROLE_SKILLS_FILE = os.path.join(
    DATA_DIR,
    "role_skills.csv"
)

ROLE_DEMAND_FILE = os.path.join(
    DATA_DIR,
    "role_demand.csv"
)

COMPANY_DEMAND_FILE = os.path.join(
    DATA_DIR,
    "company_demand.csv"
)

LOCATION_DEMAND_FILE = os.path.join(
    DATA_DIR,
    "location_demand.csv"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "salary_model.pkl"
)


# ============================================================
# SAFE CSV LOADER
# ============================================================

def load_csv(path):

    if not os.path.exists(path):
        return None

    try:

        df = pd.read_csv(
            path,
            encoding="utf-8"
        )

        return df

    except UnicodeDecodeError:

        try:

            return pd.read_csv(
                path,
                encoding="latin1"
            )

        except Exception:
            return None

    except Exception:
        return None


# ============================================================
# LOAD ALL DATA
# ============================================================

@st.cache_data
def load_all_data():

    jobs = load_csv(
        JOBS_FILE
    )

    skills = load_csv(
        SKILLS_FILE
    )

    role_skills = load_csv(
        ROLE_SKILLS_FILE
    )

    role_demand = load_csv(
        ROLE_DEMAND_FILE
    )

    company_demand = load_csv(
        COMPANY_DEMAND_FILE
    )

    location_demand = load_csv(
        LOCATION_DEMAND_FILE
    )

    return (
        jobs,
        skills,
        role_skills,
        role_demand,
        company_demand,
        location_demand
    )


(
    jobs,
    skills,
    role_skills,
    role_demand,
    company_demand,
    location_demand
) = load_all_data()


# ============================================================
# CHECK MAIN DATASET
# ============================================================

if jobs is None:

    st.error(
        "❌ cleaned_jobs.csv was not found."
    )

    st.info(
        "Run the following command from the project root:"
    )

    st.code(
        "python src/data_cleaning.py",
        language="bash"
    )

    st.write(
        "Expected file:"
    )

    st.code(
        "data/processed/cleaned_jobs.csv"
    )

    st.stop()


# ============================================================
# STANDARDIZE AVAILABLE COLUMNS
# ============================================================

jobs.columns = (
    jobs.columns
    .astype(str)
    .str.strip()
    .str.lower()
)


# Add missing columns safely

required_columns = [
    "job_title",
    "company",
    "location",
    "salary_raw",
    "salary_numeric",
    "experience",
    "skills_raw",
    "description",
    "source"
]


for column in required_columns:

    if column not in jobs.columns:

        jobs[column] = ""


# ============================================================
# CLEAN BASIC VALUES
# ============================================================

for column in [
    "job_title",
    "company",
    "location",
    "salary_raw",
    "experience",
    "skills_raw",
    "description",
    "source"
]:

    jobs[column] = (
        jobs[column]
        .fillna("")
        .astype(str)
    )


jobs["salary_numeric"] = pd.to_numeric(
    jobs["salary_numeric"],
    errors="coerce"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎯 Career Map'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore job-market demand, discover essential skills, '
    'analyze career opportunities and estimate salary.'
    '</div>',
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🎯 Career Map"
)

st.sidebar.caption(
    "Data-driven career market analysis"
)


page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🔥 Skill Intelligence",
        "💼 Job Market",
        "🎯 Career Skill Recommender",
        "🤖 Salary Predictor",
        "🔎 Job Explorer"
    ]
)


st.sidebar.divider()

st.sidebar.info(
    "Data source: "
    "DS Jobs + Job Market+ Indian Job datasets"
)


# ============================================================
# COMMON METRICS
# ============================================================

total_jobs = len(jobs)

total_companies = (
    jobs["company"]
    .replace("", pd.NA)
    .dropna()
    .nunique()
)

total_locations = (
    jobs["location"]
    .replace("", pd.NA)
    .dropna()
    .nunique()
)

salary_records = (
    jobs["salary_numeric"]
    .notna()
    .sum()
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">'
        '📊 Market Overview'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "💼 Total Jobs",
            f"{total_jobs:,}"
        )


    with col2:

        st.metric(
            "🏢 Companies",
            f"{total_companies:,}"
        )


    with col3:

        st.metric(
            "📍 Locations",
            f"{total_locations:,}"
        )


    with col4:

        st.metric(
            "💰 Salary Records",
            f"{salary_records:,}"
        )


    st.divider()


    # --------------------------------------------------------
    # TOP ROLES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "💼 Top Job Roles"
        )

        if role_demand is not None and len(role_demand) > 0:

            role_demand.columns = (
                role_demand.columns
                .str.lower()
            )

            if (
                "job_title" in role_demand.columns
                and
                "job_count" in role_demand.columns
            ):

                chart_data = (
                    role_demand
                    .sort_values(
                        "job_count",
                        ascending=False
                    )
                    .head(10)
                )

                fig = px.bar(
                    chart_data,
                    x="job_count",
                    y="job_title",
                    orientation="h",
                    title="Most Demanded Roles"
                )

                fig.update_layout(
                    template="plotly_dark",
                    yaxis={
                        "categoryorder": "total ascending"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        else:

            st.info(
                "role_demand.csv not available."
            )


    with col2:

        st.subheader(
            "🔥 Top Skills"
        )

        if skills is not None and len(skills) > 0:

            skills.columns = (
                skills.columns
                .str.lower()
            )

            if (
                "skill" in skills.columns
                and
                "job_count" in skills.columns
            ):

                chart_data = (
                    skills
                    .sort_values(
                        "job_count",
                        ascending=False
                    )
                    .head(10)
                )

                fig = px.bar(
                    chart_data,
                    x="job_count",
                    y="skill",
                    orientation="h",
                    title="Most In-Demand Skills"
                )

                fig.update_layout(
                    template="plotly_dark",
                    yaxis={
                        "categoryorder": "total ascending"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        else:

            st.info(
                "skill_trends.csv not available."
            )


    # --------------------------------------------------------
    # DATA SOURCE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📚 Dataset Distribution"
    )


    source_counts = (
        jobs["source"]
        .replace("", "Unknown")
        .value_counts()
        .reset_index()
    )

    source_counts.columns = [
        "source",
        "count"
    ]


    fig = px.pie(
        source_counts,
        names="source",
        values="count",
        title="Jobs by Dataset Source"
    )

    fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PAGE 2 — SKILL INTELLIGENCE
# ============================================================

elif page == "🔥 Skill Intelligence":

    st.title(
        "🔥 Skill Intelligence"
    )

    st.write(
        "Understand which technical skills appear most frequently "
        "across the job market."
    )


    if skills is None or len(skills) == 0:

        st.warning(
            "skill_trends.csv is missing."
        )

        st.stop()


    skills.columns = (
        skills.columns
        .str.lower()
    )


    if "skill" not in skills.columns:

        st.error(
            "The skill_trends.csv file does not contain a "
            "'skill' column."
        )

        st.stop()


    if "job_count" not in skills.columns:

        st.error(
            "The skill_trends.csv file does not contain "
            "'job_count'."
        )

        st.stop()


    skills = skills.sort_values(
        "job_count",
        ascending=False
    )


    top_n = st.slider(
        "Number of skills",
        5,
        min(30, len(skills)),
        15
    )


    chart_data = skills.head(
        top_n
    )


    fig = px.bar(
        chart_data,
        x="job_count",
        y="skill",
        orientation="h",
        title="Skill Demand"
    )

    fig.update_layout(
        template="plotly_dark",
        yaxis={
            "categoryorder": "total ascending"
        }
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader(
        "📋 Skill Demand Table"
    )


    display_columns = [
        column
        for column in [
            "skill",
            "job_count",
            "demand_percentage"
        ]
        if column in skills.columns
    ]


    st.dataframe(
        skills[
            display_columns
        ].head(top_n),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3 — JOB MARKET
# ============================================================

elif page == "💼 Job Market":

    st.title(
        "💼 Job Market Intelligence"
    )


    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    st.subheader(
        "📍 Job Opportunities by Location"
    )


    if (
        location_demand is not None
        and
        len(location_demand) > 0
    ):

        location_demand.columns = (
            location_demand.columns
            .str.lower()
        )


        if (
            "location" in location_demand.columns
            and
            "job_count" in location_demand.columns
        ):

            data = (
                location_demand
                .sort_values(
                    "job_count",
                    ascending=False
                )
                .head(15)
            )


            fig = px.bar(
                data,
                x="job_count",
                y="location",
                orientation="h",
                title="Top Job Locations"
            )


            fig.update_layout(
                template="plotly_dark",
                yaxis={
                    "categoryorder":
                    "total ascending"
                }
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # COMPANIES
    # --------------------------------------------------------

    st.subheader(
        "🏢 Companies with Most Jobs"
    )


    if (
        company_demand is not None
        and
        len(company_demand) > 0
    ):

        company_demand.columns = (
            company_demand.columns
            .str.lower()
        )


        if (
            "company" in company_demand.columns
            and
            "job_count" in company_demand.columns
        ):

            data = (
                company_demand
                .sort_values(
                    "job_count",
                    ascending=False
                )
                .head(15)
            )


            fig = px.bar(
                data,
                x="job_count",
                y="company",
                orientation="h",
                title="Top Hiring Companies"
            )


            fig.update_layout(
                template="plotly_dark",
                yaxis={
                    "categoryorder":
                    "total ascending"
                }
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # SALARY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "💰 Salary Distribution"
    )


    salary_data = jobs[
        jobs["salary_numeric"].notna()
    ]


    if len(salary_data) > 0:

        fig = px.histogram(
            salary_data,
            x="salary_numeric",
            nbins=30,
            title="Salary Distribution"
        )


        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Annual Salary (INR)",
            yaxis_title="Number of Jobs"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No numeric salary data available."
        )

# ============================================================
# PAGE 4 — CAREER RECOMMENDER
# ============================================================

elif page == "🎯 Career Skill Recommender":

    st.title(
        "🎯 Career Skill Recommender"
    )

    st.write(
        "Enter a target career and discover the skills "
        "frequently associated with that role."
    )

    # --------------------------------------------------------
    # ROLE INPUT
    # --------------------------------------------------------

    target_role = st.text_input(
        "Target Job Role",
        placeholder="Example: Java Developer"
    )

    if target_role:

        # ----------------------------------------------------
        # ROLE-SPECIFIC SEARCH TERMS
        # ----------------------------------------------------

        role_lower = target_role.lower().strip()

        role_aliases = {

            "java developer": [
                "java developer",
                "java",
                "spring boot developer",
                "spring developer",
                "java backend developer",
                "java software developer",
                "java engineer",
                "backend java"
            ],

            "python developer": [
                "python developer",
                "python",
                "django developer",
                "flask developer",
                "fastapi developer"
            ],

            "data analyst": [
                "data analyst",
                "business analyst",
                "reporting analyst",
                "data analytics"
            ],

            "data engineer": [
                "data engineer",
                "big data engineer",
                "etl developer",
                "data platform engineer"
            ],

            "full stack developer": [
                "full stack developer",
                "full-stack developer",
                "full stack",
                "web developer"
            ],

            "backend developer": [
                "backend developer",
                "back-end developer",
                "backend engineer",
                "server-side developer"
            ],

            "frontend developer": [
                "frontend developer",
                "front-end developer",
                "frontend engineer",
                "web developer"
            ],

            "ai/ml engineer": [
                "ai/ml engineer",
                "ai engineer",
                "ml engineer",
                "machine learning engineer",
                "artificial intelligence engineer"
            ],

            "devops engineer": [
                "devops engineer",
                "devops",
                "cloud engineer",
                "site reliability engineer"
            ]
        }

        # ----------------------------------------------------
        # FIND BEST ROLE ALIASES
        # ----------------------------------------------------

        search_terms = [target_role]

        for role_name, aliases in role_aliases.items():

            if role_lower == role_name or role_lower in aliases:

                search_terms = aliases
                break

        # ----------------------------------------------------
        # MATCH JOB TITLES
        # ----------------------------------------------------

        matching_masks = []

        for term in search_terms:

            if term.strip():

                matching_masks.append(
                    jobs["job_title"].str.contains(
                        re.escape(term),
                        case=False,
                        na=False
                    )
                )

        if matching_masks:

            role_mask = matching_masks[0]

            for mask in matching_masks[1:]:
                role_mask = role_mask | mask

            matching = jobs[role_mask].copy()

        else:

            matching = jobs[
                jobs["job_title"].str.contains(
                    re.escape(target_role),
                    case=False,
                    na=False
                )
            ].copy()

        # ----------------------------------------------------
        # NO JOBS FOUND
        # ----------------------------------------------------

        if len(matching) == 0:

            st.warning(
                "No matching jobs found. "
                "Try a broader role such as "
                "'Java', 'Python', 'Data Analyst', "
                "'Backend Developer', etc."
            )

        else:

            st.success(
                f"{len(matching):,} matching job records found."
            )

            # ------------------------------------------------
            # COMBINE TEXT FOR SKILL DETECTION
            # ------------------------------------------------

            combined_text_series = (
                matching["skills_raw"].fillna("").astype(str)
                + " "
                + matching["description"].fillna("").astype(str)
                + " "
                + matching["job_title"].fillna("").astype(str)
            ).str.lower()

            total_matches = len(matching)

            jobs_with_text = (
                combined_text_series.str.strip().str.len() > 0
            ).sum()

            # ------------------------------------------------
            # SKILL CATEGORIES
            # ------------------------------------------------

            skill_categories = {

                "Languages": {

                    "Java": r"\bjava\b",

                    "Python": r"\bpython\b",

                    "JavaScript": (
                        r"\bjavascript\b|\bjs\b"
                    ),

                    "TypeScript": (
                        r"\btypescript\b"
                    ),

                    "C++": r"c\+\+",

                    "C#": r"c#",

                    "PHP": r"\bphp\b"
                },

                "Frameworks": {

                    "Spring Boot": (
                        r"spring\s*boot|\bspring\b"
                    ),

                    "React": (
                        r"\breact\b|reactjs"
                    ),

                    "Node.js": (
                        r"node\.?js"
                    ),

                    "Angular": (
                        r"\bangular\b"
                    ),

                    "Vue.js": (
                        r"vue\.?js|\bvuejs\b"
                    ),

                    "Express.js": (
                        r"express\.?js|\bexpressjs\b"
                    ),

                    "Next.js": (
                        r"next\.?js|\bnextjs\b"
                    ),

                    "Laravel": (
                        r"\blaravel\b"
                    ),

                    "HTML": (
                        r"\bhtml\b"
                    ),

                    "CSS": (
                        r"\bcss\b"
                    ),

                    "REST API": (
                        r"rest\s*api|restful"
                    ),

                    "GraphQL": (
                        r"\bgraphql\b"
                    ),

                    "Redux": (
                        r"\bredux\b"
                    ),

                    "Bootstrap": (
                        r"\bbootstrap\b"
                    ),

                    "Microservices": (
                        r"microservices?"
                    )
                },

                "Data & Databases": {

                    "SQL": (
                        r"\bsql\b"
                    ),

                    "MySQL": (
                        r"\bmysql\b"
                    ),

                    "PostgreSQL": (
                        r"\bpostgresql\b|\bpostgres\b"
                    ),

                    "MongoDB": (
                        r"\bmongodb\b|\bmongo\b"
                    ),

                    "Excel": (
                        r"\bexcel\b"
                    ),

                    "Power BI": (
                        r"power\s*bi|powerbi"
                    ),

                    "Tableau": (
                        r"\btableau\b"
                    ),

                    "Looker": (
                        r"\blooker\b"
                    ),

                    "VBA": (
                        r"\bvba\b"
                    ),

                    "Google Analytics": (
                        r"google analytics"
                    )
                },

                "AI / ML": {

                    "Machine Learning": (
                        r"machine learning"
                    ),

                    "Deep Learning": (
                        r"deep learning"
                    ),

                    "Artificial Intelligence": (
                        r"artificial intelligence"
                    ),

                    "TensorFlow": (
                        r"tensorflow"
                    ),

                    "PyTorch": (
                        r"pytorch"
                    ),

                    "Keras": (
                        r"\bkeras\b"
                    ),

                    "Scikit-learn": (
                        r"scikit[-\s]?learn|\bsklearn\b"
                    ),

                    "NLP": (
                        r"\bnlp\b|natural language processing"
                    ),

                    "Computer Vision": (
                        r"computer vision"
                    )
                },

                "Statistics & Analytics": {

                    "Statistics": (
                        r"\bstatistics\b|\bstatistical\b"
                    ),

                    "Data Visualization": (
                        r"data visuali[sz]ation"
                    ),

                    "A/B Testing": (
                        r"a/b\s*testing|ab\s*testing"
                    ),

                    "Pandas": (
                        r"\bpandas\b"
                    ),

                    "NumPy": (
                        r"\bnumpy\b"
                    ),

                    "SAS": (
                        r"\bsas\b"
                    ),

                    "Big Data": (
                        r"big data"
                    ),

                    "Hadoop": (
                        r"\bhadoop\b"
                    ),

                    "Jupyter": (
                        r"\bjupyter\b"
                    )
                },

                "Cloud & DevOps": {

                    "AWS": (
                        r"\baws\b"
                    ),

                    "Azure": (
                        r"\bazure\b"
                    ),

                    "GCP": (
                        r"\bgcp\b|google cloud"
                    ),

                    "Docker": (
                        r"docker"
                    ),

                    "Kubernetes": (
                        r"kubernetes"
                    ),

                    "Jenkins": (
                        r"\bjenkins\b"
                    ),

                    "CI/CD": (
                        r"ci\s*/\s*cd|continuous integration|continuous deployment"
                    ),

                    "Terraform": (
                        r"\bterraform\b"
                    ),

                    "Ansible": (
                        r"\bansible\b"
                    ),

                    "Linux": (
                        r"\blinux\b|\bunix\b"
                    ),

                    "Spark": (
                        r"\bspark\b"
                    ),

                    "Git": (
                        r"\bgit\b|github"
                    )
                }
            }

            # ------------------------------------------------
            # ROLE-SPECIFIC SKILL FILTERS
            # ------------------------------------------------

            role_skill_focus = {

                "java": {
                    "Java",
                    "Spring Boot",
                    "SQL",
                    "MySQL",
                    "PostgreSQL",
                    "REST API",
                    "Microservices",
                    "Git",
                    "Docker",
                    "AWS",
                    "Linux",
                    "HTML",
                    "CSS",
                    "JavaScript",
                    "MongoDB",
                    "Jenkins",
                    "Kubernetes"
                },

                "python": {
                    "Python",
                    "Django",
                    "Flask",
                    "FastAPI",
                    "SQL",
                    "MySQL",
                    "PostgreSQL",
                    "MongoDB",
                    "REST API",
                    "Git",
                    "Docker",
                    "AWS",
                    "Linux"
                },

                "data analyst": {
                    "SQL",
                    "Excel",
                    "Power BI",
                    "Tableau",
                    "Python",
                    "Pandas",
                    "NumPy",
                    "Statistics",
                    "Data Visualization",
                    "Google Analytics",
                    "A/B Testing"
                },

                "data engineer": {
                    "Python",
                    "SQL",
                    "MySQL",
                    "PostgreSQL",
                    "MongoDB",
                    "Big Data",
                    "Hadoop",
                    "Spark",
                    "AWS",
                    "Azure",
                    "GCP",
                    "Docker",
                    "Git",
                    "Linux"
                },

                "full stack": {
                    "Java",
                    "Python",
                    "JavaScript",
                    "TypeScript",
                    "React",
                    "Angular",
                    "Node.js",
                    "Express.js",
                    "HTML",
                    "CSS",
                    "SQL",
                    "MySQL",
                    "PostgreSQL",
                    "MongoDB",
                    "REST API",
                    "Git",
                    "Docker"
                },

                "backend": {
                    "Java",
                    "Python",
                    "Spring Boot",
                    "JavaScript",
                    "Node.js",
                    "SQL",
                    "MySQL",
                    "PostgreSQL",
                    "MongoDB",
                    "REST API",
                    "Microservices",
                    "Git",
                    "Docker",
                    "AWS",
                    "Linux"
                },

                "frontend": {
                    "JavaScript",
                    "TypeScript",
                    "React",
                    "Angular",
                    "Vue.js",
                    "HTML",
                    "CSS",
                    "Bootstrap",
                    "Redux",
                    "Git"
                },

                "devops": {
                    "AWS",
                    "Azure",
                    "GCP",
                    "Docker",
                    "Kubernetes",
                    "Jenkins",
                    "CI/CD",
                    "Terraform",
                    "Ansible",
                    "Linux",
                    "Git"
                },

                "ai/ml": {
                    "Python",
                    "Machine Learning",
                    "Deep Learning",
                    "Artificial Intelligence",
                    "TensorFlow",
                    "PyTorch",
                    "Keras",
                    "Scikit-learn",
                    "NLP",
                    "Computer Vision",
                    "Pandas",
                    "NumPy"
                }
            }

            # ------------------------------------------------
            # FIND ROLE FOCUS
            # ------------------------------------------------

            focus_skills = None

            if "java" in role_lower:

                focus_skills = role_skill_focus["java"]

            elif "python" in role_lower:

                focus_skills = role_skill_focus["python"]

            elif "data analyst" in role_lower:

                focus_skills = role_skill_focus["data analyst"]

            elif "data engineer" in role_lower:

                focus_skills = role_skill_focus["data engineer"]

            elif "full stack" in role_lower:

                focus_skills = role_skill_focus["full stack"]

            elif "backend" in role_lower:

                focus_skills = role_skill_focus["backend"]

            elif "frontend" in role_lower:

                focus_skills = role_skill_focus["frontend"]

            elif "devops" in role_lower:

                focus_skills = role_skill_focus["devops"]

            elif (
                "ai" in role_lower
                or "ml" in role_lower
                or "machine learning" in role_lower
            ):

                focus_skills = role_skill_focus["ai/ml"]

            # ------------------------------------------------
            # CALCULATE SKILL DEMAND
            # ------------------------------------------------

            grouped_recommendations = {}

            min_threshold = max(
                1,
                min(
                    int(max(jobs_with_text, 1) * 0.05),
                    20
                )
            )

            for category, patterns in skill_categories.items():

                category_skills = []

                for skill, pattern in patterns.items():

                    # For known roles, only show relevant skills
                    if (
                        focus_skills is not None
                        and skill not in focus_skills
                    ):
                        continue

                    count = combined_text_series.str.contains(
                        pattern,
                        regex=True,
                        na=False
                    ).sum()

                    # Always show the role's main skill
                    main_role_skill = (
                        skill.lower() in role_lower
                        or (
                            "java" in role_lower
                            and skill == "Java"
                        )
                    )

                    if count >= min_threshold or main_role_skill:

                        category_skills.append(
                            (skill, int(count))
                        )

                if category_skills:

                    category_skills = sorted(
                        category_skills,
                        key=lambda pair: pair[1],
                        reverse=True
                    )

                    grouped_recommendations[category] = (
                        category_skills
                    )

            # ------------------------------------------------
            # RECOMMENDED SKILLS
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🔥 Recommended Skills"
            )

            if grouped_recommendations:

                for category, category_skills in (
                    grouped_recommendations.items()
                ):

                    st.markdown(
                        f"**{category}**"
                    )

                    columns = st.columns(
                        min(
                            4,
                            len(category_skills)
                        )
                    )

                    for index, (skill, count) in enumerate(
                        category_skills
                    ):

                        with columns[
                            index % len(columns)
                        ]:

                            st.markdown(
                                f"""
                                <div class="skill-card">
                                    <strong>{skill}</strong><br>
                                    <span style="font-size:12px;color:#9ca3af;">
                                        {count}/{total_matches} jobs
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.write("")

            else:

                st.info(
                    "No recognized skills were found for this role."
                )

            # ------------------------------------------------
            # MATCHING JOBS
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "💼 Matching Jobs"
            )

            st.write(
                f"Jobs matching **{target_role}** based on "
                "the job title."
            )

            # Columns to display
            show_columns = [
                column
                for column in [
                    "job_title",
                    "company",
                    "location",
                    "experience",
                    "salary_raw",
                    "source"
                ]
                if column in matching.columns
            ]

            # Rename columns for better display
            jobs_table = matching[
                show_columns
            ].head(100).copy()

            jobs_table = jobs_table.rename(
                columns={
                    "job_title": "Job Title",
                    "company": "Company",
                    "location": "Location",
                    "experience": "Experience",
                    "salary_raw": "Salary",
                    "source": "Source"
                }
            )

            st.dataframe(
                jobs_table,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                f"Showing up to 100 of {len(matching):,} "
                "matching job records."
            )


# ============================================================
# PAGE 5 — SALARY PREDICTOR
# ============================================================

elif page == "🤖 Salary Predictor":

    st.title(
        "🤖 Machine Learning Salary Predictor"
    )


    st.write(
        "Estimate salary using the trained machine-learning "
        "model based on role, location and experience."
    )


    if not os.path.exists(
        MODEL_FILE
    ):

        st.error(
            "❌ salary_model.pkl was not found."
        )


        st.info(
            "Train the model first:"
        )


        st.code(
            "python src/train_model.py",
            language="bash"
        )


    else:

        try:

            model = joblib.load(
                MODEL_FILE
            )


            col1, col2 = st.columns(2)


            with col1:

                prediction_role = st.text_input(
                    "Job Role",
                    placeholder="Data Analyst"
                )


                prediction_location = st.text_input(
                    "Location",
                    placeholder="Hyderabad"
                )


            with col2:

                prediction_experience = st.text_input(
                    "Experience",
                    placeholder="2 years"
                )


                prediction_skills = st.text_input(
                    "Skills",
                    placeholder="Python SQL Excel"
                )


            if st.button(
                "🚀 Predict Salary",
                use_container_width=True
            ):

                if not prediction_role:

                    st.warning(
                        "Please enter a job role."
                    )

                else:

                    prediction_input = pd.DataFrame({

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
                            + " "
                            + prediction_skills
                        ]

                    })


                    prediction = model.predict(
                        prediction_input
                    )[0]


                    prediction = max(
                        0,
                        prediction
                    )


                    st.success(
                        f"### Estimated Annual Salary: "
                        f"₹{prediction:,.0f}"
                    )


                    st.caption(
                        "This is a model-based estimate, "
                        "not a guaranteed salary."
                    )


        except Exception as error:

            st.error(
                "The salary model could not be loaded."
            )

            st.exception(
                error
            )


# ============================================================
# PAGE 6 — JOB EXPLORER
# ============================================================

elif page == "🔎 Job Explorer":

    st.title(
        "🔎 Job Explorer"
    )


    st.write(
        "Search and filter the complete job dataset."
    )


    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        search_text = st.text_input(
            "Search",
            placeholder="Python, Data Analyst..."
        )


    with col2:

        source_options = sorted(
            jobs["source"]
            .replace("", "Unknown")
            .unique()
            .tolist()
        )


        selected_sources = st.multiselect(
            "Dataset Source",
            source_options,
            default=source_options
        )


    with col3:

        location_search = st.text_input(
            "Location",
            placeholder="Hyderabad"
        )


    filtered_jobs = jobs.copy()


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    if search_text:

        search_mask = (
            filtered_jobs
            .astype(str)
            .apply(
                lambda row:
                row.str.contains(
                    search_text,
                    case=False,
                    na=False
                ).any(),
                axis=1
            )
        )


        filtered_jobs = filtered_jobs[
            search_mask
        ]


    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    if selected_sources:

        filtered_jobs = filtered_jobs[
            filtered_jobs["source"]
            .replace("", "Unknown")
            .isin(
                selected_sources
            )
        ]


    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    if location_search:

        filtered_jobs = filtered_jobs[
            filtered_jobs["location"]
            .str.contains(
                location_search,
                case=False,
                na=False
            )
        ]


    st.metric(
        "Matching Jobs",
        f"{len(filtered_jobs):,}"
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    display_columns = [
        column
        for column in [
            "job_title",
            "company",
            "location",
            "source"
        ]
        if column in filtered_jobs.columns
    ]


    st.dataframe(
        filtered_jobs[
            display_columns
        ].head(500),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎯 Career Map • "
    "Python + Pandas + Scikit-learn + Streamlit"
)

