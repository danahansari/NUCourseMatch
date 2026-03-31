import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re

st.set_page_config(page_title="NU CourseMatch", page_icon="💜", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .course-card { border: 1px solid #ddd; padding: 15px; border-radius: 10px; background-color: white; margin-bottom: 20px; }
    
    div.stButton > button {
        background-color: #4e2a84; /* NU Purple */
        color: white;
        border-radius: 8px;
        border: none;
        height: 3em;
        width: 100%;
        transition: none; /* Removes the fade effect */
    }

    /* 3. Force the hover state to stay the exact same color */
    div.stButton > button:hover, div.stButton > button:active, div.stButton > button:focus {
        background-color: #4e2a84;
        color: white;
        border: none;
    }

    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    df = pd.read_csv("cs_courses_updated.csv")
    df['title'] = df['title'].str.replace(r'Academics.*Descriptions', '', regex=True).str.strip()
    
    # Extract course number for sorting
    def get_num(text):
        match = re.search(r'\d+', text)
        return int(match.group()) if match else 999
    
    df['course_num'] = df['title'].apply(get_num)
    df = df.sort_values('course_num') # Sort once here
    
    cols_to_fix = ['prerequisites', 'days (spring 26)', 'times (spring 26)', 
                   'location (spring 26)', 'professor (spring 26)', 'languages']
    for col in cols_to_fix:
        df[col] = df[col].fillna("N/A")
    return df

df = load_data()
model = get_model()

@st.cache_resource
def get_embeddings(_df):
    # Combine Title and Description to make the search more robust
    combined_text = (_df['title'] + " " + _df['description']).tolist()
    return model.encode(combined_text, convert_to_tensor=True)

embeddings = get_embeddings(df)

st.sidebar.image("https://www.mccormick.northwestern.edu/images/structure/engineering-logo.svg")
st.sidebar.title("Preferences")
only_spring = st.sidebar.toggle("Only Spring 2026", value=True)
project_only = st.sidebar.toggle("Project-Based Only")
pref_lang = st.sidebar.selectbox("Filter by Language", ["All", "Python", "C++", "Java", "Rust", "SQL"])

st.title("NU CourseMatch")
st.markdown("#### *Discover your perfect CS course at Northwestern!*")
st.write("---")

col_left, col_right = st.columns(2)
with col_left:
    # Use the sorted unique titles
    selected_course_title = st.selectbox(
        "Option A: Select a course you enjoyed:", 
        ["None"] + list(df['title'].unique())
    )
with col_right:
    keyword_search = st.text_input(
        "Option B: Search by topic/interest:", 
        placeholder="Try 'AI in music' or 'Hardware'"
    )

if st.button("Generate Recommendations"):
    query_embedding = None
    source_name = ""

    if keyword_search:
        query_embedding = model.encode(keyword_search, convert_to_tensor=True)
        source_name = f"'{keyword_search}'"
    elif selected_course_title != "None":
        idx = df[df['title'] == selected_course_title].index[0]
        query_embedding = embeddings[idx]
        source_name = f"'{selected_course_title}'"
    else:
        st.error("Please enter a search term or select a course.")
        st.stop()

    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
    results = df.copy()
    results['similarity'] = cosine_scores.tolist()
    
    # Filter logic
    if only_spring:
        results = results[results['is_spring_26'] == 1]
    if pref_lang != "All":
        results = results[results['languages'].str.contains(pref_lang, case=False, na=False)]
    if project_only:
        results = results[results['is_project_based'] == 1]
        
    recs = results[results['title'] != selected_course_title].sort_values('similarity', ascending=False).head(5)
    
    if recs.empty:
        st.warning("No matches found. Try relaxing your filters!")
    else:
        st.success(f"Matches for {source_name}:")
        for i, row in recs.iterrows():
            with st.container():
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.subheader(row['title'])
                    st.caption(f"**Similarity Score:** {round(row['similarity'] * 100)}% | **Prereqs:** {row['prerequisites']}")
                    st.write(row['description'][:250] + "...")
                    with st.expander("Read Full Description"):
                        st.write(row['description'])
                        st.write(f"[View Official Course Page]({row['url']})")
                with c2:
                    if row['is_spring_26'] == 1:
                        # Spring info re-added here
                        st.info(f"📅 **Spring 2026**\n\n"
                                f"**Prof:** {row['professor (spring 26)']}\n\n"
                                f"**Days:** {row['days (spring 26)']}\n\n"
                                f"**Time:** {row['times (spring 26)']}\n\n"
                                f"**Location:** {row['location (spring 26)']}")
                    else:
                        st.info("*Not offered this Spring*")
                    st.write(f"**Languages:** {row['languages']}")
                st.divider()