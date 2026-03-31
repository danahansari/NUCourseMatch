import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load data
df = pd.read_csv("cs_courses_updated.csv")

# Initialize the NLP model
print("Loading NLP model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
print("Vectorizing course descriptions...")
embeddings = model.encode(df['description'].tolist(), convert_to_tensor=True)

def recommend_course(liked_title_keywords, top_n=5, only_spring=False, language=None):
    # Find the course the user liked
    match = df[df['title'].str.contains(liked_title_keywords, case=False, na=False)]
    
    if match.empty:
        return "Course not found. Try a different keyword (e.g., 'Graphics' or '311')."
    
    idx = match.index[0]
    print(f"\nFinding matches for: {df.iloc[idx]['title']}")

    # Calculate cosine similarity (formula: (A . B) / (||A|| * ||B||))
    cosine_scores = util.cos_sim(embeddings[idx], embeddings)[0]
    
    # Add scores to a copy of the dataframe
    results = df.copy()
    results['score'] = cosine_scores.tolist()
    
    # Apply filters
    if only_spring:
        results = results[results['is_spring_26'] == 1]
    
    # Check the 'languages' column
    if language:
        results = results[results['languages'].str.contains(language, case=False, na=False)]

    # Sort and return (skip the first one because it's the course itself)
    return results.sort_values(by='score', ascending=False).head(top_n + 1)[1:]