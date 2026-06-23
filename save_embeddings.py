import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_data():
    df = pd.read_csv("cs_courses_updated.csv")
    df['title'] = df['title'].str.replace(r'Academics.*Descriptions', '', regex=True).str.strip()

    def get_num(text):
        match = re.search(r'\d+', text)
        return int(match.group()) if match else 999

    df['course_num'] = df['title'].apply(get_num)
    df = df.sort_values('course_num').reset_index(drop=True)  # <-- critical fix

    cols_to_fix = ['prerequisites', 'days (spring 26)', 'times (spring 26)',
                   'location (spring 26)', 'professor (spring 26)', 'languages']
    for col in cols_to_fix:
        df[col] = df[col].fillna("N/A")
    return df

df = load_data()
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device='cpu')

combined_text = df['description'].tolist()
embeddings = model.encode(combined_text, convert_to_tensor=False, device='cpu')

np.save("embeddings.npy", embeddings)
print(f"Saved!")