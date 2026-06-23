import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("cs_courses_updated.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')

combined_text = (df['title'] + " " + df['description']).tolist()
embeddings = model.encode(combined_text, convert_to_tensor=False)

np.save("embeddings.npy", embeddings)
print("Saved!")