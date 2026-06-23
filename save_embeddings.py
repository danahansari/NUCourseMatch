import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("cs_courses_updated.csv")
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

embeddings = model.encode((df['description']).tolist(), convert_to_tensor=False, device='cpu')

np.save("embeddings.npy", embeddings)
print("Saved!")