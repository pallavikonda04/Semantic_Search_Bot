import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# App Title
st.title("🔍 Semantic Search Bot")
st.write("Search documents based on meaning using Sentence Transformers and FAISS.")

# Load Sentence Transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Sample documents
documents = [
    "Python is a popular programming language for AI and data science.",
    "Machine learning enables computers to learn from data.",
    "Deep learning is a subset of machine learning.",
    "Natural Language Processing helps computers understand human language.",
    "Streamlit is used to build interactive web applications in Python."
]

# Generate embeddings
embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# User input
query = st.text_input("Enter your query:")

# Search button
if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a search query.")
    else:
        # Generate query embedding
        query_embedding = model.encode([query])

        # Search top 2 similar documents
        distances, indices = index.search(np.array(query_embedding), k=2)

        st.subheader("Top Matching Documents")

        for i in indices[0]:
            st.write("•", documents[i])