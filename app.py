import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

st.title("Document QA Bot")

# DB connection logic
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./db")
try:
    collection = client.get_collection(name="document_knowledge_base", embedding_function=embedding_fn)
except:
    collection=client.create_collection(name="document_knowledge_base", embedding_function=embedding_fn)

query = st.text_input("Meeru adhagalanukune prashna ikkada type cheyandi:")

if st.button("Submit"):
    if query:
        results = collection.query(query_texts=[query], n_results=2)
        if results['documents']:
            for doc in results['documents'][0]:
                st.write(f"-> {doc}")
        else:
            st.write("Samaacharam dorakaledhu.")