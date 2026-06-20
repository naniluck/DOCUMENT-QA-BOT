import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# పేజీ టైటిల్
st.set_page_config(page_title="Document QA Bot")
st.title("Document QA Bot")

# 1. ఎంబెడ్డింగ్ ఫంక్షన్ సెటప్
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# డేటాబేస్ కనెక్షన్ (Local DB)
client = chromadb.PersistentClient(path="./db")
# కలెక్షన్ క్రియేట్ చేయడం
collection = client.get_or_create_collection(name="document_knowledge_base", embedding_function=embedding_fn)

# 2. ఫైల్ అప్‌లోడ్ సెక్షన్
uploaded_file = st.file_uploader("Upload Your PDF here", type=["pdf"])

if uploaded_file is not None:
    # PDF చదవడం
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # డేటాబేస్ లో సేవ్ చేయడం
    collection.upsert(documents=[text], ids=["doc1"])
    st.success("PDF ఫైల్ సక్సెస్‌ఫుల్‌గా అప్‌లోడ్ అయ్యింది! ఇప్పుడు మీరు ప్రశ్న అడగవచ్చు.")

# 3. ప్రశ్న అడిగే సెక్షన్
query = st.text_input("Type your Question here:")

if st.button("Submit"):
    if query:
        # సమాధానం కోసం వెతకడం
        results = collection.query(query_texts=[query], n_results=1)
        
        if results['documents'] and results['documents'][0]:
            st.subheader("సమాధానం:")
            st.write(results['documents'][0][0])
        else:
            st.warning("క్షమించండి, డాక్యుమెంట్‌లో సమాచారం దొరకలేదు.")
    else:
        st.error("దయచేసి ఏదైనా ప్రశ్న టైప్ చేయండి.")