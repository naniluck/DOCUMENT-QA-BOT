import chromadb
from chromadb.utils import embedding_functions

def query_rag(user_query):
    # లోకల్ ఎంబెడ్డింగ్ ఫంక్షన్
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # డేటాబేస్ కనెక్ట్ అవ్వడం
    client = chromadb.PersistentClient(path="./db")
    collection = client.get_collection(name="document_knowledge_base", embedding_function=embedding_fn)
    
    # డేటాబేస్ నుండి రిలేటెడ్ సమాచారాన్ని వెతకడం
    results = collection.query(query_texts=[user_query], n_results=2)
    
    # సమాధానం ప్రింట్ చేయడం
    print("\n--- బాట్ సమాధానం ---")
    if results['documents']:
        for doc in results['documents'][0]:
            print(f"-> {doc}")
    else:
        print("సమాచారం దొరకలేదు.")

if __name__ == "__main__":
    question = input("మీ ప్రశ్న అడగండి: ")
    query_rag(question)