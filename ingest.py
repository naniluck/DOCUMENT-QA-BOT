import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# SentenceTransformer మోడల్‌ను లోకల్‌గా వాడుతుంది
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def extract_pdf_pages(file_path):
    extracted_data = []
    reader = PdfReader(file_path)
    for index, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            extracted_data.append({
                "text": " ".join(text.split()), 
                "metadata": {"source": os.path.basename(file_path), "page": index + 1}
            })
    return extracted_data

def save_to_vector_db(chunks, db_path="./db"):
    client = chromadb.PersistentClient(path=db_path)
    
    # ఇక్కడ మనం సెంటెన్స్ ట్రాన్స్‌ఫార్మర్‌ని వాడుతున్నాం
    collection = client.get_or_create_collection(
        name="document_knowledge_base", 
        embedding_function=embedding_fn
    )
    
    collection.add(
        ids=[f"id_{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )
    print(f"Successfully indexed {len(chunks)} chunks!")

if __name__ == "__main__":
    all_chunks = []
    # ./data ఫోల్డర్‌లో PDFలు ఉన్నాయని నిర్ధారించుకోండి
    for file in os.listdir("./data"):
        if file.endswith(".pdf"):
            print(f"Processing {file}...")
            pages = extract_pdf_pages(os.path.join("./data", file))
            all_chunks.extend(pages)
    
    if all_chunks:
        save_to_vector_db(all_chunks)
    else:
        print("No files found!")