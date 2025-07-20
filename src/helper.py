import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from serpapi import GoogleSearch

#Extract data from pdf file

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Tag each document with the PDF filename it came from
    for doc in documents:
        file_path = doc.metadata.get("source", "")
        doc.metadata["source"] = os.path.basename(file_path)

    return documents


#Split the data into smaller chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

#Download embeddings from HuggingFace
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

#Search for relevant documents using SerpAPI
def google_search(query):
    params = {
        "q": query,
        "api_key": os.getenv('SERPAPI_API_KEY'),
        "num":1  # Number of results to return
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Try answer box first
    if "answer_box" in results and "snippet" in results["answer_box"]:
        return results["answer_box"]["snippet"]
    
    # Then try organic results
    if "organic_results" in results and results["organic_results"]:
        return results["organic_results"][0].get("snippet", "No relevant result found.")
    
    return "Sorry, I couldn’t find anything relevant online either."