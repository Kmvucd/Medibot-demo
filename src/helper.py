from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings



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