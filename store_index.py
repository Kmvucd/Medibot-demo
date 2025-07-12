from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from langchian.vectorstores import Pinecone 
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

# Ensure the PINECONE_API_KEY is set in your environment variables
# You can set it in a .env file or directly in your environment
api_key = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = api_key

# Store the index in Pinecone
# Load the PDF files, split the text, and download embeddings
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
# Ensure you have the PINECONE_API_KEY set in your environment variables
pc = Pinecone(api_key=api_key)

index_name = "medicalbot"

# Create a new index in Pinecone if it doesn't already exist
pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
)


# Create the Pinecone vector store from the text chunks and embeddings
# This will create a new index in Pinecone if it doesn't already exist
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)


