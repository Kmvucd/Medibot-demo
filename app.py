from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
#from langchain_openai import OpenAI
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
#from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.runnables import RunnableMap
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import messages_from_dict, messages_to_dict
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

api_key = os.environ.get('PINECONE_API_KEY')
groq_api_key= os.environ.get('GROQ_API_KEY')

secret_key= os.environ.get('SECRET_KEY', "68150090fed0c03f25e726f01c80b7887afd93537357374e122475341eb42900")
app.secret_key = secret_key #Tells Flask the secret key to use for session management. 

#openai_api_key = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = api_key
os.environ["GROQ_API_KEY"] = groq_api_key


#os.environ["OPENAI_API_KEY"] = openai_api_key

# Load the PDF files, split the text, and download embeddings
embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

#Embed each document and upload to the existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create a retriever from the Pinecone vector store
# This will allow us to retrieve relevant documents based on user queries
retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k": 3})

# Create a question-answering chain using the OpenAI model and the retriever
#llm = OpenAI(temperature=0.4, max_tokens=500)

llm = ChatGroq(temperature=0.7, model_name="llama3-70b-8192")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
# # Compose RAG chain manually using LCEL style
# rag_chain = (
#     RunnableMap({
#         "context": lambda x: retriever.invoke(x["input"]),
#         "input": lambda x: x["input"]
#     })
#     | question_answer_chain
# )

# Create a memory dictionary for each session
user_memories = {}  

# 1. Create memory object
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

# 2. Create conversational chain with memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True  # Optional: helpful for debugging
) 



 
@app.route("/")
def index():
    return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     question = msg
#     print(question)
#     response = qa_chain.invoke({"question": msg})
#     print("Response : ", response)
#     return str(response["answer"])

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)

    # Load previous chat history from session
    raw_history = session.get("chat_history", [])
    messages = messages_from_dict(raw_history)

    # Get session chat history
    chat_history = session.get("chat_history", [])

    # Rebuild memory with previous chat history
    # memory = ConversationBufferMemory(return_messages=True)
    # chat_history = memory.chat_memory.messages 
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    memory.chat_memory.messages = messages
    # Rebuild chain with memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )

    # Invoke the chain
    #response = qa_chain.invoke({"question": msg, "chat_history": chat_history})
    
    response = qa_chain.run(msg)  # Let LangChain manage chat_history


    # Save updated memory to session
    session["chat_history"] = messages_to_dict(memory.chat_memory.messages)

    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
