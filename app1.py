from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
#from langchain_openai import OpenAI
from langchain_groq import ChatGroq
# from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
#from langchain_openai import ChatOpenAI
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.runnables import RunnableMap
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.schema import messages_from_dict, messages_to_dict
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv
from src.prompt import *
import os




app = Flask(__name__)

load_dotenv()

# api_key = os.environ.get('PINECONE_API_KEY')
# groq_api_key= os.environ.get('GROQ_API_KEY')

os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGSMITH_TRACING_V2"] = "true"

os.environ["SERPAPI_API_KEY"] = os.getenv('SERPAPI_API_KEY')


# secret_key= os.environ.get('SECRET_KEY', "68150090fed0c03f25e726f01c80b7887afd93537357374e122475341eb42900")
# app.secret_key = secret_key #Tells Flask the secret key to use for session management. 

#openai_api_key = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')


#os.environ["OPENAI_API_KEY"] = openai_api_key

# Load the PDF files, split the text, and download embeddings
embeddings = download_hugging_face_embeddings()
# ----- add cretae index code here -----

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

llm = ChatGroq(temperature=0.3, model_name="llama3-70b-8192")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
tools = load_tools(["serpapi"], llm=llm)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # Optional: helpful for debugging
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
# user_memories = {}  

# # 1. Create memory object
# memory = ConversationBufferMemory(
#     memory_key="chat_history", 
#     return_messages=True
# )

# # 2. Create conversational chain with memory
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=retriever,
#     memory=memory,
#     verbose=True  # Optional: helpful for debugging
# ) 



 
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

# 


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)

    # Load previous chat history from session
    # raw_history = session.get("chat_history", [])
    # messages = messages_from_dict(raw_history)

    # Rebuild memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    memory.chat_memory.messages = messages

    # Rebuild RAG chain with memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )

    response = qa_chain.run(msg) 

    return str(response)


    # try:
    #     # Step 1: Try to get answer from book context
    #     response = qa_chain.invoke({"question": msg})

    #     if response.get("answer", "").lower().startswith("i don't know"):
    #         raise ValueError("No good answer from book")

    #     final_answer = response["answer"]

    # except Exception as e:
    #     print("Fallback to agent due to:", str(e))
    #     response = qa_chain.invoke({"question": msg})
    #     final_answer = response["answer"]

    #     # Step 2: Fall back to agent if no answer from book
    #     agent_response = agent.run(msg)
    #     final_answer = (
    #         "Note: The following answer is based on external medical information retrieved via SerpAPI.\n"
    #         f"{agent_response}"
    #     )

    # Save updated memory
    # session["chat_history"] = messages_to_dict(memory.chat_memory.messages)

    return str(final_answer)




# @app.route("/reset", methods=["GET"])
# def reset_chat():
#     session.pop("chat_history", None)
#     return redirect(url_for("index"))  # Replace "index" with your home route name


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
