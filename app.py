from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
#from langchain_openai import OpenAI
#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents.agent_types import AgentType
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv
from src.prompt import *
import os

# comments
app = Flask(__name__)

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
os.environ["SERPAPI_API_KEY"] = os.getenv('SERPAPI_API_KEY')

#os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

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

# Load SerpAPI tool
tools = load_tools(["serpapi"], llm=llm)
# Initialize SerpAPI agent with the same LLM
serp_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)



def is_uncertain_answer(answer: str) -> bool:
    """Check if the answer indicates uncertainty or lack of information."""
    uncertain_phrases = [
        "i don't know", "not sure", "unable to answer", "not mentioned",
        "no relevant information", "no data available", "cannot determine", "not enough information"
    ]
    return any(phrase in answer.lower() for phrase in uncertain_phrases)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)

    # Use invoke() instead of deprecated run()
    response = qa_chain.invoke({"question": msg})
    raw_answer = response["answer"]
    print("Book answer:", raw_answer)

    # Step 2: If unsure, fallback to SerpAPI
    if is_uncertain_answer(raw_answer):
        print("Falling back to SerpAPI...")
        serp_response = serp_agent.run(msg)
        final_answer = (
            "Note: This answer is based on external information from the web (SerpAPI):\n"
            f"{serp_response}"
        )
    else:
        final_answer = raw_answer

    # Return the answer directly
    return str(final_answer)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8501, debug= True)