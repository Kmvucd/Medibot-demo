from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import load_tools, initialize_agent, AgentType


os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGSMITH_TRACING_V2"] = "true"

os.environ["SERPAPI_API_KEY"] = os.getenv('SERPAPI_API_KEY')

# SerpAPI setup
tools = load_tools(["serpapi"], llm=llm)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # Optional: helpful for debugging
)



# Create a memory dictionary for each session
user_memories = {}  

# 1. Create memory object
memory = ConversationBufferMemory()

# 2. Create conversational chain with memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True  # Optional: helpful for debugging
)


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)

    # Stateless QA chain (no memory, no session)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        verbose=False
    )

    # Step 1: Attempt response from the PDF/book
    response = qa_chain.invoke({"query": msg})
    raw_answer = response.get("result", "").strip().lower()

    # Step 2: If answer is missing or vague, fallback to SerpAPI agent
   
   
#    if not raw_answer or 
    if "i don't know" in raw_answer:   
#    if any(
#     phrase in raw_answer for phrase in ["i don't know", "not sure", "unable to answer", "no relevant information"]):
        print("Falling back to SerpAPI agent...")
        agent_response = agent.run(msg)
        final_answer = (
            "Note: The following answer is based on external medical information retrieved via SerpAPI.\n"
            f"{agent_response}"
        )
    else:
        print("Answer from book context:", raw_answer)
        final_answer = response["result"]

    return str(final_answer)























