# AI-Medical-LLM

## Project Overview:

This project is a chatbot application built with Flask, Pinecone, HuggingFace embeddings, and Groq. The chatbot is designed to answer questions about medical topics by retrieving relevant information from a document database. The system integrates document embeddings and similarity search for fast and accurate responses, leveraging retrieval-augmented generation (RAG).

The key functionality of this application is to:

Embed documents (medical content in this case).

Store the embeddings in Pinecone, a vector database.

Use a Groq model for generating responses.

Provide a real-time chatbot interface using Flask.

## Key Techniques and Tools Used:

Flask: A Python web framework used to create a simple web interface where users can interact with the chatbot.

Pinecone: A vector database used to store document embeddings and enable fast similarity search for information retrieval.

HuggingFace Embeddings: A pre-trained model used to convert documents into 384-dimensional embeddings that represent the semantic meaning of the content.

Groq (ChatGroq): A powerful large language model (LLM) used to generate responses to user queries. It is used for retrieving and generating answers based on relevant documents.

Langchain: An end-to-end framework for developing and deploying LLM-based applications. It is used to build the retrieval chain and manage prompt templates.

## Steps Taken in the Project:

1. Environment Setup:
Environment Variables: The project uses dotenv to load sensitive API keys, such as Pinecone API key and Groq API key, from environment variables. This ensures secure handling of credentials without hardcoding them into the source code.

2. Loading Embeddings and Setting Up Pinecone:
Embeddings: The download_hugging_face_embeddings function loads the pre-trained HuggingFace model (sentence-transformers/all-MiniLM-L6-v2) for generating embeddings.

Pinecone Setup:

The project uses PineconeVectorStore to load an existing Pinecone index named "medicalbot". This index stores the embeddings generated for the documents.

The as_retriever method is used to set up the retrieval mechanism. The retriever searches for the top 3 most relevant documents based on similarity to the input query.

3. Integrating Groq for Answer Generation:
Groq Setup: The ChatGroq class is used to interact with the Groq model (Llama3 or Mixtral), which generates answers to queries based on the retrieved documents.

Prompt Setup: The ChatPromptTemplate is used to format the conversation between the system and user. The system message likely contains some predefined context or instructions for the model.

RAG Chain: The retrieval-augmented generation chain is set up using Langchain's create_retrieval_chain method. This chain first retrieves relevant documents and then generates a response using the Groq model based on those documents.

4. Web Interface with Flask:
Flask App: The Flask app provides a web interface for users to chat with the model.

/get Route: This route receives the user’s query as input, uses the RAG chain to get relevant documents, and generates a response. The response is then returned as the chatbot's answer.

/ Route: This route serves the chat interface (likely an HTML template for the front-end).

5. Running the Application:
The app is set to run on host="0.0.0.0" and port=8080, making it accessible on any IP address in a local network.

## Challenges Faced and Overcome:
Embedding and Vector Storage: One challenge was embedding the documents and managing them efficiently. Using Pinecone as a vector database allowed easy management of the embeddings and fast retrieval of relevant documents.

Integrating Multiple APIs: Integrating Groq with Pinecone and Langchain required proper setup and handling of various dependencies. Using Langchain's structured framework for document retrieval and question answering simplified this integration.

Real-Time Interaction: Setting up a real-time interaction with Flask required handling multiple layers of functionality: receiving user input, querying Pinecone, and generating responses with Groq.

## Results Achieved:
Real-Time Chatbot: The chatbot can now process user queries in real-time, retrieve relevant documents from a database, and generate answers using the Groq model.

Efficient Document Search: By using Pinecone for similarity search, the system ensures that only the most relevant documents are used for generating answers, providing accurate and contextually appropriate responses.

Scalability: The use of Groq and Pinecone allows the system to scale with more documents and queries, making it suitable for large-scale AI applications like medical information retrieval.