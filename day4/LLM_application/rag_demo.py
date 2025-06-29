# Import necessary libraries
import os
from dotenv import load_dotenv  # For loading environment variables from .env file
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI  # Using ChatOpenAI with OpenRouter base URL
from langchain_community.vectorstores import FAISS  # Vector store for similarity search
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable chunks
from langchain.chains import RetrievalQA  # For creating retrieval-based question answering chains

# --- 1. Load Environment Variables ---
# Load environment variables from a .env file.
# This is where you'll store your OPENROUTER_API_KEY.
load_dotenv()

# It's good practice to check if the API key is loaded.
if not os.getenv("OPENROUTER_API_KEY"):
    print("Error: OPENROUTER_API_KEY not found. Please create a .env file and add your key.")
    exit()

# --- 2. Load and Prepare Data ---
# We'll load a simple text file as our data source.
# In a real-world application, this could be a collection of documents,
# a website's content, or any other text-based data.
try:
    with open("data/sample.txt", "r", encoding="utf-8") as f:
        text_data = f.read()
except FileNotFoundError:
    print("Error: data/sample.txt not found. Please make sure the file exists.")
    exit()

# Text Splitter: To handle long texts, we split them into smaller chunks.
# This is crucial for the model to process the information effectively.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # The size of each chunk in characters.
    chunk_overlap=200,  # The number of characters to overlap between chunks.
)
texts = text_splitter.split_text(text_data)

# --- 3. Create Embeddings and Vector Store ---
# Embeddings are numerical representations of text.
# We will use a local model running in Ollama.
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
)

# Vector Store (FAISS): This is a database that stores the text chunks
# and their corresponding embeddings. It allows for efficient searching
# of the most relevant text chunks based on a query.
# FAISS (Facebook AI Similarity Search) is a library for efficient similarity search.
try:
    vector_store = FAISS.from_texts(texts, embeddings)
except Exception as e:
    print(f"Error creating vector store: {e}")
    exit()


# --- 4. Initialize the Language Model ---
# We'll use a chat model from OpenAI, accessed through OpenRouter.
# You can choose from various models available on OpenRouter.
llm = ChatOpenAI(
    model_name="google/gemini-2.0-flash-exp:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.3,  # Controls the randomness of the model's output.
)

# --- 5. Create the RAG Chain ---
# The RetrievalQA chain combines the language model and the vector store.
# It will first retrieve relevant documents from the vector store
# and then use the language model to answer the question based on the retrieved documents.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" is a simple method that stuffs all retrieved text into the prompt.
    retriever=vector_store.as_retriever(),
)

# --- 6. Run the Application ---
# Now, we can ask questions to our RAG application.
print("RAG Demo Application")
print("Ask a question about the content in data/sample.txt")
print("Type 'exit' to quit.")

while True:
    query = input("> ")
    if query.lower() == "exit":
        break
    
    # Invoke the chain with the user's query.
    try:
        answer = qa_chain.invoke(query)
        print(answer['result'])
    except Exception as e:
        print(f"An error occurred: {e}")

