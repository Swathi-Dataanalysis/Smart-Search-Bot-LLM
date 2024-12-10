import os
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables (e.g., OpenAI API key from .env)
load_dotenv()

# Initialize ChatOpenAI with GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0.6, max_tokens=500)

# Input URLs manually (Replace these with your URLs)
urls = [
    "https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html",
    "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html"
]

# Load data from URLs
loader = UnstructuredURLLoader(urls=urls)
print("Data Loading...Started...✅✅✅")
data = loader.load()

# Print loaded data for debugging
print(f"Loaded data: {data}")

# Split data with chunk overlapping
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','],
    chunk_size=1500,
    chunk_overlap=200  # Adding chunk overlap to maintain context across chunks
)
print("Text Splitter...Started...✅✅✅")
docs = text_splitter.split_documents(data)

# Print split documents for debugging
print(f"Split documents: {docs}")

if not docs:  # Check if docs list is empty
    print("No documents to create embeddings from. Exiting...")
else:
    # Create embeddings and save them to Chroma vector store
    embeddings = OpenAIEmbeddings()
    vectorstore_path = "chroma_vectorstore"
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=vectorstore_path)
    print("Embedding Vector Started Building...✅✅✅")
    print("Embeddings created and stored in Chroma successfully.")
    time.sleep(2)

# Simulate a question input (Replace with your question)
query = "What is the Tiago iCNG price?"

if query:
    # Load the Chroma vector store from the directory if it exists
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=OpenAIEmbeddings())

    # Create the retrieval chain
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    result = chain({"question": query})
    
    # Print the answer
    print("Answer:")
    print(result["answer"])

    # Display sources, if available
    sources = result.get("sources", "")
    if sources:
        print("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            print(source)
