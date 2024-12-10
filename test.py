import os
import streamlit as st
import time
import hashlib
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Langsmith Tracking configuration
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Smart search Chatbot With OPENAI"

# Initialize Streamlit UI
st.title("SmartSearch: Research Tool 🌐🔗 🔍")
st.sidebar.title("Article URLs")

# Initialize session state
if "satisfied" not in st.session_state:
    st.session_state.satisfied = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "rating" not in st.session_state:
    st.session_state.rating = 3  # Default slider value
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []  # To track ratings over time
if "query_times" not in st.session_state:
    st.session_state.query_times = []  # To track query response times
if "url_content_hashes" not in st.session_state:
    st.session_state.url_content_hashes = {}  # To store URL and content hash

# Optional debugging checkbox
debug_mode = st.sidebar.checkbox("Enable Debugging")

# Collect URLs from user input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Button to initiate processing
process_url_clicked = st.sidebar.button("Process URLs")
vectorstore_path = "chroma_vectorstore"
main_placeholder = st.empty()

# Initialize the language model (ChatGPT-4)
llm = ChatOpenAI(model="gpt-4", temperature=0.6, max_tokens=500)

# Function to compute hash of content
def compute_hash(content):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

# Function to load data from URLs
def load_data(urls):
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... Started...")
    data = loader.load()
    # Add 'source' metadata to each document
    for doc, url in zip(data, urls):
        doc.metadata['source'] = url
    if debug_mode:
        st.write(f"Loaded data with metadata: {data}")  # For debugging purposes
    return data

# Function to split text into chunks
def split_data(data):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1500,
        chunk_overlap=200,
    )
    main_placeholder.text("Splitting Text... Started...")
    docs = []
    for doc in data:
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata["source"] = doc.metadata["source"]  # Propagate source metadata
        docs.extend(chunks)
    if debug_mode:
        for doc in docs:
            st.write(f"Document preview: {doc.page_content[:100]}...")
            st.write(f"Metadata: {doc.metadata}")
    return docs

# Function to create embeddings and store them in Chroma
def create_embeddings(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=vectorstore_path,  # Ensure persistence is enabled
        embedding_function=embeddings
    )
    vectorstore.add_texts(
        texts=[doc.page_content for doc in docs],
        metadatas=[doc.metadata for doc in docs]
    )
    st.write("Embeddings created and stored in Chroma successfully.")
    return vectorstore

# Process URLs and create embeddings when button is clicked
vectorstore = None
if process_url_clicked and urls:
    updated_docs = []  # List of documents with updated content
    data = load_data(urls)

    if data:
        for doc, url in zip(data, urls):
            if url:  # Ensure the URL is valid
                content_hash = compute_hash(doc.page_content)
                previous_hash = st.session_state.url_content_hashes.get(url)

                # Process only if content hash has changed or URL is new
                if previous_hash != content_hash:
                    updated_docs.append(doc)
                    st.session_state.url_content_hashes[url] = content_hash  # Update stored hash

        if updated_docs:
            docs = split_data(updated_docs)
            if docs:
                vectorstore = create_embeddings(docs)
                time.sleep(2)
            else:
                st.write("No new or updated documents to process.")
        else:
            st.write("No new or updated content found in the provided URLs.")
    else:
        st.write("Failed to load data from the URLs.")

# Query input and feedback loop
query = main_placeholder.text_input("Enter your question:")

if query and query != st.session_state.last_query:
    st.session_state.satisfied = False
    st.session_state.last_query = query

if query and not st.session_state.satisfied:
    if vectorstore is not None or os.path.exists(vectorstore_path):
        if vectorstore is None:
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=OpenAIEmbeddings()
            )
        
        # Initialize retrieval chain with source document support
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, 
            retriever=vectorstore.as_retriever(), 
            return_source_documents=True
        )
        
        # Measure query response time
        start_time = time.time()
        result = chain.invoke({"question": query})
        response_time = time.time() - start_time
        st.session_state.query_times.append(response_time)
        
        # Display the answer
        st.header("Answer")
        st.write(result["answer"])

        # Deduplicate and display sources
        sources = result.get("source_documents", [])
        if sources:
            st.subheader("Sources:")
            unique_sources = set(source.metadata.get('source', 'Unknown') for source in sources)
            for unique_source in unique_sources:
                st.write(unique_source)

        # Collect user feedback using a form
        with st.form("feedback_form"):
            st.markdown("""
            ### Rate the Answer
            - **1**: Unrelated or not helpful
            - **2**: Somewhat related but vague
            - **3**: Average, could be more descriptive
            - **4**: Good, detailed but could use slight improvement
            - **5**: Excellent, clear and straight to the point
            """)
            
            temp_rating = st.slider(
                "Rate this answer:",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
            )
            
            feedback_comment = st.text_area("Additional Comments (optional):")
            submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            # Save feedback to session state
            st.session_state.feedback_history.append({
                "query": query,
                "rating": temp_rating,
                "comments": feedback_comment
            })

            if temp_rating >= 4:
                st.success(f"Thank you for your feedback! Rating: {temp_rating}/5")
                st.session_state.satisfied = True  # Allow next query
            else:
                st.warning("Refining the response based on feedback...")
                refined_query = f"Refine the answer: Rating {temp_rating}/5. {feedback_comment or 'No comments'}"
                refined_result = chain.invoke({"question": refined_query})
                st.write("Refined Answer:")
                st.write(refined_result["answer"])

# Feedback visualization
if st.session_state.feedback_history:
    feedback_df = pd.DataFrame(st.session_state.feedback_history)
    if "rating" in feedback_df.columns:
        feedback_summary = feedback_df["rating"].value_counts().sort_index()
        fig, ax = plt.subplots()
        feedback_summary.plot.bar(ax=ax, color=["red", "orange", "yellow", "green", "blue"], edgecolor="black")
        ax.set_title("Number of Queries by Feedback Rating")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# Query response time visualization
if st.session_state.query_times:
    response_df = pd.DataFrame({
        "Query #": range(1, len(st.session_state.query_times) + 1),
        "Response Time (s)": st.session_state.query_times
    })
    fig, ax = plt.subplots()
    ax.plot(response_df["Query #"], response_df["Response Time (s)"], marker="o", color="purple")
    ax.set_title("Query Response Time")
    ax.set_xlabel("Query #")
    ax.set_ylabel("Response Time (s)")
    st.pyplot(fig)
