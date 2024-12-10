import os
import streamlit as st
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

# Initialize Streamlit UI
st.title("SmartSearch: Research Tool 🌐🔗 🔍")
st.sidebar.title("Configuration and Article URLs")

# Collect user's OpenAI API Key
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""  # Default to empty

st.session_state.user_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    value=st.session_state.user_api_key
)

# Ensure the API Key is provided before continuing
if not st.session_state.user_api_key:
    st.sidebar.warning("Please provide your OpenAI API Key to proceed.")
    st.stop()
else:
    os.environ["OPENAI_API_KEY"] = st.session_state.user_api_key

# Initialize session state for additional functionalities
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

# Optional debugging checkbox
debug_mode = st.sidebar.checkbox("Enable Debugging")

# Sidebar for input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")
vectorstore_path = "chroma_vectorstore"
main_placeholder = st.empty()

# Initialize Language Model
try:
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.6,
        max_tokens=500
    )
except Exception as e:
    st.error("Failed to initialize the language model. Please check your OpenAI API Key.")
    st.stop()

# Function to load data from URLs
def load_data(urls):
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... Started...")
    data = loader.load()
    if debug_mode:
        st.write(f"Loaded data: {data}")  # For debugging purposes
    return data

# Function to split text into chunks
def split_data(data):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1500,
        chunk_overlap=200
    )
    main_placeholder.text("Splitting Text... Started...")
    docs = text_splitter.split_documents(data)
    if debug_mode:
        st.write(f"Split documents: {docs}")
    return docs

# Function to create embeddings and store them in Chroma
def create_embeddings(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=vectorstore_path)
    main_placeholder.text("Creating Embedding Vectors... Completed.")
    st.write("Embeddings created and stored in Chroma successfully.")
    return vectorstore

# Process URLs and create embeddings when button is clicked
vectorstore = None
if process_url_clicked and urls:
    data = load_data(urls)
    if data:
        docs = split_data(data)
        if docs:
            vectorstore = create_embeddings(docs)
            time.sleep(2)
        else:
            st.write("No content found in the URLs provided.")
    else:
        st.write("Failed to load data from URLs.")

# Query input and feedback loop
query = st.text_input("Enter your question:", value=st.session_state.last_query if not st.session_state.satisfied else "")

if query and query != st.session_state.last_query:
    st.session_state.satisfied = False
    st.session_state.last_query = query

if query and not st.session_state.satisfied:
    if vectorstore is not None or os.path.exists(vectorstore_path):
        if vectorstore is None:
            vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=OpenAIEmbeddings())
        
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

        # Display the sources if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)

        # Collect user feedback using `with` to wait for button submission
        with st.form("feedback_form"):
            st.markdown("""
            ### Rate the Answer
            - **1**: Unrelated or not helpful
            - **2**: Somewhat related but vague
            - **3**: Average, could be more descriptive
            - **4**: Good, detailed but could use slight improvement
            - **5**: Excellent, clear and straight to the point
            """)

            # Slider for rating
            temp_rating = st.slider(
                "Rate this answer:",
                min_value=1,
                max_value=5,
                value=3,  # Default value
                step=1,
            )

            # Optional comments for more feedback
            feedback_comment = st.text_area("Additional Comments (optional):")

            # Submit feedback button within the form
            submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            # Save feedback to session state
            st.session_state.rating = temp_rating
            st.session_state.feedback_history.append({
                "query": query,
                "rating": temp_rating,
                "comments": feedback_comment,
            })

            # Process refinement only if the rating is less than 4
            if temp_rating < 4:
                st.warning(f"Refining the response based on your feedback (Rating: {temp_rating}/5). Please wait...")
                refined_query = (
                    f"Refine this answer based on feedback: I rated it {temp_rating}/5. "
                    f"{feedback_comment if feedback_comment else 'No additional comments provided.'}"
                )
                refined_result = chain.invoke({"question": refined_query})
                st.write("Refined Answer:")
                st.write(refined_result["answer"])
            else:
                st.success(
                    f"Execution completed successfully! The answer was accepted with a rating of {st.session_state.rating}/5."
                )
                st.write("Thank you for your feedback! You can now ask another question.")
                st.session_state.satisfied = True  # Mark satisfaction as True.

# Visualization for feedback ratings
if st.session_state.feedback_history:
    st.subheader("Feedback Ratings Distribution")
    
    feedback_df = pd.DataFrame(st.session_state.feedback_history)
    if not feedback_df.empty:
        feedback_summary = feedback_df["rating"].value_counts().sort_index()

        # Plot a colorful bar chart
        fig, ax = plt.subplots()
        feedback_summary.plot.bar(
            ax=ax,
            color=['red', 'orange', 'yellow', 'green', 'blue'],  # Distinct colors for each rating
            edgecolor='black'
        )
        ax.set_title("Number of Queries by Feedback Rating")
        ax.set_xlabel("Feedback Rating (1 to 5)")
        ax.set_ylabel("Number of Queries")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)

# Visualization for query response times
if st.session_state.query_times:
    st.subheader("Query Response Times")
    response_df = pd.DataFrame({"Query #": range(1, len(st.session_state.query_times) + 1), "Response Time (s)": st.session_state.query_times})
    fig, ax = plt.subplots()
    ax.plot(response_df["Query #"], response_df["Response Time (s)"], marker='o', color='purple', linewidth=2)
    ax.set_title("Response Time per Query")
    ax.set_xlabel("Query #")
    ax.set_ylabel("Response Time (seconds)")
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    st.pyplot(fig)
