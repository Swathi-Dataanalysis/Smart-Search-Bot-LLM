---
license: apache-2.0
title: Smart Search Bot LLM
sdk: streamlit
emoji: 🚀
colorFrom: yellow
colorTo: green
short_description: Smart Search research tool using LLM
---

# Smart_Search_Bot_LLM

SmartSearch is an advanced RAG (Retrieve and Generate) application designed to streamline information retrieval from articles across any domain. Whether you’re investigating financial reports, scientific research, industry updates, blogs, or any other topic, SmartSearch allows users to input article URLs and instantly query relevant insights directly from the content. Designed for flexibility, SmartSearch efficiently provides answers and insights regardless of the article's subject matter.

# Objective

The primary goal of SmartSearch is to enable users to process URLs, retrieve targeted insights, and visualize feedback and performance metrics, all in an interactive, secure, and user-friendly environment.

# Key Features

1. OpenAI API Key Integration: Users must input their OpenAI API Key in the sidebar to access SmartSearch features securely.
2. URL Processing: Input up to three article URLs to fetch, process, and analyze content.
3. Content Handling: Leverages LangChain's UnstructuredURL Loader to extract and preprocess textual content efficiently.
4. Advanced Search: Constructs embedding vectors using OpenAI’s embeddings and performs similarity searches with Chroma, ensuring accurate and quick information retrieval.
5. Interactive Querying: Provides detailed answers to user queries, along with source URLs, using GPT-4 and LangChain retrieval chains.
6. Feedback Mechanism: Allows users to rate query responses for iterative refinement and better results.
7. Dynamic Visualizations: Tracks feedback ratings and query response times, displaying them as visualizations to analyze system performance.

# Accessing SmartSearch Application

You can directly access the SmartSearch application deployed on Hugging Face Spaces by visiting:

👉 SmartSearch on Hugging Face Spaces: https://huggingface.co/spaces/Swathi97/Smart_Search_Bot_LLM

- To use the application, simply provide your OpenAI API key in the sidebar. No additional setup is required.

# Running SmartSearch Locally

Follow these steps to run SmartSearch on your local computer:

1. Clone this repository to your local machine using:
   
   https://github.com/Swathi-Dataanalysis/Smart-Search-Bot-LLM.git
   
2. Install the required dependencies using pip:

    pip install -r requirements.txt

3. Set up your OpenAI API key:
- Create a .env file in the project root and add your API key:
  
    OPENAI_API_KEY=your_api_key_here
  
4. Run the Streamlit app:
   
   streamlit run app.py
   
5. Access the app in your browser:
 - Enter your OpenAI API Key in the sidebar to enable features.
 - Input up to three HTTP article URLs in the sidebar to retrieve and process content.
 - Click "Process URLs" to initiate loading, preprocessing, chunking, and embedding vector creation.
 - Enter your query in the text box and receive responses with relevant sources.
 - 
# Enhancements

1. API Key Input:
 - Securely input your OpenAI API Key in the sidebar to personalize your experience.
2. Persistent Statistics:
 - Tracks and displays the total number of queries processed and ratings submitted.
3. Refined Feedback Mechanism:
 - Users can rate answers from 1 to 5.
 - Ratings below 4 trigger automatic refinement and re-querying for improved results.
4. Dynamic Visualizations:
 - Feedback ratings are displayed as a colorful bar chart.
 - Query response times are plotted to analyze system performance.

# Limitations

1. Reprocessing URLs:
 - Every time a URL is processed, its content is fetched and analyzed again, even if it was previously seen.
2. HTTP URL Restriction:
 - Currently, SmartSearch is limited to processing publicly accessible HTTP URLs. It does not support HTTPS, restricted sites, or 
   dynamically loaded content.
3. Dependence on Source Quality:
 - The relevance and accuracy of results depend on the quality and structure of the source articles. Poorly formatted or overly brief 
   articles may lead to suboptimal embeddings and query results.
4. Feedback Implementation:
 - Refinement depends on user participation and may not work effectively if feedback is inconsistent.

# Project Structure

1. main_python.py: Python file without UI implementation.
2. app.py: Core script integrating loading, preprocessing, embedding, querying, feedback mechanisms, and visualizations with UI implementation.
3. test.py: Processes URLs but focuses on checking for changes since the last session.
4. chroma_vectorstore/: Directory for storing embedding vectors.
5. README.md: Documentation detailing project features, instructions, and limitations.

# YAML File: Purpose and Creation (needed for hugging face deploymeny, if you running your application locally you don't need to worry about YAML file)

## Purpose of the YAML File
A YAML file is used for defining metadata, configurations, and settings for the project. For Hugging Face Spaces:

- It specifies the application entry point, SDK, title, license and other parameters.
- It helps Hugging Face correctly configure and display the project.
--- How to Create a YAML File ---
# Add a metadata block at the top of the README.md file:
# Refer to the beginning of the readme file 
Save this metadata as part of the README.md file. # like above
Hugging Face YAML Documentation
For more details, refer to the Hugging Face documentation: Hugging Face Spaces YAML Guide.

# Other details 
## Backend Process

The backend process of SmartSearch consists of several stages, designed to efficiently load, process, store, and retrieve information from user-provided URLs or documents. Below is a detailed explanation of each stage, as illustrated in the provided image:

1. Document Loading

- Action:The system accepts URLs (HTTP/HTTPS), PDFs, or other documents as input.

- Documents are retrieved using LangChain's UnstructuredURLLoader or similar tools.

- Purpose:To fetch raw data from various sources and prepare it for further processing.

- Outcome:The raw content is successfully loaded and passed to the next stage for splitting.

2. Splitting

- Action:The loaded documents are split into manageable chunks using RecursiveCharacterTextSplitter.

- Chunking is configured with overlap to ensure that context is preserved across splits.

- Purpose:To break down large documents into smaller, meaningful sections for efficient embedding and retrieval.

- Outcome:The content is divided into smaller, contextually coherent pieces, ready for vectorization.

3. Storage

- Action:The text chunks are converted into semantic embeddings using OpenAI embeddings.

- These embeddings are stored in a Chroma Vectorstore, optimized for high-speed retrieval during user queries.

- Purpose:To prepare and index the data for fast similarity-based searches.
- Outcome:A vectorized representation of the content is stored securely, enabling efficient query matching.
4. Retrieval

- Action:When a user submits a query, it is also converted into an embedding.

- The query embedding is compared against the stored embeddings in the Vectorstore to retrieve the most relevant text chunks.

- Purpose:To identify and retrieve content that is semantically aligned with the user's query.

- Outcome:Relevant chunks of data are selected for generating the final response.

5. Output

Action:
- The retrieved chunks are passed to GPT-4 via LangChain's RetrievalQAWithSourcesChain.

- The model generates a detailed and contextually relevant answer based on the retrieved data.

- The output includes the response and the source references.

- Purpose:To generate accurate and well-referenced answers to user queries.

-Outcome:A detailed response is displayed to the user, along with citations for transparency.

6. Feedback Loop

- Action:After receiving the response, users can rate it on a scale of 1-5.

- Ratings below 4 trigger an iterative refinement process, where the query is adjusted for improved results.

- Refinement includes analyzing user feedback and re-querying the backend with additional context.

- Purpose:To continuously improve response quality based on user satisfaction.
- Outcome:Users receive progressively better answers, and the system adapts based on feedback.

# Note on Langsmith Tracking Configuration
Langsmith Tracking configuration enables LangChain API key integration, which tracks queries submitted to SmartSearch, measures the time taken to retrieve answers, and provides useful insights into query processing. This feature is primarily used for personal development purposes, aiding in project improvement and performance evaluation.

- Optional for Capstone:
For capstone submission purposes, the code runs perfectly fine without the Langsmith Tracking configuration.
If you wish to enable tracking for your own usage, you can obtain a free API token from Langsmith by signing up with your email.

# Langsmith Tracking configuration
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Smart search Chatbot With OPENAI"
