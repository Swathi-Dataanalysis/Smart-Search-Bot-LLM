
# SmartSearch: Research Tool

SmartSearch is an advanced RAG (Retrieve and Generate) application designed to streamline information retrieval from articles across any domain. Whether you’re investigating financial reports, scientific research, industry updates, blogs, or any other topic, SmartSearch allows users to input article URLs and instantly query relevant insights directly from the content. Designed for flexibility, SmartSearch efficiently provides answers and insights regardless of the article's subject matter.

# Objective
- "The primary goal of Smart Search is to enable users to process URLs, retrieve targeted insights, and visualize feedback and performance metrics, all in an interactive, secure, and user-friendly environment."

# Key Features

- URL Processing:Input HTTP article URLs to fetch, process, and analyze content.
- Content Handling:Leverages LangChain's UnstructuredURL Loader to extract and preprocess textual content efficiently.
- Advanced Search:Constructs embedding vectors using OpenAI’s embeddings and performs similarity searches with Chroma, ensuring accurate and quick information retrieval.
- Interactive Querying:Provides detailed answers to user queries, along with source URLs, using GPT-4 and LangChain retrieval chains.
- Feedback Mechanism:Allows users to rate query responses for iterative refinement and better results.
- Persistent Statistics: Tracks the total number of queries processed and ratings submitted during the session, with visualizations.
- Improved Efficiency: SmartSearch identifies duplicate or unchanged URLs to prevent redundant processing, saving resources and time.

# Steps to Navigate

1. Navigate to the project directory:
   cd /Users/rameshchidurala/Desktop/Project/Capstone_Smartsearch_Project

2. Activate environment variables:
   source myenv/bin/activate

3. Install the required dependencies using pip:
  pip install -r requirements.txt

4. Set up your OpenAI API key by creating a .env file in the project root and adding your API key:
    OPENAI_API_KEY=your_api_key_here

# Usage Instructions

1. Run the Streamlit app by executing:
   streamlit run main.py

2. The web app will open in your default browser.

- On the sidebar, input HTTP article URLs to retrieve and process content.

- Click "Process URLs" to initiate loading, preprocessing, chunking, and embedding vector creation.

- Embeddings are stored in Chroma Vectorstore, enhancing retrieval speed and accuracy.

- Enter your query in the text box and receive responses with relevant sources.

# Enhancements

1. Efficient URL Processing:
- URLs are processed only if their content has changed, preventing redundant computations.
- Uses a hashing mechanism to track content changes.
2. Persistent Statistics:
- Tracks and displays the total number of queries processed and ratings submitted.
3. Refined Feedback Mechanism:
- Users can rate answers from 1 to 5.
- Ratings below 4 trigger automatic refinement and re-querying for improved results.
- A rating guide is provided to help users select appropriate feedback.
4. Dynamic Visualizations:
- Feedback ratings are displayed as a colorful bar chart.
- Query response times are plotted to analyze system performance.

# Example URLs
  - https://docs.snowflake.com/en/user-guide/cleanrooms/introduction
  - https://python.langchain.com/docs/introduction/
  - https://www.analyticsvidhya.com/blog/2021/10/everything-you-need-to-know-about-linear-regression/
  - https://learn.microsoft.com/en-us/azure/databricks/delta/
  - https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html
  - https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html
  - https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html

# Research Questions and Hypotheses

## Research Questions
1. How effectively can SmartSearch retrieve relevant information from unstructured web articles?

2. Can iterative user feedback improve the precision and relevance of retrieved results?

3. How scalable is the system when handling diverse datasets and multiple simultaneous users?

## Hypotheses
1. SmartSearch will provide query responses with at least 90% relevance accuracy, based on user feedback.

2. Users will report improved satisfaction (rating of 4/5 or higher) after iterative refinement.

3. The system’s performance will remain stable with multiple queries across different domains.

# Limitations

## HTTP URL Restriction:
- Currently, SmartSearch is limited to processing publicly accessible HTTP URLs. It does not support HTTPS, restricted sites, or dynamically loaded content.

- Dependence on Source Quality:
The relevance and accuracy of results depend on the quality and structure of the source articles. Poorly formatted or overly brief articles may lead to suboptimal embeddings and query results.

- Feedback Implementation:
Refinement depends on user participation and may not work effectively if feedback is inconsistent.

# Data Exploration and Visualization
To enhance understanding, a visual representation of SmartSearch's backend process has been added as an image. This illustration highlights the key steps in the system, including data preprocessing, embedding generation, and query retrieval, providing a clear view of how the components interact.
![alt text](<RAG Workflow.png>)

# Backend Process

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

# Project Structure

main_python.py is python file without UI implementation
app.py: Core script integrating loading, preprocessing, embedding, querying, feedback mechanisms and visualizations with UI implementation
test.py: In this file the urls only process for the first time, if the smart search encounter the same url again it seen before it refer to the previous content only update the changes if any updates made to the article from the last session.
smartsearch_vectorstore/: Directory for storing embedding vectors of app.py
chroma_vectorstore/: Directory for storing embedding vectors of test.py
README.md: Documentation detailing project features, instructions, and limitations.

# Visualization Examples

- Feedback Ratings Distribution:
Displays a bar chart of ratings (1-5) submitted by users.
- Query Response Times:
Plots query response times for performance analysis.


# Note on Langsmith Tracking Configuration
Langsmith Tracking configuration enables LangChain API key integration, which tracks queries submitted to SmartSearch, measures the time taken to retrieve answers, and provides useful insights into query processing. This feature is primarily used for personal development purposes, aiding in project improvement and performance evaluation.

- Optional for Capstone:
For capstone submission purposes, the code runs perfectly fine without the Langsmith Tracking configuration.
If you wish to enable tracking for your own usage, you can obtain a free API token from Langsmith by signing up with your email.

# Langsmith Tracking configuration
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Smart search Chatbot With OPENAI"

# What Makes SmartSearch Unique?
1. Efficient Information Retrieval:
   - Users can input article URLs, and SmartSearch extracts and organizes content into manageable pieces, allowing for precise and efficient querying.
2. Summarization at Your Fingertips:
   - Need a quick overview? SmartSearch generates concise summaries of entire articles, helping you grasp the key points instantly without diving into the full content.
3. Domain-Independent Insights:
   - No matter the subject—be it healthcare, finance, technology, or education—SmartSearch provides accurate, context-aware answers tailored to your queries.
4. Interactive and Flexible:
   - Designed with user feedback in mind, SmartSearch enables you to refine results through interactive feedback loops, improving performance with every interaction.
5. Semantic search approach:
   - Unlike generic search engines, which often rely on keyword matching, SmartSearch leverages semantic search techniques. This means it goes beyond surface-level keyword recognition to truly understand the context and meaning behind your queries. As a result, it retrieves precise answers that are highly relevant and even provides source references for deeper exploration.
6. Secure and Memory-Less:
   - Prioritizing user privacy, SmartSearch retains no session data after it's closed. Your information is secure, and every session starts fresh.

# Who Benefits from SmartSearch?
SmartSearch is perfect for:

- Researchers seeking deep dives into academic content.
- Professionals navigating industry reports or market trends.
- Students requiring clarity on complex topics.
- Everyday users curious about a specific subject or looking for concise summaries.

# The SmartSearch Vision
With SmartSearch, information retrieval is no longer a tedious task. Instead, it’s an intuitive, engaging, and efficient experience—putting the power of accurate, domain-specific insights at your fingertips.