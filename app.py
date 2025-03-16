import streamlit as st
import tempfile

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings #Import the correct embedding.

# Streamlit UI Title
st.title("🔍 Chat with Search & PDF using LangChain and GROQ!")

st.write("Upload a PDF or ask me anything. I can search Wikipedia, Arxiv, and DuckDuckGo.")

# Sidebar for API Key
st.sidebar.title("⚙️ Settings")
groq_api_key = st.sidebar.text_input("Enter your GROQ API Key:", type="password")

# Validate Groq API key before proceeding
if groq_api_key:
    try:
        # Proper validation: Initialize and make a simple call
        test_llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        _ = test_llm.invoke("Hello")  # Send a simple test query

        st.sidebar.success("✅ GROQ API Key loaded successfully!")
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", streaming=True)
    except Exception as e:
        st.sidebar.error(f"❌ Invalid GROQ API Key: {str(e)}")
        groq_api_key = None  # Reset key
else:
    st.sidebar.warning("⚠️ Please enter your GROQ API Key to proceed.")

# Check if API keys are available
if groq_api_key is None:
    st.stop()

# Upload PDF & Process It
uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    st.success("✅ PDF Uploaded Successfully!")

    # Load and split the PDF into smaller chunks
    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # Create vector database using FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Store vector database in session for retrieval
    st.session_state.vectors = vectorstore
    st.success("✅ PDF Processed and Indexed for Q&A!")

# Chatbot: Chat History Setup
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hi, I can answer questions from PDFs or search the web. How can I help?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Define Search Tools (Wikipedia, Arxiv, DuckDuckGo)
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_runner = ArxivQueryRun(name="arxiv-runner", arxiv_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_runner = WikipediaQueryRun(name="wiki-runner", api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="search")

tools = [search, arxiv_runner, wiki_runner]

# Chat Input and Retrieval Logic
user_prompt = st.chat_input(placeholder="Type your question here...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)  # store in chat history

    # Always search the web but include PDF context if available
    if "vectors" in st.session_state:
        retriever = st.session_state.vectors.as_retriever()
        retrieved_docs = retriever.invoke(user_prompt)

        # Extract relevant text from PDF (Top 3 relevant passages)
        pdf_context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])
    else:
        pdf_context = "No PDF available."

    # Final prompt combining PDF + Web Search
    final_prompt = f"""
    Use the following PDF context if relevant, otherwise search online.

    PDF Context:
    {pdf_context}

    User Question: {user_prompt}

    If the PDF does not contain the answer, perform an online search.
    """

    # Always conduct an internet search
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

    # Run the search agent with the final combined prompt
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = search_agent.run(final_prompt, callbacks=[st_cb])

    # Store response in chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)