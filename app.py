import streamlit as st
import os
# ...existing code...
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
import pathlib

# load_dotenv() # Not needed if using st.secrets

st.set_page_config(page_title="RAG Chatbot with Gemini")
st.title("RAG Chatbot with Gemini LLM")

# Placeholder for login logic
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def check_password():
    """Returns `True` if the user enters the correct password."""

    def password_entered():
        if (st.session_state["username"] == st.secrets["streamlit"]["username"] and
            st.session_state["password"] == st.secrets["streamlit"]["password"]):
            st.session_state["logged_in"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        if not st.session_state["logged_in"] and st.session_state["username"]:
            st.error("Invalid username/password")
        return False
    else:
        return True

if check_password():
    st.write("Welcome to the RAG Chatbot!")

    # Initialize RAG components
    if "conversation" not in st.session_state:
        with st.spinner("Processing documents..."):
            documents = load_docs("lehs1dd")
            text_chunks = get_text_chunks(documents)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation = get_conversational_chain(vector_store)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = st.session_state.conversation({"question": prompt})
        answer = response.get("answer") or response.get("chat_history", [])[-1].content

        if "I don't know" in answer or "not in the provided documents" in answer:
            answer = "I apologize, but I can only answer questions based on the provided documents. I don't know the answer to that question based on my current knowledge base."

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)

            source_docs = response.get("source_documents", [])[:3]
            if source_docs:
                st.markdown("**Retrieved document chunks:**")
                for idx, doc in enumerate(source_docs, start=1):
                    snippet = doc.page_content.strip().replace("\n", " ")
                    if len(snippet) > 500:
                        snippet = snippet[:500].rstrip() + "..."
                    st.markdown(f"**Chunk {idx}:** {snippet}")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
# os.environ["GOOGLE_API_KEY"] = st.secrets["gemini"]["api_key"] # This line is for Streamlit Cloud deployment

# Local .env file setup for GOOGLE_API_KEY
# Create a .env file in the root directory with the following content:
# GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

# Function to load PDF documents
def load_docs(directory):
    documents = []
    for item in pathlib.Path(directory).iterdir():
        if item.is_file() and item.suffix == '.pdf':
            pdf_path = str(item)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    return documents

# Function to split text into chunks
def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to create/get vector store
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory="./chroma_db")
    return vector_store

# Function to get conversational chain
def get_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=st.secrets["gemini"]["api_key"])
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain

