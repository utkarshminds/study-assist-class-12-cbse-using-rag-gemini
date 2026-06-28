# RAG Chatbot with Streamlit and Gemini LLM

This project implements a Retrieval-Augmented Generation (RAG) based chatbot using Streamlit for the UI, Google's Gemini LLM for conversational AI, and Chroma DB for vector storage. The chatbot is designed to answer questions based on a provided set of PDF documents.

## Features

- **Streamlit Deployment**: Easy to deploy and interact with via a web interface.
- **User Authentication**: Simple username/password login to protect access.
- **Secret Management**: Secure handling of API keys and credentials using Streamlit's `st.secrets`.
- **Chroma DB**: Efficient vector storage for document embeddings.
- **PDF Knowledge Base**: Extracts information from PDF documents located in a specified folder.
- **Contextual Responses**: Answers questions based on the content of the PDF documents.
- **Out-of-Scope Handling**: Politely declines to answer if the information is not found in the knowledge base.

## Setup Instructions

Follow these steps to set up and run the chatbot locally:

### 1. Clone the Repository

```bash
git clone <repository_url>
cd study-assist-class-12-cbse-using-rag-gemini
```

### 2. Set up Python Environment

It's recommended to use a virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate   # On Windows
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Secrets

**a. `secrets/secrets.toml` for Streamlit Login:**

Create a file named `secrets.toml` inside the `secrets/` directory with the following content. These credentials will be used for logging into the Streamlit application.

```toml
# secrets/secrets.toml
[streamlit]
username = "your_username"
password = "your_password"
```

**b. `.env` for Gemini API Key:**

Create a `.env` file in the root directory of your project (the same directory as `app.py`) and add your Google Gemini API key. You can obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

```
# .env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

**Important:** The `secrets/` folder and `.env` file are already added to `.gitignore` to prevent them from being committed to version control.

### 5. Add PDF Documents

Place your PDF knowledge base documents into the `lehs1dd/` directory. The chatbot will load information from these PDF files.

```
study-assist-class-12-cbse-using-rag-gemini/
├── app.py
├── requirements.txt
├── .gitignore
├── .env
├── lehs1dd/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
└── secrets/
    └── secrets.toml
```

### 6. Run the Streamlit Application

Once all configurations are in place, run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

This will open the chatbot in your web browser. You will be prompted to log in using the username and password configured in `secrets/secrets.toml`.

## How it Works

1.  **Document Loading**: PDFs from the `lehs1dd/` directory are loaded using `PyPDFLoader`.
2.  **Text Splitting**: The loaded documents are split into smaller, manageable text chunks using `RecursiveCharacterTextSplitter`.
3.  **Embeddings**: `HuggingFaceEmbeddings` (specifically `sentence-transformers/all-MiniLM-L6-v2`) are used to convert text chunks into numerical vector representations.
4.  **Vector Store**: These embeddings are stored in `Chroma DB` for efficient retrieval.
5.  **Conversational Chain**: A `ConversationalRetrievalChain` is set up with `ChatGoogleGenerativeAI` (Gemini Pro model) and a `ConversationBufferMemory` to maintain chat history.
6.  **Question Answering**: When a user asks a question, the relevant document chunks are retrieved from Chroma DB, and along with the chat history, are passed to the Gemini LLM to generate a coherent answer.
7.  **Out-of-Scope Handling**: If the LLM indicates that the answer is not within its knowledge base (i.e., the provided documents), a polite refusal message is displayed.

## Customization

-   **Login Credentials**: Modify `secrets/secrets.toml` to change the default username and password.
-   **Gemini API Key**: Update the `GOOGLE_API_KEY` in your `.env` file.
-   **PDF Directory**: If your PDF documents are in a different location, update the `load_docs` function call in `app.py`.
-   **Embedding Model**: You can change the `HuggingFaceEmbeddings` model in `get_vector_store` function if needed.
-   **LLM Model**: The `ChatGoogleGenerativeAI` model can be changed in `get_conversational_chain`.

