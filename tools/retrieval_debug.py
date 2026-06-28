import pathlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def main():
    docs = []
    data_dir = pathlib.Path("lehs1dd")
    if not data_dir.exists():
        print('Directory lehs1dd not found')
        return

    for item in data_dir.iterdir():
        if item.is_file() and item.suffix.lower() == '.pdf':
            print('Loading', item.name)
            loader = PyPDFLoader(str(item))
            docs.extend(loader.load())

    print('Loaded documents:', len(docs))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(docs)
    print('Created chunks:', len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = './chroma_db_debug'
    vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    print('Vector store created at', persist_dir)

    query = 'who is james princep'
    print('\nRunning similarity search for query:', query)
    results = vs.similarity_search(query, k=5)

    print('\nTop results:')
    for i, doc in enumerate(results, start=1):
        text = doc.page_content.replace('\n', ' ')
        found = 'james princep' in text.lower()
        print(f'-- Result {i}: found_exact={found} len={len(text)}')
        print(text[:1000])
        print('---')

if __name__ == '__main__':
    main()
