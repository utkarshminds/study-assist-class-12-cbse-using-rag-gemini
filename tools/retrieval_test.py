from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import pathlib

# Load PDFs
docs=[]
for p in pathlib.Path('lehs1dd').iterdir():
    if p.is_file() and p.suffix=='.pdf':
        docs.extend(PyPDFLoader(str(p)).load())
print('loaded documents:', len(docs))
if not docs:
    raise SystemExit('No PDF documents found in lehs1dd')

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print('created chunks:', len(chunks))

# Build embeddings and vector store in-memory
emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vs = Chroma.from_documents(chunks, embedding=emb)

# Query
query = 'who is james princep'
res = vs.similarity_search(query, k=5)
print('results returned:', len(res))
for i, d in enumerate(res, 1):
    text = d.page_content.replace('\n', ' ')
    print(f'--- Result {i} ---')
    print(text[:1200])
    print()

# Check if exact name exists in any source doc
found = any('james princep' in d.page_content.lower() for d in chunks)
print('exact lowercase "james princep" in any chunk?', found)
