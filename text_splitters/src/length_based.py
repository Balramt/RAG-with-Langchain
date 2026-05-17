from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("/home/tiwari/workspace/RAG/text_splitters/files/dl-curriculum.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ""
)

result = splitter.split_documents(docs)

print(result)

print(result[1].page_content)
print(result[1].metadata)