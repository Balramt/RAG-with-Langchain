from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls= PyPDFLoader
)

docs = loader.load()

print(len(docs))
print(docs[327].page_content)
print(docs[327].metadata)


for document in docs:          # Here we wait to load all data file and then print in once
    print(document.metadata)



lazy_docs = loader.lazy_load()   # Here it lodas one by one and print one by one 

for document in lazy_docs:
    print(document.metadata)