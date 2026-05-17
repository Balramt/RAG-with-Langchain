from langchain_community.document_loaders import CSVLoader


loader = CSVLoader(file_path = "/home/tiwari/workspace/RAG/document_loader/files/Social_Network_Ads.csv")

docs = loader.load()

print(len(docs))  # Each row  of csv file act as document object hence = 400

print(docs[1].page_content)
print(docs[1].metadata)