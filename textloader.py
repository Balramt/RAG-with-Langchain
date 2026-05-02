from langchain_community.document_loaders import TextLoader

loader = TextLoader('cricket.txt', None, True)

docs = loader.load()

print(type(docs))

print(len(docs))


print(type(docs[0]))

print(docs[0])

print(docs[0].page_content)
print(docs[0].metadata)