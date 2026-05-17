from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

model = ChatOllama(model="llama3.2:1b")

load_dotenv()

prompt = PromptTemplate(
    template= "write the summary for the following poem -\n {poem}",
    input_variables=["poem"]
)


parser = StrOutputParser()
loader = TextLoader('../files/cricket.txt', None, True)

docs = loader.load()

print(type(docs))

print(len(docs))

print(type(docs[0]))

print(docs[0])

print(docs[0].page_content)
print(docs[0].metadata)


chain = prompt | model| parser

print(chain.invoke({"poem":docs[0].page_content}))